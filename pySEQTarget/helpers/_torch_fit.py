import types

import numpy as np
import pandas as pd
import patsy
import polars as pl
import torch
from scipy.special import softmax


# weighted multinomial logistic Newton steps with class 0 as reference
def _newton(X, y, w, B, max_iter, ridge=1e-8):
    Y = torch.nn.functional.one_hot(y, B.shape[1] + 1)[:, 1:].to(X.dtype)
    w = w / w.sum()
    m = B.numel()
    eye = torch.eye(m, dtype=X.dtype, device=X.device)
    for _ in range(max_iter):
        eta = X @ B
        p = torch.softmax(torch.cat([torch.zeros_like(eta[:, :1]), eta], 1), 1)[:, 1:]
        g = X.T @ (w[:, None] * (p - Y))
        V = torch.diag_embed(p) - p[:, :, None] * p[:, None, :]
        H = torch.einsum("n,nj,nab,nk->jakb", w, X, V, X).reshape(m, m)
        B = B - torch.linalg.solve(H + ridge * eye, g.reshape(-1)).reshape(B.shape)
    return B


class _TorchFit:
    def __init__(
        self,
        formula,
        df,
        var_weights=None,
        max_iter=25,
        start_params=None,
    ):
        df_pd = df.to_pandas() if isinstance(df, pl.DataFrame) else df
        y_mat, X_mat = patsy.dmatrices(formula, df_pd, return_type="dataframe")

        design_info = X_mat.design_info
        feature_names = list(X_mat.columns)
        self._design_info = design_info
        self._feature_names = feature_names
        self._X_design = X_mat.values
        self._nobs = X_mat.shape[0]
        # cached covariance lets pickling drop the design matrix
        self._cov_cached = None
        # formula and reference frame rebuild the unpicklable patsy DesignInfo
        self._formula = formula
        self._ref_frame = df_pd.head(2).copy()
        X_arr = X_mat.drop(columns=["Intercept"], errors="ignore").values

        y_raw = y_mat.values.ravel()
        self.classes_ = np.unique(y_raw)
        self._n_classes = len(self.classes_)
        y_idx = np.searchsorted(self.classes_, y_raw)

        self._sample_weight = None
        if var_weights is not None:
            self._sample_weight = np.asarray(var_weights, dtype=float)

        # fit on standardised columns with constants guarded
        mean = X_arr.mean(axis=0)
        std = X_arr.std(axis=0) + 1e-7
        Z = np.column_stack([np.ones(len(X_arr)), (X_arr - mean) / std])
        B = self._warm_init(start_params, mean, std)
        if B is None:
            B = np.zeros((Z.shape[1], self._n_classes - 1))
        w = self._sample_weight if var_weights is not None else np.ones(len(Z))

        device = "cuda" if torch.cuda.is_available() else "cpu"

        def t(a, dtype=torch.float64):
            return torch.as_tensor(a, dtype=dtype, device=device)

        B = _newton(t(Z), t(y_idx, torch.long), t(w), t(B), max_iter).cpu().numpy()
        # back to the original scale
        self._coef = B[1:] / std[:, None]
        self._intercept = B[0] - mean @ self._coef

        # statsmodels like interface
        self._build_model_namespace(design_info, feature_names)
        self.exog_names = feature_names
        self.params = self._build_params()

    def _build_model_namespace(self, design_info, feature_names):
        self.model = types.SimpleNamespace(
            exog_names=feature_names,
            data=types.SimpleNamespace(design_info=design_info),
        )

    def __getstate__(self):
        # drop the DesignInfo and design matrix but keep the covariance
        state = self.__dict__.copy()
        state.pop("_design_info", None)
        state.pop("model", None)
        if state.get("_cov_cached") is None and self._n_classes == 2:
            state["_cov_cached"] = self.cov_params()
        state["_X_design"] = None
        return state

    def __setstate__(self, state):
        self.__dict__.update(state)
        _, X_mat = patsy.dmatrices(
            self._formula, self._ref_frame, return_type="dataframe"
        )
        if list(X_mat.columns) != list(self._feature_names):
            # coefficients would pair with the wrong columns
            raise RuntimeError(
                "_TorchFit design columns changed on unpickle: "
                f"{list(X_mat.columns)} != {list(self._feature_names)}"
            )
        self._design_info = X_mat.design_info
        self._build_model_namespace(self._design_info, self._feature_names)

    def _build_params(self):
        if self._n_classes == 2:
            return pd.Series(
                np.concatenate([self._intercept, self._coef[:, 0]]),
                index=self._feature_names,
            )
        data = np.vstack([self._intercept[None, :], self._coef])
        return pd.DataFrame(
            data,
            index=self._feature_names,
            columns=[f"class_{c}" for c in self.classes_[1:]],
        )

    def _warm_init(self, start_params, mean, std):
        if start_params is None or self._n_classes != 2:
            return None
        sp_values, sp_names = start_params
        if list(sp_names) != self._feature_names:
            return None
        sp_values = np.asarray(sp_values, dtype=float)
        coef = sp_values[1:]
        if coef.shape[0] != mean.shape[0]:
            return None
        # standardised scale with the intercept first
        return np.concatenate([[sp_values[0] + coef @ mean], coef * std])[:, None]

    def _probs(self, X_arr):
        eta = self._intercept + X_arr @ self._coef
        return softmax(np.column_stack([np.zeros(len(eta)), eta]), axis=1)

    def predict(self, data, transform=True):
        if transform:
            data_pd = data.to_pandas() if isinstance(data, pl.DataFrame) else data
            X = patsy.build_design_matrices(
                [self._design_info], data_pd, return_type="dataframe"
            )[0]
            X_arr = X.drop(columns=["Intercept"], errors="ignore").values
        else:
            X_arr = np.asarray(data)[:, 1:]
        probs = self._probs(X_arr)
        return probs[:, 1] if self._n_classes == 2 else probs

    def remove_data(self):
        # covariance is binary only
        if self._n_classes == 2:
            self.cov_params()
        self._X_design = None
        self._sample_weight = None

    def cov_params(self):
        if self._n_classes != 2:
            raise NotImplementedError(
                "Standard errors are only implemented for binary torch fits."
            )
        if self._cov_cached is not None:
            return self._cov_cached
        X = self._X_design
        if X is None:
            raise RuntimeError(
                "cov_params unavailable: design matrix was dropped on pickle "
                "and no covariance was cached."
            )
        mu = self._probs(X[:, 1:])[:, 1]
        w = mu * (1.0 - mu)
        if self._sample_weight is not None:
            w = w * self._sample_weight
        self._cov_cached = np.linalg.pinv(X.T @ (w[:, None] * X))
        return self._cov_cached

    @property
    def bse(self):
        return pd.Series(np.sqrt(np.diag(self.cov_params())), index=self.params.index)

    def _coef_table(self):
        from scipy import stats

        coef = self.params.values
        se = self.bse.values
        with np.errstate(divide="ignore", invalid="ignore"):
            z = coef / se
        pvals = 2.0 * stats.norm.sf(np.abs(z))
        crit = stats.norm.ppf(0.975)
        return pd.DataFrame(
            {
                "Coef.": coef,
                "Std.Err.": se,
                "z": z,
                "P>|z|": pvals,
                "[0.025": coef - crit * se,
                "0.975]": coef + crit * se,
            },
            index=list(self.params.index),
        )

    def summary(self):
        from statsmodels.iolib.summary2 import Summary

        info = pd.DataFrame(
            {
                " ": [
                    "GLM (torch backend)",
                    "Binomial",
                    "logit",
                    str(self._nobs),
                ]
            },
            index=["Model:", "Family:", "Link:", "No. Observations:"],
        )
        smry = Summary()
        smry.add_title("Generalized Linear Model Regression Results")
        smry.add_df(info, header=False)
        smry.add_df(self._coef_table())
        return smry


def _fit_torch(formula, data, var_weights=None, start_params=None):
    return _TorchFit(formula, data, var_weights=var_weights, start_params=start_params)
