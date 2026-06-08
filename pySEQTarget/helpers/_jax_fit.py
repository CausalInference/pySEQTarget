import types

import numpy as np
import pandas as pd
import patsy
import polars as pl

from ._jax_logistic import MultinomialLogisticRegression


class _JaxFit:
    def __init__(
        self,
        formula,
        df,
        var_weights=None,
        learning_rate=0.1,
        num_epochs=2000,
        start_params=None,
    ):
        df_pd = df.to_pandas() if isinstance(df, pl.DataFrame) else df
        y_mat, X_mat = patsy.dmatrices(formula, df_pd, return_type="dataframe")

        design_info = X_mat.design_info
        feature_names = list(X_mat.columns)
        self._design_info = design_info
        self._feature_names = feature_names
        self._X_design = X_mat.values
        X_arr = X_mat.drop(columns=["Intercept"], errors="ignore").values

        y_raw = y_mat.values.ravel()
        self.classes_ = np.unique(y_raw)
        self._n_classes = len(self.classes_)
        y_idx = np.searchsorted(self.classes_, y_raw)

        self._sample_weight = None
        if var_weights is not None:
            self._sample_weight = np.asarray(var_weights, dtype=float)

        jax_model = MultinomialLogisticRegression(
            learning_rate=learning_rate,
            num_epochs=num_epochs,
            n_classes=self._n_classes,
        )
        
        jax_model.mean_ = X_arr.mean(axis=0)
        jax_model.std_ = X_arr.std(axis=0) + jax_model.eps  # guard constants
        self._jax = jax_model

        init_params = self._warm_init(start_params, X_arr.shape[1])
        jax_model.fit(
            X_arr, y_idx, sample_weight=self._sample_weight, init_params=init_params
        )

        # statsmodels 'like' exposure
        self.model = types.SimpleNamespace(
            exog_names=feature_names,
            data=types.SimpleNamespace(design_info=design_info),
        )
        self.exog_names = feature_names
        self.params = self._build_params()

    def _coef_components(self):
        W, b = self._jax.params
        W = np.asarray(W)
        b = np.asarray(b)
        coef = W[:, 1:] - W[:, :1]
        intercept = b[1:] - b[0]
        mean = np.asarray(self._jax.mean_)
        std = np.asarray(self._jax.std_)
        intercept = intercept - (coef * (mean / std)[:, None]).sum(axis=0)
        coef = coef / std[:, None]

        return intercept, coef

    def _build_params(self):
        intercept, coef = self._coef_components()
        if self._n_classes == 2:
            return pd.Series(
                np.concatenate([intercept, coef[:, 0]]), index=self._feature_names
            )
        data = np.vstack([intercept[None, :], coef])
        return pd.DataFrame(
            data,
            index=self._feature_names,
            columns=[f"class_{c}" for c in self.classes_[1:]],
        )

    def _warm_init(self, start_params, n_features):
        if start_params is None or self._n_classes != 2:
            return None
        sp_values, sp_names = start_params
        if list(sp_names) != self._feature_names:
            return None
        sp_values = np.asarray(sp_values, dtype=float)
        intercept = float(sp_values[0])
        coef = sp_values[1:]
        if coef.shape[0] != n_features:
            return None
        W = np.zeros((n_features, 2))
        b = np.zeros(2)
        mean = np.asarray(self._jax.mean_)
        std = np.asarray(self._jax.std_)
        W[:, 1] = coef * std
        b[1] = intercept + float(np.sum(coef * mean))

        return (W, b)

    def predict(self, data, transform=True):
        if transform:
            data_pd = data.to_pandas() if isinstance(data, pl.DataFrame) else data
            X = patsy.build_design_matrices(
                [self._design_info], data_pd, return_type="dataframe"
            )[0]
            X_arr = X.drop(columns=["Intercept"], errors="ignore").values
        else:
            X_arr = np.asarray(data)[:, 1:]
        probs = np.asarray(self._jax.predict(X_arr), dtype=np.float64)
        return probs[:, 1] if self._n_classes == 2 else probs

    def cov_params(self):
        if self._n_classes != 2:
            raise NotImplementedError(
                "Standard errors are only implemented for binary jax fits."
            )
        X = self._X_design
        mu = np.asarray(self._jax.predict(X[:, 1:]))[:, 1]
        w = mu * (1.0 - mu)
        if self._sample_weight is not None:
            w = w * self._sample_weight
        return np.linalg.pinv(X.T @ (w[:, None] * X))

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
                    "GLM (jax backend)",
                    "Binomial",
                    "logit",
                    str(self._X_design.shape[0]),
                ]
            },
            index=["Model:", "Family:", "Link:", "No. Observations:"],
        )
        smry = Summary()
        smry.add_title("Generalized Linear Model Regression Results")
        smry.add_df(info, header=False)
        smry.add_df(self._coef_table())
        return smry


def _fit_jax(formula, data, var_weights=None, start_params=None):
    return _JaxFit(formula, data, var_weights=var_weights, start_params=start_params)
