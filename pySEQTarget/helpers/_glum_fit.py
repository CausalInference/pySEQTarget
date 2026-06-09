import types

import numpy as np
import pandas as pd
import patsy
from glum import GeneralizedLinearRegressor

from ._fix_categories import _fix_categories_for_predict


def _align_categories(design_info, data):
    """
    Re-align ``data``'s categorical columns to the level set and ORDER frozen
    in ``design_info``. Wraps ``_fix_categories_for_predict`` (which expects a
    model-like object) so the cached design info can be re-applied to a
    bootstrap resample whose categoricals materialised in a different order.
    """

    class _Stub:
        class model:
            class data:
                pass

    stub = _Stub()
    stub.model.data.design_info = design_info
    return _fix_categories_for_predict(stub, data)


class _GlumFit:
    """
    Wraps a fitted glum model exposing the statsmodels interface the rest of
    the codebase (and users) expect:
      .params (Series), .model.exog_names, .model.data.design_info,
      .predict(df) / .predict(X_numpy, transform=False),
      .bse, .summary().

    Standard errors are derived lazily from the stored design matrix using the
    GLM asymptotic covariance (X' W X)^-1, which matches statsmodels for the
    binomial/logit family (incl. var_weights). The design matrix is retained
    just like statsmodels keeps model.exog, so memory use is comparable.
    """

    def __init__(
        self,
        glum_model,
        design_info,
        feature_names,
        X_design,
        sample_weight,
        formula=None,
        ref_frame=None,
    ):
        self._glum = glum_model
        self._design_info = design_info
        self._X_design = X_design  # includes the intercept column
        self._nobs = X_design.shape[0]
        self._sample_weight = sample_weight
        # Lazily-filled cache of the (small) coefficient covariance matrix. It
        # lets __getstate__ drop the full design matrix (_X_design can be 100s
        # of MB) while keeping bse/summary working after unpickle — important
        # for the process pool and offload, which ship many fitted models.
        self._cov_cached = None
        # Inputs to rebuild ``design_info`` on unpickle: the patsy DesignInfo
        # itself cannot be pickled (patsy #26), so we keep the formula and a
        # tiny reference frame (which preserves each categorical column's full,
        # ordered dtype categories) and re-parse on __setstate__.
        self._formula = formula
        self._ref_frame = ref_frame

        self._build_model_namespace(design_info, feature_names)
        self.exog_names = feature_names

        # statsmodels convention: intercept first
        all_coefs = np.concatenate([[glum_model.intercept_], glum_model.coef_])
        self.params = pd.Series(all_coefs, index=feature_names)

    def _build_model_namespace(self, design_info, feature_names):
        self.model = types.SimpleNamespace(
            exog_names=feature_names,
            data=types.SimpleNamespace(design_info=design_info),
        )

    def __getstate__(self):
        # Drop the unpicklable patsy DesignInfo and the SimpleNamespaces that
        # reference it; __setstate__ rebuilds them from the formula + ref_frame.
        state = self.__dict__.copy()
        state.pop("_design_info", None)
        state.pop("model", None)
        # Replace the full design matrix with the small cached covariance so the
        # pickled model stays lightweight (the design matrix can be 100s of MB).
        # bse/summary still work via _cov_cached; predict never needs _X_design.
        if state.get("_cov_cached") is None:
            state["_cov_cached"] = self.cov_params()
        state["_X_design"] = None
        return state

    def __setstate__(self, state):
        self.__dict__.update(state)
        if self._formula is None or self._ref_frame is None:
            raise RuntimeError(
                "Cannot unpickle _GlumFit fitted before formula/ref_frame were "
                "recorded; refit with the current pySEQTarget version."
            )
        _, X_mat = patsy.dmatrices(
            self._formula, self._ref_frame, return_type="dataframe"
        )
        if list(X_mat.columns) != list(self.exog_names):
            # The reference frame's categorical ordering must reproduce the
            # frozen column structure exactly, or glum's coefficients would be
            # paired with the wrong design columns on predict. Fail loudly
            # rather than return silently wrong predictions.
            raise RuntimeError(
                "_GlumFit design columns changed on unpickle: "
                f"{list(X_mat.columns)} != {list(self.exog_names)}"
            )
        self._design_info = X_mat.design_info
        self._build_model_namespace(self._design_info, self.exog_names)

    def predict(self, data, transform=True):
        if transform:
            # data is a pandas DataFrame — build design matrix via stored patsy info
            X = patsy.build_design_matrices(
                [self._design_info], data, return_type="dataframe"
            )[0]
            X_arr = X.drop(columns=["Intercept"], errors="ignore").values
        else:
            # data is a pre-built numpy design matrix (includes intercept col — drop it)
            X_arr = np.asarray(data)[:, 1:]
        return self._glum.predict(X_arr)

    def cov_params(self):
        if self._cov_cached is not None:
            return self._cov_cached
        X = self._X_design
        if X is None:
            raise RuntimeError(
                "cov_params unavailable: design matrix was dropped on pickle and "
                "no covariance was cached."
            )
        mu = self._glum.predict(X[:, 1:])
        w = mu * (1.0 - mu)
        if self._sample_weight is not None:
            w = w * np.asarray(self._sample_weight)
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
                    "GLM (glum backend)",
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


def _fit_glum(formula, data, var_weights=None, start_params=None, design_cache=None):
    """Fit a binomial GLM with glum and return a _GlumFit wrapper.

    ``start_params`` is the cached ``(values, names)`` tuple from a previous fit,
    used as a warm-start in the bootstrap loop. It is only honoured when the
    design matrix columns line up exactly with ``names`` - a bootstrap resample
    can drop a categorical level and shift the column structure, in which case
    the cached coefs are meaningless and using them as init would derail the
    coordinate-descent solver.

    ``design_cache`` is an optional ``dict`` keyed by ``formula``. On a hit, the
    formula parse and patsy model.frame construction are skipped and the cached
    ``(y_design_info, X_design_info)`` are re-applied to ``data`` via
    ``patsy.build_design_matrices``. On a miss, ``patsy.dmatrices`` parses the
    formula and the result is stored. Caching freezes the categorical encoding
    to the main fit's column structure, which also makes the warm-start
    guarantee trivially satisfied for every replicate.
    """
    if design_cache is not None and formula in design_cache:
        y_dinfo, x_dinfo = design_cache[formula]
        try:
            y_mat, X_mat = patsy.build_design_matrices(
                [y_dinfo, x_dinfo], data, return_type="dataframe"
            )
        except patsy.PatsyError as e:
            if "mismatching levels" not in str(e):
                raise
            # A bootstrap resample can realise the same categorical levels in a
            # different ORDER than the cached design_info froze. Re-align the
            # categories to the cached structure and retry, so the cached column
            # layout (and the warm-start that relies on it) stays valid.
            data = _align_categories(x_dinfo, data.copy())
            y_mat, X_mat = patsy.build_design_matrices(
                [y_dinfo, x_dinfo], data, return_type="dataframe"
            )
    else:
        y_mat, X_mat = patsy.dmatrices(formula, data, return_type="dataframe")
        if design_cache is not None:
            design_cache[formula] = (y_mat.design_info, X_mat.design_info)

    y_arr = y_mat.values.ravel()
    design_info = X_mat.design_info
    feature_names = list(X_mat.columns)  # "Intercept" first, then predictors
    X_design = X_mat.values  # includes intercept column (for covariance)
    X_arr = X_mat.drop(columns=["Intercept"]).values

    init = None
    if start_params is not None:
        sp_values, sp_names = start_params
        if list(sp_names) == feature_names:
            init = np.asarray(sp_values, dtype=float)

    glm = GeneralizedLinearRegressor(
        family="binomial", fit_intercept=True, start_params=init
    )

    sample_weight = None
    fit_kwargs = {}
    if var_weights is not None:
        sample_weight = np.asarray(var_weights)
        fit_kwargs["sample_weight"] = sample_weight

    glm.fit(X_arr, y_arr, **fit_kwargs)

    # Keep a minimal reference frame so the (unpicklable) design_info can be
    # rebuilt on unpickle. Two rows suffice: patsy derives categorical contrasts
    # from each column's full dtype categories, not the observed values, and the
    # codebase uses only stateless transforms (precomputed squares, explicit-knot
    # splines), so no fit-time state needs preserving.
    ref_frame = data.head(2).copy()
    return _GlumFit(
        glm,
        design_info,
        feature_names,
        X_design,
        sample_weight,
        formula=formula,
        ref_frame=ref_frame,
    )
