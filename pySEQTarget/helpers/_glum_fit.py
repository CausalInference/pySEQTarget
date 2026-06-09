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

    def __init__(self, glum_model, design_info, feature_names, X_design, sample_weight):
        self._glum = glum_model
        self._design_info = design_info
        self._X_design = X_design  # includes the intercept column
        self._sample_weight = sample_weight

        self.model = types.SimpleNamespace(
            exog_names=feature_names,
            data=types.SimpleNamespace(design_info=design_info),
        )
        self.exog_names = feature_names

        # statsmodels convention: intercept first
        all_coefs = np.concatenate([[glum_model.intercept_], glum_model.coef_])
        self.params = pd.Series(all_coefs, index=feature_names)

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
        X = self._X_design
        mu = self._glum.predict(X[:, 1:])
        w = mu * (1.0 - mu)
        if self._sample_weight is not None:
            w = w * np.asarray(self._sample_weight)
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
                    "GLM (glum backend)",
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
    return _GlumFit(glm, design_info, feature_names, X_design, sample_weight)
