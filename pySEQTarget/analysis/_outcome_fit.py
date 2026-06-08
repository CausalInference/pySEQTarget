import re

import numpy as np
import pandas as pd
import polars as pl
import statsmodels.api as sm
import statsmodels.formula.api as smf
from pandas.api.types import is_numeric_dtype


def _compute_spline_knots(followup_arr, df=3):
    lower = float(np.min(followup_arr))
    upper = float(np.max(followup_arr))
    n_inner = df - 2
    if n_inner == 0:
        inner_knots = []
    else:
        # Replicate patsy's knot placement: percentiles of unique values in [lower, upper]
        x = np.unique(followup_arr[(lower <= followup_arr) & (followup_arr <= upper)])
        q = np.linspace(0, 100, n_inner + 2)[1:-1]
        inner_knots = np.percentile(x, q.tolist()).tolist()
    return inner_knots, lower, upper


def _apply_spline_formula(formula, indicator_squared, spline_knots):
    inner_knots, lower, upper = spline_knots
    spline = (
        f"cr(followup, knots={inner_knots}, lower_bound={lower}, upper_bound={upper})"
    )

    formula = re.sub(r"(\w+)\s*\*\s*followup\b", rf"\1*{spline}", formula)
    formula = re.sub(r"\bfollowup\s*\*\s*(\w+)", rf"{spline}*\1", formula)
    formula = re.sub(rf"\bfollowup{re.escape(indicator_squared)}\b", "", formula)
    formula = re.sub(r"\bfollowup\b", "", formula)

    formula = re.sub(r"\s+", " ", formula)
    formula = re.sub(r"\+\s*\+", "+", formula)
    formula = re.sub(r"^\s*\+\s*|\s*\+\s*$", "", formula).strip()

    if formula:
        return f"{formula} + {spline}"
    return spline


def _categorical_tv_columns(self, df_pd):
    """
    Names of the categorical (non-numeric) time-varying covariate columns
    present in ``df_pd``, including their baseline (``indicator_baseline``)
    versions used by the outcome model.
    """
    cols = []
    for col in self.time_varying_cols or []:
        for variant in (col, f"{col}{self.indicator_baseline}"):
            if variant in df_pd.columns and not is_numeric_dtype(df_pd[variant]):
                cols.append(variant)
    return cols


def _cast_categories(self, df_pd):
    if self.treatment_col in df_pd.columns:
        df_pd[self.treatment_col] = df_pd[self.treatment_col].astype("category")
    tx_bas = f"{self.treatment_col}{self.indicator_baseline}"
    if tx_bas in df_pd.columns:
        df_pd[tx_bas] = df_pd[tx_bas].astype("category")

    if self.followup_class and not self.followup_spline:
        df_pd["followup"] = df_pd["followup"].astype("category")
        squared_col = f"followup{self.indicator_squared}"
        if squared_col in df_pd.columns:
            df_pd[squared_col] = df_pd[squared_col].astype("category")

    if self.fixed_cols:
        for col in self.fixed_cols:
            if col in df_pd.columns:
                df_pd[col] = df_pd[col].astype("category")

    # Stable factor encoding for categorical time-varying covariates: fix the
    # level set from the full expanded data (captured on the non-bootstrap
    # pass) so a bootstrap resample cannot realise a different set of levels —
    # otherwise a level absent from the resample would be unknown to that fit
    # and crash counterfactual prediction with NaNs.
    tv_cat_cols = _categorical_tv_columns(self, df_pd)
    if getattr(self, "_current_boot_idx", None) is None:
        cats = getattr(self, "_covariate_categories", {})
        for col in tv_cat_cols:
            cats[col] = sorted(df_pd[col].dropna().unique().tolist())
        self._covariate_categories = cats
    cats = getattr(self, "_covariate_categories", {})
    for col in tv_cat_cols:
        if col in cats:
            df_pd[col] = pd.Categorical(df_pd[col], categories=cats[col])

    return df_pd


def _outcome_fit(
    self,
    df: pl.DataFrame,
    outcome: str,
    formula: str,
    weighted: bool = False,
    weight_col: str = "weight",
    start_params=None,
):
    if weighted:
        df = df.with_columns(
            pl.col(weight_col).clip(
                lower_bound=self.weight_min, upper_bound=self.weight_max
            )
        )

    if self.method == "censoring":
        df = df.filter(pl.col("switch") != 1)

    df_pd = _cast_categories(self, df.to_pandas())

    if self.followup_spline:
        if getattr(self, "_current_boot_idx", None) is None:
            self._followup_spline_knots = _compute_spline_knots(
                self.DT["followup"].to_numpy(), df=self.followup_spline_df
            )
        formula = _apply_spline_formula(
            formula, self.indicator_squared, self._followup_spline_knots
        )

    full_formula = f"{outcome} ~ {formula}"

    if getattr(self, "glm_package", "statsmodels") == "glum":
        from ..helpers._glum_fit import _fit_glum

        return _fit_glum(
            full_formula,
            df_pd,
            var_weights=df_pd[weight_col] if weighted else None,
        )

    glm_kwargs = {
        "formula": full_formula,
        "data": df_pd,
        "family": sm.families.Binomial(),
    }

    if weighted:
        glm_kwargs["var_weights"] = df_pd[weight_col]

    model = smf.glm(**glm_kwargs)

    # Drop warm-start coefs unless the design matrix columns match exactly
    # by name — bootstrap resamples can shift categorical reference levels or
    # column ordering, in which case the cached coefs are meaningless and
    # IRLS can diverge into NaN/Inf and crash LAPACK.
    if start_params is not None:
        sp_values, sp_names = start_params
        if list(model.exog_names) != list(sp_names):
            start_params = None
        else:
            start_params = sp_values

    try:
        model_fit = model.fit(start_params=start_params)
    except Exception:
        if start_params is not None:
            model_fit = model.fit()
        else:
            raise
    return model_fit
