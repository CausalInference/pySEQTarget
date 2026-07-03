import statsmodels.api as sm
import statsmodels.formula.api as smf

from ..error._check_separation import _check_separation
from ..helpers._glum_fit import _fit_glum


def _formula_rhs_for_level(spec, level_idx):
    """
    Resolve the right-hand-side formula for a treatment level. ``spec`` is either
    a single formula string (shared across arms) or a list with one formula per
    treatment level (in treatment_level order), fitting a separate model per arm.
    """
    if isinstance(spec, (list, tuple)):
        return spec[level_idx]
    return spec


def _get_subset_for_level(
    self, WDT, level_idx, level, tx_lag_col, exclude_followup_zero=False
):
    """
    Helper to create the subset of data for a given treatment level.
    Consolidates the repeated filtering logic from _fit_numerator and _fit_denominator.
    """
    DT_subset = WDT

    # Filter by excused column if applicable
    if self.excused and self.excused_colnames[level_idx] is not None:
        DT_subset = DT_subset[DT_subset[self.excused_colnames[level_idx]] == 0]

    # Filter by treatment lag condition
    if self.weight_lag_condition:
        DT_subset = DT_subset[DT_subset[tx_lag_col] == level]

    # Exclude followup == 0 for denominator (not pre-expansion)
    if exclude_followup_zero:
        DT_subset = DT_subset[DT_subset["followup"] != 0]

    # Filter by eligibility column if applicable
    if self.weight_eligible_colnames[level_idx] is not None:
        DT_subset = DT_subset[DT_subset[self.weight_eligible_colnames[level_idx]] == 1]

    return DT_subset


def _fit_pair(
    self, WDT, outcome_attr, formula_attr, output_attrs, eligible_colname_attr=None
):
    outcome = getattr(self, outcome_attr)

    if eligible_colname_attr is not None:
        _eligible_col = getattr(self, eligible_colname_attr)
        if _eligible_col is not None:
            WDT = WDT[WDT[_eligible_col] == 1]

    for rhs, out in zip(formula_attr, output_attrs):
        if len(WDT[outcome].unique()) < 2:
            setattr(self, out, None)
            continue
        formula = f"{outcome}~{rhs}"
        if getattr(self, "glm_package", "statsmodels") == "glum":
            fitted = _fit_glum(formula, WDT)
        else:
            model = smf.glm(formula, WDT, family=sm.families.Binomial())
            fitted = model.fit(disp=0, method=self.weight_fit_method)
        _check_separation(fitted, label=out.replace("_model", "").replace("_", " "))
        setattr(self, out, fitted)


def _fit_LTFU(self, WDT):
    if self.cense_colname is None:
        return
    _fit_pair(
        self,
        WDT,
        "cense_colname",
        [self.cense_numerator, self.cense_denominator],
        ["cense_numerator_model", "cense_denominator_model"],
        "cense_eligible_colname",
    )


def _fit_visit(self, WDT):
    if self.visit_colname is None:
        return
    _fit_pair(
        self,
        WDT,
        "visit_colname",
        [self.cense_numerator, self.cense_denominator],
        ["visit_numerator_model", "visit_denominator_model"],
    )


def _fit_numerator(self, WDT):
    if self.weight_preexpansion and self.excused:
        return
    if self.method == "ITT":
        return
    predictor = "switch" if self.excused else self.treatment_col
    tx_lag_col = (
        f"{self.treatment_col}{self.indicator_baseline}" if self.excused else "tx_lag"
    )
    fits = []
    # Use logit for binary 0/1 treatment with censoring method only
    # treatment_level=[1,2] or dose-response always uses mnlogit
    is_binary = sorted(self.treatment_level) == [0, 1] and self.method == "censoring"
    for i, level in enumerate(self.treatment_level):
        # numerator may be per-treatment-level; select this arm's formula.
        rhs = _formula_rhs_for_level(self.numerator, i)
        formula = f"{predictor}~1" if rhs in ("1", "") else f"{predictor}~{rhs}"
        DT_subset = _get_subset_for_level(self, WDT, i, level, tx_lag_col)
        if len(DT_subset[predictor].unique()) < 2:
            fits.append(None)
            continue
        # Use logit for binary 0/1 censoring, mnlogit otherwise
        if is_binary and getattr(self, "glm_package", "statsmodels") == "glum":
            model_fit = _fit_glum(formula, DT_subset)
        elif is_binary:
            model_fit = smf.logit(formula, DT_subset).fit(
                disp=0, method=self.weight_fit_method
            )
        else:
            model_fit = smf.mnlogit(formula, DT_subset).fit(
                disp=0, method=self.weight_fit_method
            )
        _check_separation(model_fit, label=f"numerator (level {level})")
        fits.append(model_fit)

    self.numerator_model = fits
    self._is_binary_treatment = is_binary


def _fit_denominator(self, WDT):
    if self.method == "ITT":
        return
    predictor = (
        "switch"
        if self.excused and not self.weight_preexpansion
        else self.treatment_col
    )
    fits = []
    # Use logit for binary 0/1 treatment with censoring method only
    # treatment_level=[1,2] or dose-response always uses mnlogit
    is_binary = sorted(self.treatment_level) == [0, 1] and self.method == "censoring"
    exclude_followup_zero = not self.weight_preexpansion
    for i, level in enumerate(self.treatment_level):
        # denominator may be per-treatment-level; select this arm's formula.
        rhs = _formula_rhs_for_level(self.denominator, i)
        formula = f"{predictor}~1" if rhs in ("1", "") else f"{predictor}~{rhs}"
        DT_subset = _get_subset_for_level(
            self, WDT, i, level, "tx_lag", exclude_followup_zero=exclude_followup_zero
        )
        if len(DT_subset[predictor].unique()) < 2:
            fits.append(None)
            continue
        # Use logit for binary 0/1 censoring, mnlogit otherwise
        if is_binary and getattr(self, "glm_package", "statsmodels") == "glum":
            model_fit = _fit_glum(formula, DT_subset)
        elif is_binary:
            model_fit = smf.logit(formula, DT_subset).fit(
                disp=0, method=self.weight_fit_method
            )
        else:
            model_fit = smf.mnlogit(formula, DT_subset).fit(
                disp=0, method=self.weight_fit_method
            )
        _check_separation(model_fit, label=f"denominator (level {level})")
        fits.append(model_fit)

    self.denominator_model = fits
    self._is_binary_treatment = is_binary
