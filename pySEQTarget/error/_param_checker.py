import warnings

from ..helpers import _pad


def _param_checker(self):
    overlap = set(self.time_varying_cols) & set(self.fixed_cols)
    if overlap:
        raise ValueError(
            f"Columns cannot appear in both time_varying_cols and fixed_cols: {sorted(overlap)}"
        )

    actual_levels = set(self.data[self.treatment_col].unique().to_list())
    missing_levels = set(self.treatment_level) - actual_levels
    if missing_levels:
        raise ValueError(
            f"treatment_level contains values not found in '{self.treatment_col}': {sorted(missing_levels)}"
        )

    if (
        self.subgroup_colname is not None
        and self.subgroup_colname not in self.fixed_cols
    ):
        raise ValueError("subgroup_colname must be included in fixed_cols.")

    if self.followup_max is None:
        self.followup_max = self.data.select(self.time_col).to_series().max()

    if len(self.excused_colnames) == 0 and self.excused:
        self.excused = False
        warnings.warn(
            "Excused column names not provided but excused is set to True. Automatically set excused to False"
        )

    if len(self.excused_colnames) > 0 and not self.excused:
        self.excused = True
        warnings.warn(
            "Excused column names provided but excused is set to False. Automatically set excused to True"
        )

    if self.km_curves and self.hazard_estimate:
        raise ValueError("km_curves and hazard cannot both be set to True.")

    if self.hazard_estimate and self.method == "dose-response":
        raise ValueError(
            "Hazard ratio estimation is not supported for method='dose-response': "
            "the counterfactual simulation only sets the baseline treatment, but "
            "the dose-response outcome model depends on the cumulative dose, so "
            "both arms would simulate identical outcomes (HR ≈ 1)."
        )

    if sum([self.followup_class, self.followup_include, self.followup_spline]) > 1:
        raise ValueError(
            "Only one of followup_class or followup_include can be set to True."
        )

    if self.followup_spline_df < 2:
        raise ValueError("followup_spline_df must be at least 2.")

    if (
        self.weighted
        and self.method == "ITT"
        and self.cense_colname is None
        and self.visit_colname is None
    ):
        raise ValueError(
            "For weighted ITT analyses, cense_colname or visit_colname must be provided."
        )

    if (
        self.weighted
        and self.method != "ITT"
        and self.numerator is not None
        and self.denominator is not None
        and self.numerator == self.denominator
    ):
        warnings.warn(
            f"Numerator and denominator weight models use identical "
            f"covariates ('{self.numerator}'); the stabilized weights "
            "will all equal 1 (i.e., no weighting). The denominator "
            "should typically include the time-varying confounders "
            "that the numerator omits — check for a typo in either or "
            "both of 'numerator' and 'denominator'.",
            UserWarning,
            stacklevel=2,
        )

    if self.excused:
        _, self.excused_colnames = _pad(self.treatment_level, self.excused_colnames)
    _, self.weight_eligible_colnames = _pad(
        self.treatment_level, self.weight_eligible_colnames
    )

    return
