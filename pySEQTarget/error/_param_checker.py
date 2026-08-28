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

    # End-of-follow-up outcomes replace the survival outcome model with a direct
    # weighted average at a single follow-up time, so the survival-based outputs
    # have no meaning there and the requested time must lie inside the expansion.
    if self.end_of_fup:
        if self.km_curves or self.hazard_estimate:
            raise ValueError(
                "end_of_fup is not compatible with km_curves or hazard_estimate: "
                "an end-of-follow-up outcome is evaluated at a single time, so "
                "there is no survival curve or hazard to estimate."
            )
        if self.method == "dose-response":
            raise ValueError(
                "end_of_fup is not supported for the dose-response method."
            )
        if self.compevent_colname is not None:
            raise ValueError(
                "end_of_fup is not compatible with compevent_colname: competing "
                "events are a survival-outcome concept."
            )
        upper = self.end_of_fup_time + self.end_of_fup_window
        if upper > self.followup_max:
            raise ValueError(
                f"end_of_fup_time plus end_of_fup_window ({upper}) exceeds the "
                f"maximum follow-up ({self.followup_max}); widen followup_max or "
                "narrow the window."
            )
        if self.end_of_fup_time - self.end_of_fup_window < self.followup_min:
            raise ValueError(
                "end_of_fup_time minus end_of_fup_window "
                f"({self.end_of_fup_time - self.end_of_fup_window}) is below the "
                f"minimum follow-up ({self.followup_min})."
            )

    if self.hazard_estimate and self.method == "dose-response":
        raise ValueError(
            "Hazard ratio estimation is not supported for method='dose-response': "
            "the counterfactual simulation only sets the baseline treatment, but "
            "the dose-response outcome model depends on the cumulative dose, so "
            "both arms would simulate identical outcomes (HR ≈ 1)."
        )

    if sum([self.followup_class, self.followup_include, self.followup_spline]) > 1:
        raise ValueError(
            "Only one of followup_class, followup_include, or followup_spline "
            "can be set to True."
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

    # Per-treatment-level weight models: 'numerator'/'denominator' may be a list
    # with one formula per treatment_level (in treatment_level order), fitting a
    # separate model per arm. Only supported for post-expansion weights.
    for name in ("numerator", "denominator"):
        spec = getattr(self, name)
        if isinstance(spec, (list, tuple)):
            if not self.weighted or self.method == "ITT":
                raise ValueError(
                    f"Per-treatment-level '{name}' models require a weighted, "
                    "non-ITT analysis."
                )
            if self.weight_preexpansion:
                raise ValueError(
                    f"Per-treatment-level '{name}' models are only supported for "
                    "post-expansion weights (weight_preexpansion=False)."
                )
            if any(f is None for f in spec):
                raise ValueError(
                    f"Per-treatment-level '{name}' formulas contain None; supply "
                    "one formula per treatment level."
                )
            if len(spec) != len(self.treatment_level):
                raise ValueError(
                    f"'{name}' must be a single formula or one per treatment "
                    f"level ({len(self.treatment_level)} expected, in "
                    f"'treatment_level' order) but {len(spec)} were supplied."
                )

    if (
        self.weighted
        and self.method != "ITT"
        and self.numerator is not None
        and self.denominator is not None
    ):
        num_list = (
            list(self.numerator)
            if isinstance(self.numerator, (list, tuple))
            else [self.numerator]
        )
        den_list = (
            list(self.denominator)
            if isinstance(self.denominator, (list, tuple))
            else [self.denominator]
        )
        # Warn on any arm whose numerator and denominator formulas coincide
        # (element-wise when both are per-arm; the shared/shared case reduces to
        # comparing the two single formulas).
        if len(num_list) == len(den_list):
            same = sorted({n for n, d in zip(num_list, den_list) if n == d})
            if same:
                covs = "', '".join(same)
                warnings.warn(
                    f"Numerator and denominator weight models use identical "
                    f"covariates ('{covs}'); the stabilized weights "
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
