import multiprocessing
import os
from dataclasses import dataclass, field
from typing import List, Literal, Optional


@dataclass
class SEQopts:
    """
    Parameter builder for ``pySEQTarget.SEQuential`` analysis

    :param bootstrap_nboot: Number of bootstraps to perform
    :param bootstrap_sample: Subsampling proportion of ID-Trials gathered for each bootstrapping iteration
    :param bootstrap_CI: If bootstrapped, confidence interval level
    :param bootstrap_CI_method: If bootstrapped, confidence interval method ['SE' or 'percentile']
    :param cense_colname: Column name for censoring effect (LTFU, etc.)
    :param cense_denominator: Override to specify denominator patsy formula for
        censoring models; "1" or "" indicate intercept only model
    :param cense_numerator: Override to specify numerator patsy formula for censoring models
    :param cense_eligible_colname: Column name to identify which rows are eligible for censoring model fitting
    :param compevent_colname: Column name specifying a competing event to the outcome
    :param covariates: Override to specify the outcome patsy formula for outcome model fitting
    :param denominator: Override to specify the outcome patsy formula for denominator model fitting
    :param excused: Boolean to allow excused conditions when method is censoring
    :param excused_colnames: Column names (at the same length of treatment_level) specifying excused conditions, default ``[]``
    :param expand_only: If True, ``SEQuential.expand()`` returns the expanded dataset and skips weighting,
        modelling, and survival steps
    :param glm_package: Backend for fitting logistic (outcome/competing-event) models ["statsmodels", "glum", or "jax"], default "statsmodels".
    :param followup_class: Boolean to force followup values to be treated as classes
    :param followup_include: Boolean to force regular followup values into model covariates
    :param followup_spline: Boolean to force followup values to be fit to cubic spline
    :param followup_spline_df: Degrees of freedom for the followup cubic spline, default ``4``
    :param followup_max: Maximum allowed followup in analysis
    :param followup_min: Minimum allowed followup in analysis
    :param hazard_estimate: Boolean to create hazard estimates
    :param indicator_baseline: How to indicate baseline columns in models
    :param indicator_squared: How to indicate squared columns in models
    :param km_curves: Boolean to create survival, risk, and incidence (if applicable) estimates
    :param ncores: Number of cores to use if running in parallel, default ``max(1, cpu_count() - 1)``
    :param numerator: Override to specify the outcome patsy formula for
        numerator models; "1" or "" indicate intercept only model
    :param offload: Boolean to offload intermediate model data to disk
    :param offload_dir: Directory to offload intermediate model data
    :param parallel: Boolean to run model fitting in parallel
    :param plot_colors: List of colors for KM plots, if applicable, default ``["#F8766D", "#00BFC4", "#555555"]``
    :param plot_labels: List of length treat_level to specify treatment labeling, default ``[]``
    :param plot_title: Plot title
    :param plot_type: Type of plot to show ["risk", "survival" or "incidence" if compevent is specified]
    :param risk_times: Followup times at which to report risk difference and risk ratio when ``km_curves = True``.
        Each requested time is snapped to the latest available followup at or before it, and the maximum
        followup is always included. Defaults to ``None`` (report at the maximum followup only).
    :param seed: RNG seed
    :param selection_first_trial: Boolean to only use first trial for analysis (similar to non-expanded)
    :param selection_sample: Subsampling proportion of ID-trials which did not initiate a treatment
    :param selection_random: Boolean to randomly downsample ID-trials which did not initiate a treatment
    :param subgroup_colname: Column name for subgroups to share the same weighting but different outcome model fits
    :param treatment_level: List of eligible treatment levels within treatment_col, default ``[0, 1]``
    :param trial_include: Boolean to force trial values into model covariates
    :param visit_colname: Column name specifying visit number
    :param weight_eligible_colnames: List of column names of length
        treatment_level to identify which rows are eligible for weight fitting, default ``[]``
    :param weight_fit_method: The fitting method to be used ["newton", "bfgs", "lbfgs", "nm"], default "newton"
    :param weight_min: Minimum weight
    :param weight_max: Maximum weight
    :param weight_lag_condition: Boolean to fit weights based on their treatment lag
    :param weight_p99: Boolean to force weight min and max to be 1st and 99th percentile respectively
    :param weight_preexpansion: Boolean to fit weights on preexpanded data
    :param verbose: Boolean to print dataset size summaries and bootstrap information
    :param weighted: Boolean to weight analysis
    """

    bootstrap_nboot: int = 0
    bootstrap_sample: float = 0.8
    bootstrap_CI: float = 0.95
    bootstrap_CI_method: Literal["se", "percentile"] = "se"
    cense_colname: Optional[str] = None
    cense_denominator: Optional[str] = None
    cense_numerator: Optional[str] = None
    cense_eligible_colname: Optional[str] = None
    compevent_colname: Optional[str] = None
    covariates: Optional[str] = None
    cox_package: Literal["lifelines", "scikit-survival"] = "lifelines"
    denominator: Optional[str] = None
    excused: bool = False
    excused_colnames: List[str] = field(default_factory=lambda: [])
    expand_only: bool = False
    glm_package: Literal["statsmodels", "glum", "jax"] = "statsmodels"
    followup_class: bool = False
    followup_include: bool = True
    followup_max: int = None
    followup_min: int = 0
    followup_spline: bool = False
    followup_spline_df: int = 4
    hazard_estimate: bool = False
    indicator_baseline: str = "_bas"
    indicator_squared: str = "_sq"
    km_curves: bool = False
    ncores: Optional[int] = None
    numerator: Optional[str] = None
    offload: bool = False
    offload_dir: str = "_seq_models"
    parallel: bool = False
    plot_colors: List[str] = field(
        default_factory=lambda: ["#F8766D", "#00BFC4", "#555555"]
    )
    plot_labels: List[str] = field(default_factory=lambda: [])
    plot_title: str = None
    plot_type: Literal["risk", "survival", "incidence"] = "survival"
    risk_times: Optional[List[float]] = None
    seed: Optional[int] = None
    selection_first_trial: bool = False
    selection_sample: float = 0.8
    selection_random: bool = False
    subgroup_colname: str = None
    treatment_level: List[int] = field(default_factory=lambda: [0, 1])
    trial_include: bool = True
    visit_colname: str = None
    weight_eligible_colnames: List[str] = field(default_factory=lambda: [])
    weight_fit_method: Literal["newton", "bfgs", "lbfgs", "nm"] = "newton"
    weight_min: float = 0.0
    weight_max: float = None
    weight_lag_condition: bool = True
    weight_p99: bool = False
    weight_preexpansion: bool = True
    verbose: bool = False
    weighted: bool = False

    def _validate_bools(self):
        bools = [
            "excused",
            "expand_only",
            "followup_class",
            "followup_include",
            "followup_spline",
            "hazard_estimate",
            "km_curves",
            "parallel",
            "selection_first_trial",
            "selection_random",
            "trial_include",
            "verbose",
            "weight_lag_condition",
            "weight_p99",
            "weight_preexpansion",
            "weighted",
        ]
        for i in bools:
            if not isinstance(getattr(self, i), bool):
                raise TypeError(f"{i} must be a boolean value.")

    def _validate_ranges(self):
        if not isinstance(self.bootstrap_nboot, int) or self.bootstrap_nboot < 0:
            raise ValueError("bootstrap_nboot must be a positive integer.")
        if self.ncores < 1 or not isinstance(self.ncores, int):
            raise ValueError("ncores must be a positive integer.")
        if not (0.0 < self.bootstrap_sample <= 1.0):
            raise ValueError("bootstrap_sample must be between 0 (exclusive) and 1.")
        if not (0.0 < self.bootstrap_CI < 1.0):
            raise ValueError("bootstrap_CI must be between 0 and 1.")
        if not (0.0 < self.selection_sample <= 1.0):
            raise ValueError("selection_sample must be between 0 (exclusive) and 1.")
        if self.weight_max is not None and self.weight_max <= self.weight_min:
            raise ValueError(
                f"weight_min ({self.weight_min}) must be less than weight_max ({self.weight_max})."
            )
        if self.followup_max is not None and self.followup_max <= self.followup_min:
            raise ValueError(
                f"followup_min ({self.followup_min}) must be less than followup_max ({self.followup_max})."
            )
        if self.risk_times is not None:
            times = (
                self.risk_times
                if isinstance(self.risk_times, (list, tuple))
                else [self.risk_times]
            )
            if any(not isinstance(t, (int, float)) or t < 0 for t in times):
                raise ValueError("risk_times values must be non-negative numbers.")

    def _validate_choices(self):
        if self.plot_type not in ["risk", "survival", "incidence"]:
            raise ValueError(
                "plot_type must be either 'risk', 'survival', or 'incidence'."
            )
        if self.bootstrap_CI_method not in ["se", "percentile"]:
            raise ValueError("bootstrap_CI_method must be one of 'se' or 'percentile'")
        if self.glm_package not in ["statsmodels", "glum", "jax"]:
            raise ValueError("glm_package must be 'statsmodels', 'glum', or 'jax'")
        if self.cox_package not in ["lifelines", "scikit-survival"]:
            raise ValueError("cox_package must be 'lifelines' or 'scikit-survival'")

    def _normalize_formulas(self):
        for i in (
            "covariates",
            "numerator",
            "denominator",
            "cense_numerator",
            "cense_denominator",
        ):
            attr = getattr(self, i)
            if attr is not None and not isinstance(attr, list):
                setattr(self, i, "".join(attr.split()))

    def __post_init__(self):
        if self.ncores is None:
            self.ncores = max(1, multiprocessing.cpu_count() - 1)
        self._validate_bools()
        self._validate_ranges()
        self._validate_choices()
        self._normalize_formulas()

        if self.offload:
            os.makedirs(self.offload_dir, exist_ok=True)
