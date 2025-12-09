import multiprocessing
from dataclasses import dataclass, field
from typing import List, Literal, Optional


@dataclass
class SEQopts:
    """
    Parameter builder for ``pySEQTarget.SEQuential`` analysis

    :param bootstrap_nboot: Number of bootstraps to preform
    :type bootstrap_nboot: int
    :param bootstrap_sample: Subsampling proportion of ID-Trials gathered for each bootstrapping iteration
    :type bootstrap_sample: float
    :param bootstrap_CI: If bootstrapped, confidence interval level
    :type bootstrap_CI: float
    :param bootstrap_CI_method: If bootstrapped, confidence method generation method ['SE' or 'percentile']
    :type bootstrap_CI_method: str
    :param cense_colname: Column name for censoring effect (LTFU, etc.)
    :type cense_colname: str
    :param cense_denominator: Override to specify denominator patsy formula for censoring models
    :type cense_denominator: Optional[str] or None
    :param cense_numerator: Override to specify numerator patsy formula for censoring models
    :type cense_numerator: Optional[str] or None
    :param cense_eligible_colname: Column name to identify which rows are eligible for censoring model fitting
    :type cense_eligible_colname: Optional[str] or None
    :param compevent_colname: Column name specifying a competing event to the outcome
    :type compevent_colname: str
    :param covariates: Override to specify the outcome patsy formula for outcome model fitting
    :type covariates: Optional[str] or None
    :param denominator: Override to specify the outcome patsy formula for denominator model fitting
    :type denominator: Optional[str] or None
    :param excused: Boolean to allow excused conditions when method is censoring
    :type excused: bool
    :param excused_colnames: Column names (at the same length of treatment_level) specifying excused conditions
    :type excused_colnames: List[str] or []
    :param followup_class: Boolean to force followup values to be treated as classes
    :type followup_class: bool
    :param followup_include: Boolean to force regular followup values into model covariates
    :type followup_include: bool
    :param followup_spline: Boolean to force followup values to be fit to cubic spline
    :type followup_spline: bool
    :param followup_max: Maximum allowed followup in analysis
    :type followup_max: int or None
    :param followup_min: Minimum allowed followup in analysis
    :type followup_min: int
    :param hazard_estimate: Boolean to create hazard estimates
    :type hazard_estimate: bool
    :param indicator_baseline: How to indicate baseline columns in models
    :type indicator_baseline: str
    :param indicator_squared: How to indicate squared columns in models
    :type indicator_baseline: str
    :param km_curves: Boolean to create survival, risk, and incidence (if applicable) estimates
    :type km_curves: bool
    :param ncores: Number of cores to use if running in parallel
    :type ncores: int
    :param numerator: Override to specify the outcome patsy formula for numerator models
    :type numerator: str
    :param parallel: Boolean to run model fitting in parallel
    :type parallel: bool
    :param plot_colors: List of colors for KM plots, if applicable
    :type plot_colors: List[str]
    :param plot_labels: List of length treat_level to specify treatment labeling
    :type plot_labels: List[str]
    :param plot_title: Plot title
    :type plot_title: str
    :param plot_type: Type of plot to show ["risk", "survival" or "incidence" if compevent is specified]
    :type plot_type: str
    :param seed: RNG seed
    :type seed: int
    :param selection_first_trial: Boolean to only use first trial for analysis (similar to non-expanded)
    :type selection_first_trial: bool
    :param selection_sample: Subsampling proportion of ID-trials which did not initiate a treatment
    :type selection_sample: float
    :param selection_random: Boolean to randomly downsample ID-trials which did not initiate a treatment
    :type selection_random: bool
    :param subgroup_colname: Column name for subgroups to share the same weighting but different outcome model fits
    :type subgroup_colname: str
    :param treatment_level: List of eligible treatment levels within treatment_col
    :type treatment_level: List[int]
    :param trial_include: Boolean to force trial values into model covariates
    :type trial_include: bool
    :param weight_eligible_colnames: List of column names of length treatment_level to identify which rows are eligible for weight fitting
    :type weight_eligible_colnames: List[str]
    :param weight_min: Minimum weight
    :type weight_min: float
    :param weight_max: Maximum weight
    :type weight_max: float or None
    :param weight_lag_condition: Boolean to fit weights based on their treatment lag
    :type weight_lag_condition: bool
    :param weight_p99: Boolean to force weight min and max to be 1st and 99th percentile respectively
    :type weight_p99: bool
    :param weight_preexpansion: Boolean to fit weights on preexpanded data
    :type weight_preexpansion: bool
    :param weighted: Boolean to weight analysis
    :type weighted: bool
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
    denominator: Optional[str] = None
    excused: bool = False
    excused_colnames: List[str] = field(default_factory=lambda: [])
    followup_class: bool = False
    followup_include: bool = True
    followup_max: int = None
    followup_min: int = 0
    followup_spline: bool = False
    hazard_estimate: bool = False
    indicator_baseline: str = "_bas"
    indicator_squared: str = "_sq"
    km_curves: bool = False
    ncores: int = multiprocessing.cpu_count()
    numerator: Optional[str] = None
    parallel: bool = False
    plot_colors: List[str] = field(
        default_factory=lambda: ["#F8766D", "#00BFC4", "#555555"]
    )
    plot_labels: List[str] = field(default_factory=lambda: [])
    plot_title: str = None
    plot_type: Literal["risk", "survival", "incidence"] = "risk"
    seed: Optional[int] = None
    selection_first_trial: bool = False
    selection_sample: float = 0.8
    selection_random: bool = False
    subgroup_colname: str = None
    treatment_level: List[int] = field(default_factory=lambda: [0, 1])
    trial_include: bool = True
    visit_colname: str = None
    weight_eligible_colnames: List[str] = field(default_factory=lambda: [])
    weight_min: float = 0.0
    weight_max: float = None
    weight_lag_condition: bool = True
    weight_p99: bool = False
    weight_preexpansion: bool = False
    weighted: bool = False

    def __post_init__(self):
        bools = [
            "excused",
            "followup_class",
            "followup_include",
            "followup_spline",
            "hazard_estimate",
            "km_curves",
            "parallel",
            "selection_first_trial",
            "selection_random",
            "trial_include",
            "weight_lag_condition",
            "weight_p99",
            "weight_preexpansion",
            "weighted",
        ]
        for i in bools:
            if not isinstance(getattr(self, i), bool):
                raise TypeError(f"{i} must be a boolean value.")

        if not isinstance(self.bootstrap_nboot, int) or self.bootstrap_nboot < 0:
            raise ValueError("bootstrap_nboot must be a positive integer.")

        if self.ncores < 1 or not isinstance(self.ncores, int):
            raise ValueError("ncores must be a positive integer.")

        if not (0.0 <= self.bootstrap_sample <= 1.0):
            raise ValueError("bootstrap_sample must be between 0 and 1.")
        if not (0.0 < self.bootstrap_CI < 1.0):
            raise ValueError("bootstrap_CI must be between 0 and 1.")
        if not (0.0 <= self.selection_sample <= 1.0):
            raise ValueError("selection_sample must be between 0 and 1.")

        if self.plot_type not in ["risk", "survival", "incidence"]:
            raise ValueError(
                "plot_type must be either 'risk', 'survival', or 'incidence'."
            )

        if self.bootstrap_CI_method not in ["se", "percentile"]:
            raise ValueError("bootstrap_CI_method must be one of 'se' or 'percentile'")

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
