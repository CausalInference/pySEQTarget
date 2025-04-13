import multiprocessing
import math

def SEQopts(parallel: bool = False,
            n_cores: int = multiprocessing.cpu_count(),
            bootstrap: bool = False,
            bootstrap_nboot: int = 100,
            sample: float = 0.8,
            seed: int = 1636,
            followup_min: int = -math.inf,
            followup_max: int = math.inf,
            survival_max: int = math.inf,
            followup_include: bool = True,
            trial_include: bool = True,
            covariates: str = None,
            numerator: str = None,
            denominator: str = None,
            censor_numerator: str = None,
            censor_denominator: str = None,
            weighted: bool = False,
            weight_lower: float = -math.inf,
            weight_upper: float = math.inf,
            weight_p99: bool = False,
            weight_eligible0: str = None,
            weight_eligible1: str = None,
            weight_preexpansion: bool = True,
            calculate_variance: bool = False,
            hazard: bool = False,
            selection_random: bool = False,
            selection_probability: float = 0.8,
            excused: bool = False,
            cense: str = None,
            cense_eligible: str = None,
            multinomial: bool = False,
            treatment_levels: list = None,
            compevent: str = None,
            excused_col0: str = None,
            excused_col1: str = None,
            km_curves: bool = False,
            indicator_baseline: str = "_bas",
            indicator_squared: str = "_sq",
            scipy_method: str = "saga",
            followup_class: bool = False,
            followup_spline: bool = False,
            plot_title: str = None,
            plot_subtitle: str = None,
            plot_labels: str = None,
            plot_colors: list = None,
            plot_type: str = None,
            subgroup: str = None
            ) -> dict:
    """
    Dictionary parameter builder for SEQuential
        :param parallel: Whether the function should run in parallel
        :type parallel: bool
        :param ncores: The number of cores to use if running in parallel
        :type ncores: int
        :param bootstrap: Whether the function should bootstrap
        :type bootstrap: bool
        :param bootstrap_nboot

    """
    