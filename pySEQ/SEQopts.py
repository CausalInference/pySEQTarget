import multiprocessing
from dataclasses import dataclass, field
from typing import List, Optional, Literal

@dataclass
class SEQopts:
    bootstrap_nboot: int = 0
    bootstrap_sample: float = 0.8
    bootstrap_CI: float = 0.95
    bootstrap_CI_method: Literal["se", "percentile"] = "se"
    cense_colname : Optional[str] = None
    cense_denominator: Optional[str] = None
    cense_numerator: Optional[str] = None
    cense_eligible_colname: Optional[str] = None
    compevent_colname: Optional[str] = None
    covariates: Optional[List[str]] = None
    data_return: bool = False
    denominator: Optional[List[str]] = None
    excused: bool = False
    excused_colnames: Optional[List[str]] = None
    followup_class: bool = False
    followup_include: bool = True
    followup_max: int = None
    followup_min: int = 0
    followup_spline: bool = False
    hazard: bool = False
    indicator_baseline: str = "_bas"
    indicator_squared: str = "_sq"
    km_curves: bool = False
    multinomial: bool = False
    ncores: int = multiprocessing.cpu_count()
    numerator: Optional[List[str]] = None
    parallel: bool = False
    plot_colors: List[str] = field(default_factory=lambda: ["#F8766D", "#00BFC4", "#555555"])
    plot_labels: Optional[List[str]] = None
    plot_subtitle: str = None
    plot_title: str = None
    plot_type: Literal["risk", "survival", "inc"] = "risk"
    seed: Optional[int] = None
    selection_first_trial: bool = False
    selection_probability: float = 0.8
    selection_random: bool = False
    subgroup_colname: str = None
    survival_max: int = None
    survival_min: int = 0
    treatment_level: List[int] = field(default_factory=lambda: [0, 1])
    trial_include: bool = True
    weight_eligible_colnames: Optional[List[str]] = None
    weight_min: float = 0.0
    weight_max: float = None
    weight_lag_condition: bool = False
    weight_p99: bool = False
    weight_preexpansion: bool = False
    weighted: bool = False
    
    def __post_init__(self):
        bools = [
            'excused', 'followup_class', 'followup_include',
            'followup_spline', 'hazard', 'km_curves', 'multinomial',
            'parallel', 'selection_first_trial', 'selection_random',
            'trial_include', 'weight_lag_condition', 'weight_p99',
            'weight_preexpansion', 'weighted'
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
        if not (0.0 <= self.selection_probability <= 1.0):
            raise ValueError("selection_probability must be between 0 and 1.")
        
        if self.plot_type not in ['risk', 'survival']:
            raise ValueError("plot_type must be either 'risk' or 'survival'.")
        
        if self.bootstrap_CI_method not in ['se', 'percentile']:
            raise ValueError("bootstrap_CI_method must be one of 'se' or 'percentile'")
        
        lists = [
            
        ]

        for i in ('covariates', 'numerator', 'denominator',
                  'cense_numerator', 'cense_denominator'):
            attr = getattr(self, i)
            if attr is not None and not isinstance(attr, list):
                setattr(self, i, "".join(attr.split()))
