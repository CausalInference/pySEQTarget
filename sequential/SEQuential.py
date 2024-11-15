from typing import Optional, List, Dict
import polars as pl
from .SEQexpand import SEQ as SEQexpand

class SEQ(SEQexpand):
    def __init__(
            self,
            data: pl.DataFrame,
            id_col: str,
            time_col: str,
            eligible_col: str,
            treatment_col: str,
            outcome_col: str,
            time_varying_cols: Optional[List[str]] = None,
            fixed_cols: Optional[List[str]] = None,
            method: str = "ITT",
            weighted: bool = False,
            pre_expansion: bool = False,
            bootstrap: bool = False,
            nboot: int = 100,
            parallel: bool = False,
            ncores: int = 1,
            include_period: bool = True,
            include_trial: bool = False,
            compevent: Optional[str] = None,
            excused: bool = False,
            excused_col1: Optional[str] = None,
            excused_col0: Optional[str] = None,
            cense: Optional[str] = None,
            max_followup: float = float('inf'),
            max_survival: float = float('inf'),
            seed: int = 1636,
            km_curves: bool = False,
            covariates: Optional[str] = None,
            numerator: Optional[str] = None,
            denominator: Optional[str] = None,
            ltfu_denominator: Optional[str] = None,
            ltfu_numerator: Optional[str] = None,
            baseline_indicator: Optional[str] = "_bas",
            squared_indicator: Optional[str] = "_sq"
            ):
        
        self.data = data,
        self.id_col = id_col,
        self.time_col = time_col,
        self.eligible_col = eligible_col,
        self.treatment_col = treatment_col,
        self.outcome_col = outcome_col,
        self.time_varying_cols = time_varying_cols,
        self.fixed_cols = fixed_cols,
        self.method = method,
        self.weighted = weighted,
        self.pre_expansion = pre_expansion,
        self.bootstrap = bootstrap,
        self.nboot = nboot,
        self.bootstrap = bootstrap,
        self.nboot = nboot,
        self.parallel = parallel,
        self.ncores = ncores,
        self.include_period = include_period,
        self.include_trial = include_trial,
        self.compevent = compevent,
        self.excused = excused,
        self.excused_col0 = excused_col0,
        self.excused_col1 = excused_col1,
        self.cense = cense,
        self.max_followup = max_followup,
        self.max_survival = max_survival,
        self.seed = seed,
        self.km_curves = km_curves,
        self.covariates = covariates,
        self.numerator = numerator,
        self.denominator = denominator,
        self.ltfu_numerator = ltfu_numerator,
        self.ltfu_denominator = ltfu_denominator,
        self.baseline_indicator = baseline_indicator,
        self.squared_indicator = squared_indicator

        self._validate_inputs()
    
    def _validate_inputs(self):
        """Validate the inputs to ensure required columns and method are provided."""
        required_columns = [
            self.id_col,
            self.time_col,
            self.eligible_col,
            self.treatment_col,
            self.outcome_col,
        ] + self.time_varying_cols + self.fixed_cols
        missing_columns = [col for col in required_columns if col not in self.data.columns]
        if missing_columns:
            raise ValueError(f"Missing columns in data: {', '.join(missing_columns)}")

        if self.method not in {"ITT", "dose-response", "censoring"}:
            raise ValueError(
                f"Unsupported method '{self.method}'. Supported methods: 'ITT', 'dose-response', 'censoring'."
            )