from typing import Optional, List
import sys
from dataclasses import asdict
from collections import Counter
import polars as pl
import numpy as np

from .SEQopts import SEQopts
from .helpers import _colString, bootstrap_loop
from .initialization import _outcome, _numerator, _denominator, _cense_numerator, _cense_denominator
from .expansion import _mapper, _binder, _dynamic, _randomSelection
from .weighting import _weight_prepare_data, _weight_model, _weight_predict, _weight_bind, _weight_cumprod
from .analysis import _outcome_fit, _survival_prepare_data, _survival_predict
from .plot import _survival_plot


class SEQuential:
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
            parameters: Optional[SEQopts] = None
    ):
        self.data = data
        self.id_col = id_col
        self.time_col = time_col
        self.eligible_col = eligible_col
        self.treatment_col = treatment_col
        self.outcome_col = outcome_col
        self.time_varying_cols = time_varying_cols
        self.fixed_cols = fixed_cols
        self.method = method
        
        if parameters is None:
            parameters = SEQopts()
            
        # Absorb parameters from SEQopts dataclass
        for name, value in asdict(parameters).items():
            setattr(self, name, value)
        
        # Create default covariates
        if self.covariates is None:
            self.covariates = _outcome(self)

        if self.weighted:
            if self.numerator is None:
                self.numerator = _numerator(self)

            if self.denominator is None:
                self.denominator = _denominator(self)

            if self.cense is not None:
                if self.cense_numerator is None:
                    self.cense_numerator = _cense_numerator()

                if self.cense_denominator is None:
                    self.cense_denominator = _cense_denominator()

    def expand(self):
        self.DT = _binder(_mapper(self.data, self.id_col, self.time_col), self.data, _colString([
            self.covariates, self.numerator, self.denominator, self.cense_numerator, self.cense_denominator
            ]), self.eligible_col, self.excused_colnames,
            self.indicator_baseline, self.indicator_squared)
        
        if self.method != "ITT":
            self.DT = _dynamic(self.DT)
        if self.selection_random:
            self.DT = _randomSelection(self.DT)

    def weight(self):
        if not self.weighted: 
            sys.exit("""No planned weighting initialized, 
                     consider adding weighted=True in your parameter dictionary""")
        # for level in treatment level do
        # data creation - subset data to where the baseline is level
        # modeling - model from the data and covariates
        # predition - predict on the data the odds for adherence 
        # anti-prediction - where there is no adherence, 1- prediction
        # next
        
    def bootstrap(self):
        rng = np.random.RandomState(self.seed) if self.seed is not None else np.random
        UIDs = self.DT.select(pl.col(self.id_col)).unique().to_series().to_list()
        NIDs = len(UIDs)
        
        self._boot_samples = []
        for _ in range(self.bootstrap_nboot):
            sampled_IDs = rng.choice(UIDs, size=int(self.bootstrap_sample * NIDs), replace=True)
            id_counts = Counter(sampled_IDs)
            self._boot_samples.append(id_counts)
        return self
    
    @bootstrap_loop      
    def outcome(self):
        if self.weighted and "weight" not in self.DT:
            print("It seems like you have not weighted your data yet, consider running the weight() method first.")
        self.outcome_model = _outcome_fit(self.DT, 
                                           self.outcome_col, 
                                           self.covariates, 
                                           self.weighted, 
                                           "weight")

    def survival():
        pass

    def plot():
        pass
