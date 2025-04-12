from typing import Optional, List
import sys
import polars as pl
from .SEQopts import SEQopts
from .initialization import __outcome, __numerator, __denominator, __censor_numerator, __censor_denominator
from .expansion import __mapper, __binder, __dynamic, __randomSelection
from .weighting import __weight_prepare_data, __weight_model, __weight_predict, __weight_bind, __weight_cumprod
from .analysis import __outcome_predict, __survival_prepare_data, __survival_predict
from .plot import __survival_plot


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
            parameters: dict = SEQopts()):
        
        self.data = data
        self.id_col = id_col
        self.time_col = time_col
        self.eligible_col = eligible_col
        self.treatment_col = treatment_col
        self.outcome_col = outcome_col
        self.time_varying_cols = time_varying_cols
        self.fixed_cols = fixed_cols
        self.method = method
        self.weighted = parameters['weighted']
        self.censor = parameters['censor']
        self.random_selection = parameters['random_selection']
        self.parameters = parameters

        if parameters['covariates'] is None:
            self.covariates = __outcome()
        else: self.covariates = parameters['covariates']

        if self.weighted:
            if parameters['numerator'] is None:
                self.numerator = __numerator()
            else: self.numerator = parameters['numerator']

            if parameters['denominator'] is None:
                self.denominator = __denominator()
            else: self.denominator = parameters['denominator']

            if self.censor is not None:
                if self.parameters['censor_numerator'] is None:
                    self.cense_numerator = __censor_numerator()
                else: self.cense_numerator = self.parameters['censor_numerator']

                if parameters['censor_denominator'] is None:
                    self.cense_denominator = __censor_denominator()
                else: self.censor_denominator = parameters['censor_denominator']

    def expand(self):
        self.DT = __binder(__mapper(self.data))
        if self.method != "ITT":
            self.DT = __dynamic(self.DT)
        if self.random_selection:
            self.DT = __randomSelection(self.DT)

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
        
            
    def outcome():

    def survival():
        pass

    def plot():
        pass
