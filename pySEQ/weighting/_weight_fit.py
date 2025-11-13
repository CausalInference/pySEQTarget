import polars as pl
import statsmodels.api as sm
import statsmodels.formula.api as smf

def _fit_LTFU(self, WDT: pl.DataFrame):
    WDT_pd = WDT.to_pandas()
    formula = f"{self.cense_colname}~{self.covariates}"
    # need self.cense_numerator and self.cense_denominator
    model = smf.glm(
        formula=formula,
        data=WDT_pd,
        family=sm.families.Binomial()
    )
    
    model_fit = model.fit()
    return model_fit

def _fit_numerator(self, WDT: pl.DataFrame):
    if self.weight_preexpansion and self.excused:
        pass
    else:
        formula = f"{self.treatment_col}~{self.numerator}"
        model = smf.mnlogit(
            formula,
            WDT
            )
    model_fit = model.fit()
    self.numerator_model = model.fit()
        
def _fit_denominator(self, WDT):
    formula = f"{self.treatment_col}~{self.numerator}"
    model = smf.mnlogit(
        formula,
        WDT
        )
    model_fit = model.fit()
    self.denominator_model = model_fit