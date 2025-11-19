import polars as pl
import statsmodels.api as sm
import statsmodels.formula.api as smf
import pandas as pd

def _fit_LTFU(self, WDT: pl.DataFrame):
    if self.cense_colname is None:
        return
    else:
        fits = []
        for i in [self.cense_numerator, self.cense_denominator]:
            formula = f"{self.cense_colname}~{i}"
            model = smf.glm(
                formula,
                WDT,
                family=sm.families.Binomial()
            )
            model_fit = model.fit()
            fits.append(model_fit)
        
        self.cense_numerator = fits[0]
        self.cense_denominator = fits[1]

def _fit_numerator(self, WDT: pl.DataFrame):
    if self.weight_preexpansion and self.excused:
        return
    if self.method == "ITT":
        return
    predictor = "switch" if self.method == "censoring" else self.treatment_col
    formula = f"{predictor}~{self.numerator}"
    tx_bas = f"{self.treatment_col}{self.indicator_baseline}" if not self.weight_preexpansion else self.treatment_col
    fits = []
    
    for i in self.treatment_level:
        DT_subset = WDT[WDT[tx_bas] == i]
        print(DT_subset)
        model = smf.mnlogit(
            formula,
            DT_subset
            )
        model_fit = model.fit()
        fits.append(model_fit)
        
    self.numerator_model = model_fit
        
def _fit_denominator(self, WDT):
    if self.method == "ITT":
        return
    predictor = "switch" if self.method == "censoring" else self.treatment_col
    formula = f"{predictor}~{self.denominator}"
    tx_bas = f"{self.treatment_col}{self.indicator_baseline}" if not self.weight_preexpansion else self.treatment_col
    fits = []
    for i in self.treatment_level:
        DT_subset = WDT[WDT[tx_bas] == i]
        model = smf.mnlogit(
            formula,
            DT_subset
            )
        model_fit = model.fit()
        fits.append(model_fit)
        print(model_fit.summary())
        
    self.denominator_model = model_fit