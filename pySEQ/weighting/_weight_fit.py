import statsmodels.api as sm
import statsmodels.formula.api as smf

def _fit_LTFU(self, WDT):
    if self.cense_colname is None:
        return
    else:
        fits = []
        if self.cense_eligible_colname is not None:
            WDT = WDT[WDT[self.cense_eligible_colname] == 1]
            
        for i in [self.cense_numerator, self.cense_denominator]:
            formula = f"{self.cense_colname}~{i}"
            model = smf.glm(
                formula,
                WDT,
                family=sm.families.Binomial()
            )
            model_fit = model.fit(disp=0)
            fits.append(model_fit)
        
        self.cense_numerator = fits[0]
        self.cense_denominator = fits[1]

def _fit_numerator(self, WDT):
    if self.weight_preexpansion and self.excused:
        return
    if self.method == "ITT":
        return
    predictor = "switch" if self.excused else self.treatment_col
    formula = f"{predictor}~{self.numerator}"
    tx_bas = f"{self.treatment_col}{self.indicator_baseline}" if self.excused else "tx_lag"
    fits = []
    for i, level in enumerate(self.treatment_level):
        if self.excused and self.excused_colnames[i] is not None:
            DT_subset = WDT[WDT[self.excused_colnames[i]] == 0]
        else:
            DT_subset = WDT
        if self.weight_lag_condition:
            DT_subset = DT_subset[DT_subset[tx_bas] == level]
        if self.weight_eligible_colnames[i] is not None:
            DT_subset = DT_subset[DT_subset[self.weight_eligible_colnames[i]] == 1]
            
        model = smf.mnlogit(
            formula,
            DT_subset
            )
        model_fit = model.fit(disp=0)
        fits.append(model_fit)
        
    self.numerator_model = fits
        
def _fit_denominator(self, WDT):
    if self.method == "ITT":
        return
    predictor = "switch" if self.excused and not self.weight_preexpansion else self.treatment_col
    formula = f"{predictor}~{self.denominator}"
    fits = []
    for i, level in enumerate(self.treatment_level):
        if self.excused and self.excused_colnames[i] is not None:
            DT_subset = WDT[WDT[self.excused_colnames[i]] == 0]
        else:
            DT_subset = WDT
        if self.weight_lag_condition:
            DT_subset = DT_subset[DT_subset["tx_lag"] == level]        
        if not self.weight_preexpansion and not self.excused:
            DT_subset = DT_subset[DT_subset['followup'] != 0]
        if self.weight_eligible_colnames[i] is not None:
            DT_subset = DT_subset[DT_subset[self.weight_eligible_colnames[i]] == 1]
        
        model = smf.mnlogit(
            formula,
            DT_subset
            )
        model_fit = model.fit(disp=0)
        fits.append(model_fit)
        
    self.denominator_model = fits
    