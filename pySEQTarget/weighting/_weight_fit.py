import statsmodels.api as sm
import statsmodels.formula.api as smf


def _fit_pair(
    self, WDT, outcome_attr, formula_attr, output_attrs, eligible_colname_attr=None
):
    outcome = getattr(self, outcome_attr)

    if eligible_colname_attr is not None:
        _eligible_col = getattr(self, eligible_colname_attr)
        if _eligible_col is not None:
            WDT = WDT[WDT[_eligible_col] == 1]

    for rhs, out in zip(formula_attr, output_attrs):
        formula = f"{outcome}~{rhs}"
        model = smf.glm(formula, WDT, family=sm.families.Binomial())
        setattr(self, out, model.fit(disp=0))


def _fit_LTFU(self, WDT):
    if self.cense_colname is None:
        return
    _fit_pair(
        self,
        WDT,
        "cense_colname",
        [self.cense_numerator, self.cense_denominator],
        ["cense_numerator_model", "cense_denominator_model"],
        "cense_eligible_colname",
    )


def _fit_visit(self, WDT):
    if self.visit_colname is None:
        return
    _fit_pair(
        self,
        WDT,
        "visit_colname",
        [self.cense_numerator, self.cense_denominator],
        ["visit_numerator_model", "visit_denominator_model"],
    )


def _fit_numerator(self, WDT):
    if self.weight_preexpansion and self.excused:
        return
    if self.method == "ITT":
        return
    predictor = "switch" if self.excused else self.treatment_col
    formula = f"{predictor}~{self.numerator}"
    tx_bas = (
        f"{self.treatment_col}{self.indicator_baseline}" if self.excused else "tx_lag"
    )
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
        model = smf.mnlogit(formula, DT_subset)
        model_fit = model.fit(disp=0)
        fits.append(model_fit)

    self.numerator_model = fits


def _fit_denominator(self, WDT):
    if self.method == "ITT":
        return
    predictor = (
        "switch"
        if self.excused and not self.weight_preexpansion
        else self.treatment_col
    )
    formula = f"{predictor}~{self.denominator}"
    fits = []
    for i, level in enumerate(self.treatment_level):
        if self.excused and self.excused_colnames[i] is not None:
            DT_subset = WDT[WDT[self.excused_colnames[i]] == 0]
        else:
            DT_subset = WDT
        if self.weight_lag_condition:
            DT_subset = DT_subset[DT_subset["tx_lag"] == level]
        if not self.weight_preexpansion:
            DT_subset = DT_subset[DT_subset["followup"] != 0]
        if self.weight_eligible_colnames[i] is not None:
            DT_subset = DT_subset[DT_subset[self.weight_eligible_colnames[i]] == 1]

        model = smf.mnlogit(formula, DT_subset)
        model_fit = model.fit(disp=0)
        fits.append(model_fit)

    self.denominator_model = fits
