from ..helpers import _predict_model
import polars as pl

def _weight_predict(self, WDT):
    grouping = [self.id_col]
    grouping += ["trial"] if self.weight_preexpansion else []
    time = self.time_col if self.weight_preexpansion else "followup"
    base_level = self.treatment_level[0]
    weights = []
    print(WDT.columns)
    
    if self.excused:
        # TODO
        pass
    
    if self.method == "ITT":
        WDT = WDT.with_columns([
            pl.lit(1.).alias("numerator"),
            pl.lit(1.).alias("denominator")
        ])
    else:
        for i, level in enumerate(self.treatment_level):
            # ehhhhh maybe better way to do this?
            pass
    