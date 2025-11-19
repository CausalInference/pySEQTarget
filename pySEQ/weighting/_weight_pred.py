from ..helpers import _predict_model
import polars as pl
import numpy as np

def _weight_predict(self, WDT):
    grouping = [self.id_col]
    grouping += ["trial"] if self.weight_preexpansion else []
    time = self.time_col if self.weight_preexpansion else "followup"
    classes = len(self.treatment_level)
    weights = []
    
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
            subset = WDT.filter(pl.col("tx_lag") == level)
            pred_denom = np.ones(subset.height)
            pred_num = np.ones(subset.height)

            if self.denominator_model is not None:
                p = _predict_model(self, self.denominator_model, subset) \
                    .reshape(subset.height, classes)[:, i]
                same_tx = (subset[self.treatment_col] == level).to_numpy()
                pred_denom = np.where(same_tx, p, 1. - p)
                    
            if self.numerator_model is not None:
                p = _predict_model(self, self.numerator_model, subset) \
                    .reshape(subset.height, classes)[:, i]
                same_tx = (subset[self.treatment_col] == level).to_numpy()
                pred_num = np.where(same_tx, p, 1. - p)
            
            subset = subset.with_columns([
                pl.Series("numerator", pred_num),
                pl.Series("denominator", pred_denom)
            ])
            
            weights.append(subset)
        WDT = pl.concat(weights).sort(grouping + [time])
        
        if self.cense_colname is not None:
            p_num = _predict_model(self, self.cense_numerator, WDT)
            p_denom = _predict_model(self, self.cense_denominator, WDT)
            
            WDT = WDT.with_columns([
                pl.Series("cense_numerator", p_num),
                pl.Series("cense_denominator", p_denom)
            ]).with_columns(
                (pl.col("cense_numerator") / pl.col("cense_denominator")).alias("cense")
            )
        else:
            WDT = WDT.with_columns(
                pl.lit(1.).alias("cense")
            )
    kept = ["numerator", "denominator", "cense", self.id_col, "trial", time]
    exists =[col for col in kept if col in WDT.columns]
    
    return WDT.select(exists)