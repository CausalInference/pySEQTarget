from ..helpers import _predict_model
import polars as pl
import numpy as np

def _weight_predict(self, WDT):
    grouping = [self.id_col]
    grouping += ["trial"] if not self.weight_preexpansion else []
    time = self.time_col if self.weight_preexpansion else "followup"
    classes = len(self.treatment_level)
    
    if self.excused:
        # TODO
        pass
    
    if self.method == "ITT":
        WDT = WDT.with_columns([
            pl.lit(1.).alias("numerator"),
            pl.lit(1.).alias("denominator")
        ])
    else:
        WDT = WDT.with_columns([
            pl.lit(1.).alias("numerator"),
            pl.lit(1.).alias("denominator")
        ])
        
        for i, level in enumerate(self.treatment_level):
            mask = pl.col("tx_lag") == level
            
            if self.denominator_model[i] is not None:
                p = _predict_model(self, self.denominator_model[i], WDT) \
                    .reshape(WDT.height, classes)[:, i]
                
                switched_treatment = (WDT[self.treatment_col] != WDT["tx_lag"]).to_numpy()
                pred_denom = np.where(switched_treatment, 1. - p, p)
            else:
                pred_denom = np.ones(WDT.height)
            
            if self.numerator_model[i] is not None:
                p = _predict_model(self, self.numerator_model[i], WDT) \
                    .reshape(WDT.height, classes)[:, i]
                
                switched_treatment = (WDT[self.treatment_col] != WDT["tx_lag"]).to_numpy()
                pred_num = np.where(switched_treatment, 1. - p, p)
            else:
                pred_num = np.ones(WDT.height)
            
            WDT = WDT.with_columns([
                pl.when(mask)
                  .then(pl.Series(pred_num))
                  .otherwise(pl.col("numerator"))
                  .alias("numerator"),
                pl.when(mask)
                  .then(pl.Series(pred_denom))
                  .otherwise(pl.col("denominator"))
                  .alias("denominator")
            ])
    
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
        WDT = WDT.with_columns(pl.lit(1.).alias("cense"))
    
    kept = ["numerator", "denominator", "cense", self.id_col, "trial", time, "tx_lag"]
    exists = [col for col in kept if col in WDT.columns]
    return WDT.select(exists).sort(grouping + [time])
