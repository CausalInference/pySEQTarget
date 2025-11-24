from ..helpers import _predict_model
import polars as pl
import numpy as np

def _weight_predict(self, WDT):
    grouping = [self.id_col]
    grouping += ["trial"] if not self.weight_preexpansion else []
    time = self.time_col if self.weight_preexpansion else "followup"
    
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
            lag_mask = (WDT["tx_lag"] == level).to_numpy()
            
            if self.denominator_model[i] is not None:
                pred_denom = np.ones(WDT.height)
                if lag_mask.sum() > 0:
                    subset = WDT.filter(pl.Series(lag_mask))
                    p = _predict_model(self, self.denominator_model[i], subset)
                    if p.ndim == 1:
                        p = p.reshape(-1, 1)
                    p = p[:, i]
                    switched_treatment = (subset[self.treatment_col] != subset["tx_lag"]).to_numpy()
                    pred_denom[lag_mask] = np.where(switched_treatment, 1. - p, p)
            else:
                pred_denom = np.ones(WDT.height)
            
            if hasattr(self, "numerator_model") and self.numerator_model[i] is not None:
                pred_num = np.ones(WDT.height)
                if lag_mask.sum() > 0:
                    subset = WDT.filter(pl.Series(lag_mask))
                    p = _predict_model(self, self.numerator_model[i], subset)
                    if p.ndim == 1:
                        p = p.reshape(-1, 1)
                    p = p[:, i]
                    switched_treatment = (subset[self.treatment_col] != subset["tx_lag"]).to_numpy()
                    pred_num[lag_mask] = np.where(switched_treatment, 1. - p, p)
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
            p_num = _predict_model(self, self.cense_numerator, WDT).flatten()
            p_denom = _predict_model(self, self.cense_denominator, WDT).flatten()
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
