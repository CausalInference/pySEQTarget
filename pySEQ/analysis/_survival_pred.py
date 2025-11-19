import concurrent.futures
import polars as pl
import numpy as np
from ..helpers import _predict_model

def _get_outcome_predictions(self, newdata):
    if self.compevent_colname is not None:
        pass
    
    if self.parallel and self.ncores > 1:
        with concurrent.futures.ProcessPoolExecutor(max_workers=self.ncores) as executor:
            preds = list(executor.map(_predict_model, self, self.outcome_model, newdata))
    else:
        preds = [_predict_model(self, model, newdata) for model in self.outcome_model]
    
    return preds

def _calculate_risk(self):
    a = 1 - self.bootstrap_CI
    lci = a / 2
    uci = 1 - lci
    
    if self.followup_max is None:
        self.followup_max = self.DT.select(pl.col("followup").max()).to_numpy()[0][0]
    
    SDT = (
        self.DT
        .with_columns([
            (pl.col(self.id_col).cast(pl.Utf8) + pl.col("trial").cast(pl.Utf8)).alias("TID")
        ])
        .group_by("TID")
        .first()
        .drop(["followup", f"followup{self.indicator_squared}"])
        .with_columns([
            pl.lit(list(range(self.followup_max))).alias("followup")
        ])
        .explode("followup")
        .with_columns([
            (pl.col("followup") + 1).alias("followup"),
            (pl.col("followup") ** 2).alias(f"followup{self.indicator_squared}")
        ])
    ).sort([self.id_col, "trial", "followup"])
    
    risks = []
    for i in self.treatment_level:
        TxDT = SDT.with_columns([
            pl.lit(i).alias(f"{self.treatment_col}{self.indicator_baseline}")
        ])
        
        if self.method == "dose-response":
            pass
        if self.compevent_colname is not None:
            pass
        
        preds = _get_outcome_predictions(self, TxDT)
        full = [pl.Series("pred_risk", preds[0])]
        
        if self.bootstrap_nboot > 0:
            for idx, pred in enumerate(preds[1:], start=1):
                full.append(pl.Series(f"pred_risk_{idx}", pred))
        
        names = [col.name for col in full]
        
        TxDT = TxDT.with_columns(full).with_columns([
            (1 - pl.col(col)).cum_prod().over("TID").alias(col) for col in names
        ]).group_by("followup").agg([
            pl.col(col).mean() for col in names
        ]).sort("followup").with_columns([
            (1 - pl.col(col)).alias(col) for col in names
        ])
        
        boots = [col for col in names if col != "pred_risk"]
        
        if len(boots) > 0:
            risk = TxDT.select(["followup"] + boots).unpivot(
                index="followup",
                on=boots,
                variable_name="bootID",
                value_name="risk"
            ).group_by("followup").agg([
                pl.col("risk").std().alias("SE"),
                pl.col("risk").quantile(lci).alias("LCI"),
                pl.col("risk").quantile(uci).alias("UCI")
            ])
            risk = risk.join(TxDT.select(["followup", "pred_risk"]), on="followup")
                        
            if self.bootstrap_CI_method == "se":
                from scipy.stats import norm
                z = norm.ppf(1 - a / 2)
                risk = risk.with_columns([
                    (pl.col("pred_risk") - z * pl.col("SE")).alias("LCI"),
                    (pl.col("pred_risk") + z * pl.col("SE")).alias("UCI")
                ])
            risk = risk.with_columns(pl.lit(i).alias(self.treatment_col)).select([
                "followup",
                self.treatment_col,
                "pred_risk",
                "SE",
                "LCI",
                "UCI"
            ])
            
            fup0 = pl.DataFrame({
                "followup": [0],
                self.treatment_col: [i],
                "pred_risk": [0.0],
                "SE": [0.0],
                "LCI": [0.0],
                "UCI": [0.0]
            }).with_columns([
                pl.col("followup").cast(pl.Int64),
                pl.col(self.treatment_col).cast(pl.Int32)
            ])
        else:
            risk = TxDT.select(["followup", "pred_risk"]).sort("followup").with_columns([
                pl.lit(i).alias(self.treatment_col)
            ]).select([
                "followup",
                self.treatment_col,
                "pred_risk"
            ])
            
            fup0 = pl.DataFrame({
                "followup": [0],
                self.treatment_col: [i],
                "pred_risk": [0.0]
            }).with_columns([
                pl.col("followup").cast(pl.Int64),
                pl.col(self.treatment_col).cast(pl.Int32)
            ])
        
        risk = pl.concat([fup0, risk])
        risks.append(risk)
    
    out = pl.concat(risks) \
        .with_columns(pl.lit("risk").alias("estimate")) \
        .rename({"pred_risk": "pred"})
    return out
        

def _calculate_survival(self, risk_data):
    if self.bootstrap_nboot > 0:
        surv = risk_data.with_columns([
            (1 - pl.col(col)).alias(col) for col in ["pred", "LCI", "UCI"]
        ]).with_columns(pl.lit("survival").alias("estimate"))
    else: 
        surv = risk_data.with_columns([
            (1 - pl.col("pred")).alias("pred"),
            pl.lit("survival").alias("estimate")
            ])
    return surv
    
