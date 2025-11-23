import polars as pl
import numpy as np
from ..helpers import _predict_model

def _get_outcome_predictions(self, TxDT, idx=None):
    data = TxDT.to_pandas()
    predictions = []
    for boot_model in self.outcome_model:
        model = boot_model[idx] if idx is not None else boot_model
        pred = model.predict(data)
        predictions.append(pred)
    return predictions

def _pred_risk(self):
    has_subgroups = (isinstance(self.outcome_model[0], list) if self.outcome_model else False)
    
    if not has_subgroups:
        return _calculate_risk(self, self.DT, idx=None, val=None)
    
    all_risks = []
    original_DT = self.DT
    
    for i, val in enumerate(self._unique_subgroups):
        subgroup_DT = original_DT.filter(pl.col(self.subgroup_colname) == val)
        risk = _calculate_risk(self, subgroup_DT, i, val)
        all_risks.append(risk)
    
    self.DT = original_DT
    return pl.concat(all_risks)

def _calculate_risk(self, data, idx=None, val=None):
    a = 1 - self.bootstrap_CI
    lci = a / 2
    uci = 1 - lci
    
    SDT = (
        data
        .with_columns([(pl.col(self.id_col).cast(pl.Utf8) + pl.col("trial").cast(pl.Utf8)).alias("TID")])
        .group_by("TID").first()
        .drop(["followup", f"followup{self.indicator_squared}"])
        .with_columns([pl.lit(list(range(self.followup_max))).alias("followup")])
        .explode("followup")
        .with_columns([
            (pl.col("followup") + 1).alias("followup"),
            (pl.col("followup") ** 2).alias(f"followup{self.indicator_squared}")
        ])
    ).sort([self.id_col, "trial", "followup"])
    
    risks = []
    for treatment_val in self.treatment_level:
        TxDT = SDT.with_columns([pl.lit(treatment_val).alias(f"{self.treatment_col}{self.indicator_baseline}")])
        
        if self.method == "dose-response":
            print("REMEMBER TO FIX DOSE-REPONSE RISK CALCULATION TODO")
        if self.compevent_colname is not None:
            pass
        
        preds = _get_outcome_predictions(self, TxDT, idx=idx)
        pred_series = [pl.Series("pred_risk", preds[0])]
        
        if self.bootstrap_nboot > 0:
            for boot_idx, pred in enumerate(preds[1:], start=1):
                pred_series.append(pl.Series(f"pred_risk_{boot_idx}", pred))
        
        names = [s.name for s in pred_series]
        
        TxDT = (
            TxDT.with_columns(pred_series)
            .with_columns([(1 - pl.col(col)).cum_prod().over("TID").alias(col) for col in names])
            .group_by("followup").agg([pl.col(col).mean() for col in names])
            .sort("followup")
            .with_columns([(1 - pl.col(col)).alias(col) for col in names])
        )
        
        boot_cols = [col for col in names if col != "pred_risk"]
        
        if boot_cols:
            risk = (
                TxDT.select(["followup"] + boot_cols)
                .unpivot(index="followup", on=boot_cols, variable_name="bootID", value_name="risk")
                .group_by("followup").agg([
                    pl.col("risk").std().alias("SE"),
                    pl.col("risk").quantile(lci).alias("LCI"),
                    pl.col("risk").quantile(uci).alias("UCI")
                ])
                .join(TxDT.select(["followup", "pred_risk"]), on="followup")
            )
            
            if self.bootstrap_CI_method == "se":
                from scipy.stats import norm
                z = norm.ppf(1 - a / 2)
                risk = risk.with_columns([
                    (pl.col("pred_risk") - z * pl.col("SE")).alias("LCI"),
                    (pl.col("pred_risk") + z * pl.col("SE")).alias("UCI")
                ])
            
            risk = risk.select([
                "followup",
                pl.lit(treatment_val).alias(self.treatment_col),
                "pred_risk", "SE", "LCI", "UCI"
            ])
            
            fup0 = pl.DataFrame({
                "followup": [0], self.treatment_col: [treatment_val],
                "pred_risk": [0.0], "SE": [0.0], "LCI": [0.0], "UCI": [0.0]
            }).with_columns([
                pl.col("followup").cast(pl.Int64),
                pl.col(self.treatment_col).cast(pl.Int32)
            ])
        else:
            risk = TxDT.select(["followup", pl.lit(treatment_val).alias(self.treatment_col), "pred_risk"])
            fup0 = pl.DataFrame({"followup": [0], 
                                 self.treatment_col: [treatment_val], "pred_risk": [0.0]}
                                ).with_columns([
                pl.col("followup").cast(pl.Int64),
                pl.col(self.treatment_col).cast(pl.Int32)
            ])
        
        risks.append(pl.concat([fup0, risk]))
    
    out = pl.concat(risks).with_columns(pl.lit("risk").alias("estimate"))
    if val is not None:
        out = out.with_columns(pl.lit(val).alias(self.subgroup_colname))
    
    return out.rename({"pred_risk": "pred"})
        

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
    
