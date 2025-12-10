import polars as pl


def _get_outcome_predictions(self, TxDT, idx=None):
    data = TxDT.to_pandas()
    predictions = {"outcome": []}
    if self.compevent_colname is not None:
        predictions["compevent"] = []

    for boot_model in self.outcome_model:
        model_dict = boot_model[idx] if idx is not None else boot_model
        predictions["outcome"].append(model_dict["outcome"].predict(data))
        if self.compevent_colname is not None:
            predictions["compevent"].append(model_dict["compevent"].predict(data))

    return predictions


def _pred_risk(self):
    has_subgroups = (
        isinstance(self.outcome_model[0], list) if self.outcome_model else False
    )

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
        data.with_columns(
            [
                (
                    pl.col(self.id_col).cast(pl.Utf8) + pl.col("trial").cast(pl.Utf8)
                ).alias("TID")
            ]
        )
        .group_by("TID")
        .first()
        .drop(["followup", f"followup{self.indicator_squared}"])
        .with_columns([pl.lit(list(range(self.followup_max))).alias("followup")])
        .explode("followup")
        .with_columns(
            [
                (pl.col("followup") + 1).alias("followup"),
                (pl.col("followup") ** 2).alias(f"followup{self.indicator_squared}"),
            ]
        )
    ).sort([self.id_col, "trial", "followup"])

    risks = []
    for treatment_val in self.treatment_level:
        TxDT = SDT.with_columns(
            [
                pl.lit(treatment_val).alias(
                    f"{self.treatment_col}{self.indicator_baseline}"
                )
            ]
        )

        if self.method == "dose-response":
            if treatment_val == self.treatment_level[0]:
                TxDT = TxDT.with_columns(
                    [pl.lit(0.0).alias("dose"), pl.lit(0.0).alias("dose_sq")]
                )
            else:
                TxDT = TxDT.with_columns(
                    [
                        pl.col("followup").alias("dose"),
                        pl.col(f"followup{self.indicator_squared}").alias("dose_sq"),
                    ]
                )

        preds = _get_outcome_predictions(self, TxDT, idx=idx)
        pred_series = [pl.Series("pred_outcome", preds["outcome"][0])]

        if self.bootstrap_nboot > 0:
            for boot_idx, pred in enumerate(preds["outcome"][1:], start=1):
                pred_series.append(pl.Series(f"pred_outcome_{boot_idx}", pred))

        if self.compevent_colname is not None:
            pred_series.append(pl.Series("pred_ce", preds["compevent"][0]))
            if self.bootstrap_nboot > 0:
                for boot_idx, pred in enumerate(preds["compevent"][1:], start=1):
                    pred_series.append(pl.Series(f"pred_ce_{boot_idx}", pred))

        outcome_names = [s.name for s in pred_series if "outcome" in s.name]
        ce_names = [s.name for s in pred_series if "ce" in s.name]

        TxDT = TxDT.with_columns(pred_series)

        if self.compevent_colname is not None:
            for out_col, ce_col in zip(outcome_names, ce_names):
                surv_col = out_col.replace("pred_outcome", "surv")
                cce_col = out_col.replace("pred_outcome", "cce")
                inc_col = out_col.replace("pred_outcome", "inc")

                TxDT = TxDT.with_columns(
                    [
                        (1 - pl.col(out_col)).cum_prod().over("TID").alias(surv_col),
                        ((1 - pl.col(out_col)) * (1 - pl.col(ce_col)))
                        .cum_prod()
                        .over("TID")
                        .alias(cce_col),
                    ]
                ).with_columns(
                    [
                        (pl.col(out_col) * (1 - pl.col(ce_col)) * pl.col(cce_col))
                        .cum_sum()
                        .over("TID")
                        .alias(inc_col)
                    ]
                )

            surv_names = [n.replace("pred_outcome", "surv") for n in outcome_names]
            inc_names = [n.replace("pred_outcome", "inc") for n in outcome_names]
            TxDT = (
                TxDT.group_by("followup")
                .agg([pl.col(col).mean() for col in surv_names + inc_names])
                .sort("followup")
            )
            main_col = "surv"
            boot_cols = [col for col in surv_names if col != "surv"]
        else:
            TxDT = (
                TxDT.with_columns(
                    [
                        (1 - pl.col(col)).cum_prod().over("TID").alias(col)
                        for col in outcome_names
                    ]
                )
                .group_by("followup")
                .agg([pl.col(col).mean() for col in outcome_names])
                .sort("followup")
                .with_columns([(1 - pl.col(col)).alias(col) for col in outcome_names])
            )
            main_col = "pred_outcome"
            boot_cols = [col for col in outcome_names if col != "pred_outcome"]

        if boot_cols:
            risk = (
                TxDT.select(["followup"] + boot_cols)
                .unpivot(
                    index="followup",
                    on=boot_cols,
                    variable_name="bootID",
                    value_name="risk",
                )
                .group_by("followup")
                .agg(
                    [
                        pl.col("risk").std().cast(pl.Float64).alias("SE"),
                        pl.col("risk").quantile(lci).cast(pl.Float64).alias("LCI"),
                        pl.col("risk").quantile(uci).cast(pl.Float64).alias("UCI"),
                    ]
                )
                .join(TxDT.select(["followup", main_col]), on="followup")
            )

            if self.bootstrap_CI_method == "se":
                from scipy.stats import norm

                z = norm.ppf(1 - a / 2)
                risk = risk.with_columns(
                    [
                        (pl.col(main_col) - z * pl.col("SE")).alias("LCI"),
                        (pl.col(main_col) + z * pl.col("SE")).alias("UCI"),
                    ]
                )

            fup0_val = 1.0 if self.compevent_colname else 0.0

            if self.compevent_colname is not None:
                inc_boot_cols = [col for col in inc_names if col != "inc"]
                if inc_boot_cols:
                    inc_risk = (
                        TxDT.select(["followup"] + inc_boot_cols)
                        .unpivot(
                            index="followup",
                            on=inc_boot_cols,
                            variable_name="bootID",
                            value_name="inc_val",
                        )
                        .group_by("followup")
                        .agg(
                            [
                                pl.col("inc_val")
                                .std()
                                .cast(pl.Float64)
                                .alias("inc_SE"),
                                pl.col("inc_val")
                                .quantile(lci)
                                .cast(pl.Float64)
                                .alias("inc_LCI"),
                                pl.col("inc_val")
                                .quantile(uci)
                                .cast(pl.Float64)
                                .alias("inc_UCI"),
                            ]
                        )
                        .join(TxDT.select(["followup", "inc"]), on="followup")
                    )
                    risk = risk.join(inc_risk, on="followup")
                    final_cols = [
                        "followup",
                        main_col,
                        "SE",
                        "LCI",
                        "UCI",
                        "inc",
                        "inc_SE",
                        "inc_LCI",
                        "inc_UCI",
                    ]
                    risk = risk.select(final_cols).with_columns(
                        pl.lit(treatment_val).alias(self.treatment_col)
                    )

                    fup0 = pl.DataFrame(
                        {
                            "followup": [0],
                            main_col: [fup0_val],
                            "SE": [0.0],
                            "LCI": [fup0_val],
                            "UCI": [fup0_val],
                            "inc": [0.0],
                            "inc_SE": [0.0],
                            "inc_LCI": [0.0],
                            "inc_UCI": [0.0],
                            self.treatment_col: [treatment_val],
                        }
                    ).with_columns(
                        [
                            pl.col("followup").cast(pl.Int64),
                            pl.col(self.treatment_col).cast(pl.Int32),
                        ]
                    )
                else:
                    risk = risk.select(
                        ["followup", main_col, "SE", "LCI", "UCI"]
                    ).with_columns(pl.lit(treatment_val).alias(self.treatment_col))
                    fup0 = pl.DataFrame(
                        {
                            "followup": [0],
                            main_col: [fup0_val],
                            "SE": [0.0],
                            "LCI": [fup0_val],
                            "UCI": [fup0_val],
                            self.treatment_col: [treatment_val],
                        }
                    ).with_columns(
                        [
                            pl.col("followup").cast(pl.Int64),
                            pl.col(self.treatment_col).cast(pl.Int32),
                        ]
                    )
            else:
                risk = risk.select(
                    ["followup", main_col, "SE", "LCI", "UCI"]
                ).with_columns(pl.lit(treatment_val).alias(self.treatment_col))
                fup0 = pl.DataFrame(
                    {
                        "followup": [0],
                        main_col: [fup0_val],
                        "SE": [0.0],
                        "LCI": [fup0_val],
                        "UCI": [fup0_val],
                        self.treatment_col: [treatment_val],
                    }
                ).with_columns(
                    [
                        pl.col("followup").cast(pl.Int64),
                        pl.col(self.treatment_col).cast(pl.Int32),
                    ]
                )
        else:
            fup0_val = 1.0 if self.compevent_colname else 0.0
            risk = TxDT.select(["followup", main_col]).with_columns(
                pl.lit(treatment_val).alias(self.treatment_col)
            )
            fup0 = pl.DataFrame(
                {
                    "followup": [0],
                    main_col: [fup0_val],
                    self.treatment_col: [treatment_val],
                }
            ).with_columns(
                [
                    pl.col("followup").cast(pl.Int64),
                    pl.col(self.treatment_col).cast(pl.Int32),
                ]
            )

            if self.compevent_colname is not None:
                risk = risk.join(TxDT.select(["followup", "inc"]), on="followup")
                fup0 = fup0.with_columns([pl.lit(0.0).alias("inc")])

        risks.append(pl.concat([fup0, risk]))
    out = pl.concat(risks)

    if self.compevent_colname is not None:
        has_ci = "SE" in out.columns

        surv_cols = ["followup", self.treatment_col, "surv"]
        if has_ci:
            surv_cols.extend(["SE", "LCI", "UCI"])
        surv_out = (
            out.select(surv_cols)
            .rename({"surv": "pred"})
            .with_columns(pl.lit("survival").alias("estimate"))
        )

        risk_cols = ["followup", self.treatment_col, (1 - pl.col("surv")).alias("pred")]
        if has_ci:
            risk_cols.extend(
                [
                    pl.col("SE"),
                    (1 - pl.col("UCI")).alias("LCI"),
                    (1 - pl.col("LCI")).alias("UCI"),
                ]
            )
        risk_out = out.select(risk_cols).with_columns(pl.lit("risk").alias("estimate"))

        inc_cols = ["followup", self.treatment_col, pl.col("inc").alias("pred")]
        if has_ci:
            inc_cols.extend(
                [
                    pl.col("inc_SE").alias("SE"),
                    pl.col("inc_LCI").alias("LCI"),
                    pl.col("inc_UCI").alias("UCI"),
                ]
            )
        inc_out = out.select(inc_cols).with_columns(
            pl.lit("incidence").alias("estimate")
        )

        out = pl.concat([surv_out, risk_out, inc_out])
    else:
        out = out.rename({"pred_outcome": "pred"}).with_columns(
            pl.lit("risk").alias("estimate")
        )

    if val is not None:
        out = out.with_columns(pl.lit(val).alias(self.subgroup_colname))

    return out


def _calculate_survival(self, risk_data):
    if self.bootstrap_nboot > 0:
        surv = risk_data.with_columns(
            [(1 - pl.col(col)).alias(col) for col in ["pred", "LCI", "UCI"]]
        ).with_columns(pl.lit("survival").alias("estimate"))
    else:
        surv = risk_data.with_columns(
            [(1 - pl.col("pred")).alias("pred"), pl.lit("survival").alias("estimate")]
        )
    return surv


def _clamp(data):
    """Clamp prediction and CI columns to [0, 1] bounds."""
    cols = ["pred", "LCI", "UCI"]
    exists = [c for c in cols if c in data.columns]

    return data.with_columns([pl.col(col).clip(0.0, 1.0) for col in exists])
