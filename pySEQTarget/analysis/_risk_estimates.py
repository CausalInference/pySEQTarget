import math

import polars as pl
from scipy import stats


def _ci_label(bootstrap_CI):
    """Format the CI level as a column-name fragment, e.g. 0.95 -> '95%'."""
    return f"{bootstrap_CI * 100:g}%"


def _compute_rd_rr(comp, has_bootstrap, z=None, group_cols=None, ci_label="95%"):
    """
    Compute Risk Difference and Risk Ratio from a comparison dataframe.
    Fallback used when paired bootstrap data is unavailable (e.g. subgroups).
    """
    if group_cols is None:
        group_cols = []

    if has_bootstrap:
        rd_se = (pl.col("se_x").pow(2) + pl.col("se_y").pow(2)).sqrt()
        rd_lci_lab = f"RD {ci_label} LCI"
        rd_uci_lab = f"RD {ci_label} UCI"
        rd_comp = comp.with_columns(
            [
                (pl.col("risk_x") - pl.col("risk_y")).alias("Risk Difference"),
                (pl.col("risk_x") - pl.col("risk_y") - z * rd_se).alias(rd_lci_lab),
                (pl.col("risk_x") - pl.col("risk_y") + z * rd_se).alias(rd_uci_lab),
                rd_se.alias("RD SE"),
            ]
        )
        rd_comp = rd_comp.drop(["risk_x", "risk_y", "se_x", "se_y"])
        col_order = group_cols + [
            "A_x",
            "A_y",
            "Risk Difference",
            rd_lci_lab,
            rd_uci_lab,
            "RD SE",
        ]
        rd_comp = rd_comp.select([c for c in col_order if c in rd_comp.columns])

        rr_log_se = (
            (pl.col("se_x") / pl.col("risk_x")).pow(2)
            + (pl.col("se_y") / pl.col("risk_y")).pow(2)
        ).sqrt()
        rr_lci_lab = f"RR {ci_label} LCI"
        rr_uci_lab = f"RR {ci_label} UCI"
        rr_comp = comp.with_columns(
            [
                (pl.col("risk_x") / pl.col("risk_y")).alias("Risk Ratio"),
                ((pl.col("risk_x") / pl.col("risk_y")) * (-z * rr_log_se).exp()).alias(
                    rr_lci_lab
                ),
                ((pl.col("risk_x") / pl.col("risk_y")) * (z * rr_log_se).exp()).alias(
                    rr_uci_lab
                ),
                rr_log_se.alias("log(RR) SE"),
            ]
        )
        rr_comp = rr_comp.drop(["risk_x", "risk_y", "se_x", "se_y"])
        col_order = group_cols + [
            "A_x",
            "A_y",
            "Risk Ratio",
            rr_lci_lab,
            rr_uci_lab,
            "log(RR) SE",
        ]
        rr_comp = rr_comp.select([c for c in col_order if c in rr_comp.columns])
    else:
        rd_comp = comp.with_columns(
            (pl.col("risk_x") - pl.col("risk_y")).alias("Risk Difference")
        )
        rd_comp = rd_comp.drop(["risk_x", "risk_y"])
        col_order = group_cols + ["A_x", "A_y", "Risk Difference"]
        rd_comp = rd_comp.select([c for c in col_order if c in rd_comp.columns])

        rr_comp = comp.with_columns(
            (pl.col("risk_x") / pl.col("risk_y")).alias("Risk Ratio")
        )
        rr_comp = rr_comp.drop(["risk_x", "risk_y"])
        col_order = group_cols + ["A_x", "A_y", "Risk Ratio"]
        rr_comp = rr_comp.select([c for c in col_order if c in rr_comp.columns])

    return rd_comp, rr_comp


def _resolve_risk_times(grid, risk_times):
    """
    Snap each requested risk time to the latest available followup at or before
    it, always including the maximum followup. Returns a sorted list of followup
    values that exist in ``grid``.
    """
    grid = sorted(set(grid))
    final = grid[-1]

    if risk_times is None:
        return [final]

    req = risk_times if isinstance(risk_times, (list, tuple)) else [risk_times]
    req = [float(t) for t in req if t is not None]
    if not req:
        return [final]

    above = [t for t in req if t > final]
    if above:
        raise ValueError(
            f"risk_times value(s) exceed the maximum followup ({final}): {above}"
        )
    below = [t for t in req if t < grid[0]]
    if below:
        raise ValueError(
            f"risk_times value(s) below the minimum followup ({grid[0]}): {below}"
        )

    snapped = [max(g for g in grid if g <= t) for t in req]
    return sorted(set(snapped + [final]))


def _risk_estimates(self):
    risk_all = self.km_data.filter(pl.col("estimate") == "risk")
    report_times = _resolve_risk_times(
        risk_all["followup"].unique().to_list(), self.risk_times
    )

    group_cols = [self.subgroup_colname] if self.subgroup_colname else []
    has_bootstrap = self.bootstrap_nboot > 0

    # Paired approach: compute RD_i and RR_i per bootstrap iteration then take
    # SE or percentile of those — correct because arms share the same bootstrap
    # samples, so pairing captures their correlation. Falls back to the
    # independent delta method for subgroups (not yet supported) or when
    # _boot_risks is unavailable.
    use_paired = (
        has_bootstrap
        and not group_cols
        and hasattr(self, "_boot_risks")
        and all(tx in self._boot_risks for tx in self.treatment_level)
    )

    if has_bootstrap:
        alpha = 1 - self.bootstrap_CI
        z = stats.norm.ppf(1 - alpha / 2)
        ci_label = _ci_label(self.bootstrap_CI)
    else:
        z = None
        alpha = None
        ci_label = _ci_label(self.bootstrap_CI)

    rd_comparisons = []
    rr_comparisons = []

    for followup_t in report_times:
        risk = risk_all.filter(pl.col("followup") == followup_t)

        risk_by_level = {}
        for tx in self.treatment_level:
            level_data = risk.filter(pl.col(self.treatment_col) == tx)
            risk_by_level[tx] = {"pred": level_data.select(group_cols + ["pred"])}
            if has_bootstrap and not use_paired:
                risk_by_level[tx]["SE"] = level_data.select(group_cols + ["SE"])

        for tx_x in self.treatment_level:
            for tx_y in self.treatment_level:
                if tx_x == tx_y:
                    continue

                if use_paired:
                    boot_x = (
                        self._boot_risks[tx_x]
                        .filter(pl.col("followup") == followup_t)
                        .select(["boot_idx", pl.col("risk").alias("risk_x")])
                    )
                    boot_y = (
                        self._boot_risks[tx_y]
                        .filter(pl.col("followup") == followup_t)
                        .select(["boot_idx", pl.col("risk").alias("risk_y")])
                    )
                    paired = boot_x.join(boot_y, on="boot_idx").with_columns(
                        (pl.col("risk_x") - pl.col("risk_y")).alias("RD")
                    )

                    risk_x_val = float(risk_by_level[tx_x]["pred"]["pred"][0])
                    risk_y_val = float(risk_by_level[tx_y]["pred"]["pred"][0])
                    rd_point = risk_x_val - risk_y_val
                    rr_point = (
                        risk_x_val / risk_y_val if risk_y_val != 0 else float("inf")
                    )

                    # Filter degenerate RR bootstrap values (risk_y == 0 or negative)
                    valid_rr = paired.filter(
                        (pl.col("risk_y") > 0) & (pl.col("risk_x") >= 0)
                    ).with_columns((pl.col("risk_x") / pl.col("risk_y")).alias("RR"))

                    n_valid_rr = len(valid_rr)

                    # Bootstrap SEs, retained regardless of CI method: the risk
                    # difference SE is on the natural scale, the risk ratio SE
                    # on the log scale (the scale ratio measures are pooled on
                    # for inverse-variance meta-analysis: combine the log Risk
                    # Ratio with log(RR) SE, then exponentiate).
                    rd_se = float(paired["RD"].std())
                    log_rr_se = (
                        float(valid_rr["RR"].log().std())
                        if n_valid_rr >= 2
                        else float("nan")
                    )

                    if self.bootstrap_CI_method == "percentile":
                        rd_lci = float(paired["RD"].quantile(alpha / 2))
                        rd_uci = float(paired["RD"].quantile(1 - alpha / 2))
                        if n_valid_rr >= 2:
                            rr_lci = float(valid_rr["RR"].quantile(alpha / 2))
                            rr_uci = float(valid_rr["RR"].quantile(1 - alpha / 2))
                        else:
                            rr_lci = float("nan")
                            rr_uci = float("nan")
                    else:
                        rd_lci = rd_point - z * rd_se
                        rd_uci = rd_point + z * rd_se
                        if n_valid_rr >= 2 and rr_point > 0:
                            rr_lci = math.exp(math.log(rr_point) - z * log_rr_se)
                            rr_uci = math.exp(math.log(rr_point) + z * log_rr_se)
                        else:
                            rr_lci = float("nan")
                            rr_uci = float("nan")

                    rd_comp = pl.DataFrame(
                        {
                            "Followup": [followup_t],
                            "A_x": [tx_x],
                            "A_y": [tx_y],
                            "Risk Difference": [rd_point],
                            f"RD {ci_label} LCI": [rd_lci],
                            f"RD {ci_label} UCI": [rd_uci],
                            "RD SE": [rd_se],
                        }
                    )
                    rr_comp = pl.DataFrame(
                        {
                            "Followup": [followup_t],
                            "A_x": [tx_x],
                            "A_y": [tx_y],
                            "Risk Ratio": [rr_point],
                            f"RR {ci_label} LCI": [rr_lci],
                            f"RR {ci_label} UCI": [rr_uci],
                            "log(RR) SE": [log_rr_se],
                        }
                    )
                else:
                    # Fall back to independent delta method
                    risk_x = risk_by_level[tx_x]["pred"].rename({"pred": "risk_x"})
                    risk_y = risk_by_level[tx_y]["pred"].rename({"pred": "risk_y"})

                    if group_cols:
                        comp = risk_x.join(risk_y, on=group_cols, how="left")
                    else:
                        comp = risk_x.join(risk_y, how="cross")

                    comp = comp.with_columns(
                        [pl.lit(tx_x).alias("A_x"), pl.lit(tx_y).alias("A_y")]
                    )

                    if has_bootstrap:
                        se_x = risk_by_level[tx_x]["SE"].rename({"SE": "se_x"})
                        se_y = risk_by_level[tx_y]["SE"].rename({"SE": "se_y"})
                        if group_cols:
                            comp = comp.join(se_x, on=group_cols, how="left")
                            comp = comp.join(se_y, on=group_cols, how="left")
                        else:
                            comp = comp.join(se_x, how="cross")
                            comp = comp.join(se_y, how="cross")

                    rd_comp, rr_comp = _compute_rd_rr(
                        comp, has_bootstrap, z, group_cols, ci_label
                    )
                    rd_cols = rd_comp.columns
                    rr_cols = rr_comp.columns
                    rd_comp = rd_comp.with_columns(
                        pl.lit(followup_t).alias("Followup")
                    ).select(["Followup"] + rd_cols)
                    rr_comp = rr_comp.with_columns(
                        pl.lit(followup_t).alias("Followup")
                    ).select(["Followup"] + rr_cols)

                rd_comparisons.append(rd_comp)
                rr_comparisons.append(rr_comp)

    risk_difference = pl.concat(rd_comparisons) if rd_comparisons else pl.DataFrame()
    risk_ratio = pl.concat(rr_comparisons) if rr_comparisons else pl.DataFrame()

    return {"risk_difference": risk_difference, "risk_ratio": risk_ratio}
