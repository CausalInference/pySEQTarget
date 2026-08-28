"""End-of-follow-up outcomes.

An end-of-follow-up outcome is measured once, at follow-up time
``end_of_fup_time`` (``k``), rather than as a time-to-event. Instead of fitting
a survival outcome model, the estimate in each baseline treatment arm is the
weighted average of the outcome read at ``k`` — weighted by the
period-trial-specific weight at the time the measurement was taken. Ported from
SEQTaRget's ``internal_endoffup.R``.
"""

import numpy as np
import polars as pl
from scipy import stats

from ._risk_estimates import _ci_label


def _eof_frame(self):
    """The frame the estimate is read from.

    Under ``method='censoring'`` the artificially censored (treatment switch)
    rows are not measurements, so a subject who deviates before ``k`` is
    correctly excluded rather than contributing a carried-forward value.
    """
    DT = self.DT
    if self.method == "censoring" and "switch" in DT.columns:
        DT = DT.filter(pl.col("switch") != 1)
    return DT


def _eof_group_cols(self, DT):
    tx_bas = f"{self.treatment_col}{self.indicator_baseline}"
    cols = [tx_bas]
    if self.subgroup_colname is not None and self.subgroup_colname in DT.columns:
        cols.append(self.subgroup_colname)
    return cols


def _eof_measure(self, DT):
    """Select the end-of-follow-up measurement for each (id, trial).

    The measurement at exactly ``k`` when one exists, otherwise — if
    ``end_of_fup_window`` is non-zero — the measurement nearest to ``k`` within
    ``[k - window, k + window]``, with ties (measurements equally far either
    side of ``k``) broken toward the later one, so that at least ``k`` of
    follow-up has elapsed. Trial-periods with no measurement anywhere in the
    window contribute no row, i.e. they are censored out of the estimate.
    """
    k = self.end_of_fup_time
    w = self.end_of_fup_window
    tx_bas = f"{self.treatment_col}{self.indicator_baseline}"

    cols = [self.id_col, "trial", "followup", tx_bas, self.outcome_col]
    if self.subgroup_colname is not None and self.subgroup_colname in DT.columns:
        cols.append(self.subgroup_colname)
    if "weight" in DT.columns:
        cols.append("weight")

    candidates = DT.filter(
        pl.col(self.outcome_col).is_not_null()
        & (pl.col("followup") >= k - w)
        & (pl.col("followup") <= k + w)
    ).select(cols)

    # Nearest to k wins (exact k has distance 0); equidistant ties break toward
    # the later measurement so at least k of follow-up has elapsed. followup is
    # UInt32 in the expanded frame, so cast before differencing — unsigned
    # subtraction wraps for followups below k and .abs() cannot recover it.
    measured = (
        candidates.with_columns(
            (pl.col("followup").cast(pl.Int64) - k).abs().alias("_dist")
        )
        .sort(
            [self.id_col, "trial", "_dist", "followup"],
            descending=[False, False, False, True],
        )
        .group_by([self.id_col, "trial"], maintain_order=True)
        .first()
        .drop("_dist")
        .rename({self.outcome_col: "eof_value"})
    )

    # Unweighted analysis = equal weights
    if "weight" not in measured.columns:
        measured = measured.with_columns(pl.lit(1.0).alias("weight"))
    return measured


def _eof_period_flags(self, DT):
    """One row per trial-period, flagged with what it has available."""
    k = self.end_of_fup_time
    w = self.end_of_fup_window
    valid = pl.col(self.outcome_col).is_not_null()
    in_window = valid & (pl.col("followup") >= k - w) & (pl.col("followup") <= k + w)

    return DT.group_by([self.id_col, "trial"] + _eof_group_cols(self, DT)).agg(
        [
            (valid & (pl.col("followup") == k)).any().alias("at_k"),
            in_window.any().alias("in_window"),
            valid.any().alias("measured"),
        ]
    )


def _eof_estimate(self):
    """Weighted end-of-follow-up average within each treatment arm.

    Weight truncation (``weight_min`` / ``weight_max``, including the bounds
    ``weight_p99`` resolves to) is applied here as it is for the outcome model,
    since this average is the estimator in end_of_fup mode.

    Alongside the estimate this counts the trial-periods censored for want of a
    measurement in the window — those measured at some point but not within
    ``[k - window, k + window]`` — so that share can be reported next to the
    estimate they were dropped from. Trial-periods never measured at all are
    counted separately rather than folded in, so the analysed, censored and
    never-measured counts partition the eligible total.
    """
    DT = _eof_frame(self)
    by = _eof_group_cols(self, DT)

    measured = _eof_measure(self, DT).with_columns(
        pl.col("weight").clip(lower_bound=self.weight_min, upper_bound=self.weight_max)
    )

    est = measured.group_by(by).agg(
        [
            (
                (pl.col("weight") * pl.col("eof_value")).sum() / pl.col("weight").sum()
            ).alias("estimate"),
            pl.len().alias("n"),
            pl.col(self.id_col).n_unique().alias("n_subjects"),
        ]
    )
    totals = (
        _eof_period_flags(self, DT)
        .group_by(by)
        .agg(
            [
                pl.len().alias("n_eligible"),
                (pl.col("measured") & ~pl.col("in_window")).sum().alias("n_censored"),
                (~pl.col("measured")).sum().alias("n_nomeasure"),
            ]
        )
    )
    return est.join(totals, on=by, how="inner").sort(by)


def _eof_counts(self, DT, unique):
    """Account for every trial-period at the end-of-follow-up time.

    Four mutually exclusive categories against the Eligible total: measured
    ``At k``; measured ``In window`` (no measurement at k but one within the
    window); ``Excluded (outside window)`` (measured somewhere, but not within
    the window); ``Excluded (no measurement)``. Trial-period counts partition
    Eligible; subject counts (``unique=True``) need not, since one subject can
    fall into different categories for different trials.
    """
    by = _eof_group_cols(self, DT)
    levels = [
        "At k",
        "In window",
        "Excluded (outside window)",
        "Excluded (no measurement)",
    ]

    flags = _eof_period_flags(self, DT).with_columns(
        pl.when(pl.col("at_k"))
        .then(pl.lit(levels[0]))
        .when(pl.col("in_window"))
        .then(pl.lit(levels[1]))
        .when(pl.col("measured"))
        .then(pl.lit(levels[2]))
        .otherwise(pl.lit(levels[3]))
        .alias("_category")
    )

    counter = pl.col(self.id_col).n_unique() if unique else pl.len()
    counted = flags.group_by(by + ["_category"]).agg(counter.alias("N"))
    wide = counted.pivot(on="_category", index=by, values="N").fill_null(0)
    for lv in levels:
        if lv not in wide.columns:
            wide = wide.with_columns(pl.lit(0).alias(lv))
    totals = flags.group_by(by).agg(counter.alias("Eligible"))
    return wide.join(totals, on=by).select(by + ["Eligible"] + levels).sort(by)


def _eof_summary(self, DT):
    """N, mean and SD of the raw selected measurements per baseline arm — the
    unweighted analogue of the outcome count tables, reported for continuous
    outcomes where event counts have no meaning."""
    by = _eof_group_cols(self, DT)
    measured = _eof_measure(self, DT)
    return (
        measured.group_by(by)
        .agg(
            [
                pl.len().alias("N"),
                pl.col("eof_value").mean().alias("Mean"),
                pl.col("eof_value").std().alias("SD"),
            ]
        )
        .sort(by)
        .rename({by[0]: "A"})
    )


def _create_endoffup(self):
    """Assemble end-of-follow-up estimates and bootstrap confidence intervals.

    Mirrors the risk assembly: ``eof_data`` holds the per-arm estimate and
    ``eof_comparison`` the pairwise between-arm contrasts, both with bootstrap
    confidence intervals when bootstrapped. Contrasts are paired by bootstrap
    iteration, so the interval accounts for the correlation between arms.
    """
    tx_bas = f"{self.treatment_col}{self.indicator_baseline}"
    sub = self.subgroup_colname
    is_binary = self.end_of_fup_type == "binary"
    label = "Proportion" if is_binary else "Mean"
    ci_label = _ci_label(self.bootstrap_CI)
    z = stats.norm.ppf(1 - (1 - self.bootstrap_CI) / 2)
    alpha = (1 - self.bootstrap_CI) / 2
    use_se = self.bootstrap_CI_method == "se"

    full = self.outcome_model[0]["eof"]
    boots = [
        m["eof"]
        for m in self.outcome_model[1:]
        if m["eof"] is not None and m["eof"].height > 0
    ]
    has_ci = len(boots) > 1
    sub_in_full = sub is not None and sub in full.columns
    key_cols = ([sub] if sub_in_full else []) + [tx_bas]

    def boot_estimates(key):
        """Per-iteration estimates for one (subgroup, arm) key, NaN when absent."""
        out = np.full(len(boots), np.nan)
        for i, b in enumerate(boots):
            f = b
            for col, val in zip(key_cols, key):
                f = f.filter(pl.col(col) == val)
            if f.height == 1:
                out[i] = f["estimate"][0]
        return out

    keys = [tuple(row) for row in full.select(key_cols).iter_rows()]

    # Per-arm table ==========================================================
    data = full.rename(
        {
            tx_bas: "A",
            "estimate": label,
            "n_eligible": "Trial-periods (Eligible)",
            "n": "Trial-periods (Analysed)",
            "n_censored": "Trial-periods (Censored)",
            "n_nomeasure": "Trial-periods (No measurement)",
            "n_subjects": "Subjects",
        }
    ).with_columns(
        [
            pl.lit(self.end_of_fup_type).alias("Type"),
            pl.lit(float(self.end_of_fup_time)).alias("Time"),
            (
                100
                * pl.col("Trial-periods (Censored)")
                / pl.col("Trial-periods (Eligible)")
            ).alias("% Censored"),
        ]
    )

    if has_ci:
        se_rows, lci_rows, uci_rows = [], [], []
        for key in keys:
            draws = boot_estimates(key)
            se = float(np.nanstd(draws, ddof=1))
            point = float(
                full.filter(
                    pl.all_horizontal([pl.col(c) == v for c, v in zip(key_cols, key)])
                )["estimate"][0]
            )
            if use_se:
                lci, uci = point - z * se, point + z * se
            else:
                lci = float(np.nanquantile(draws, alpha))
                uci = float(np.nanquantile(draws, 1 - alpha))
            if is_binary:
                # Proportions are bounded, so clamp the binary interval to [0, 1]
                lci, uci = max(0.0, lci), min(1.0, uci)
            se_rows.append(se)
            lci_rows.append(lci)
            uci_rows.append(uci)
        ci_frame = pl.DataFrame(
            {
                **{c: [k[i] for k in keys] for i, c in enumerate(key_cols)},
                "SE": se_rows,
                f"{ci_label} LCI": lci_rows,
                f"{ci_label} UCI": uci_rows,
            }
        )
        rename_back = {key_cols[-1]: "A"}
        data = data.join(
            ci_frame.rename(rename_back), on=(["A"] if not sub_in_full else [sub, "A"])
        )

    lead = (
        ["Type", "Time"]
        + ([sub] if sub_in_full else [])
        + [
            "A",
            label,
            "Trial-periods (Eligible)",
            "Trial-periods (Analysed)",
            "Trial-periods (Censored)",
            "Trial-periods (No measurement)",
            "% Censored",
            "Subjects",
        ]
    )
    data = data.select(lead + [c for c in data.columns if c not in lead])

    # Pairwise between-arm contrasts, both directions ========================
    rows = []
    subgroup_vals = sorted(set(full[sub].to_list())) if sub_in_full else [None]
    for g in subgroup_vals:
        sub_frame = full.filter(pl.col(sub) == g) if g is not None else full
        arms = sub_frame[tx_bas].to_list()
        point = dict(zip(arms, sub_frame["estimate"].to_list()))
        for a_x in arms:
            for a_y in arms:
                if a_x == a_y:
                    continue
                row = {"Time": float(self.end_of_fup_time)}
                if g is not None:
                    row[sub] = g
                row["A_x"] = a_x
                row["A_y"] = a_y
                diff = point[a_y] - point[a_x]
                ratio = point[a_y] / point[a_x] if point[a_x] != 0 else float("inf")
                row["Difference"] = diff
                if has_ci:
                    kx = ((g,) if g is not None else ()) + (a_x,)
                    ky = ((g,) if g is not None else ()) + (a_y,)
                    d = boot_estimates(ky) - boot_estimates(kx)
                    d_se = float(np.nanstd(d, ddof=1))
                    with np.errstate(divide="ignore", invalid="ignore"):
                        r = boot_estimates(ky) / boot_estimates(kx)
                    r_valid = r[np.isfinite(r) & (r > 0)]
                    r_logse = (
                        float(np.std(np.log(r_valid), ddof=1))
                        if r_valid.size > 1
                        else float("nan")
                    )
                    if use_se:
                        d_lci, d_uci = diff - z * d_se, diff + z * d_se
                        if np.isfinite(r_logse) and ratio > 0 and np.isfinite(ratio):
                            r_lci = float(np.exp(np.log(ratio) - z * r_logse))
                            r_uci = float(np.exp(np.log(ratio) + z * r_logse))
                        else:
                            r_lci = r_uci = float("nan")
                    else:
                        d_lci = float(np.nanquantile(d, alpha))
                        d_uci = float(np.nanquantile(d, 1 - alpha))
                        if r_valid.size > 1:
                            r_lci = float(np.quantile(r_valid, alpha))
                            r_uci = float(np.quantile(r_valid, 1 - alpha))
                        else:
                            r_lci = r_uci = float("nan")
                    row[f"Difference {ci_label} LCI"] = d_lci
                    row[f"Difference {ci_label} UCI"] = d_uci
                    row["Difference SE"] = d_se
                # Ratios need an outcome bounded away from zero, so they are
                # reported for proportions only
                if is_binary:
                    row["Ratio"] = ratio
                    if has_ci:
                        row[f"Ratio {ci_label} LCI"] = r_lci
                        row[f"Ratio {ci_label} UCI"] = r_uci
                        row["log(Ratio) SE"] = r_logse
                rows.append(row)

    comparison = pl.DataFrame(rows) if rows else pl.DataFrame()
    return {"eof_data": data, "eof_comparison": comparison}
