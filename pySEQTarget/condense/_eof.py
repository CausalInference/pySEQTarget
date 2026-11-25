from types import SimpleNamespace

import numpy as np
import polars as pl

from ..analysis._endoffup import (_create_endoffup, _eof_counts, _eof_frame,
                                  _eof_group_cols, _eof_measure,
                                  _eof_summary_of)
from ..weighting._weight_stats import _weight_bounds


def _eof_last_period(self, b):
    return _eof_frame(b)["period"].max()


def _eof_diagnostics(self, b, last):
    # the package's end-of-follow-up count tables for one batch
    DT = _eof_frame(b)
    return {
        "unique_eof": _eof_counts(self, DT, True, last),
        "nonunique_eof": _eof_counts(self, DT, False, last),
    }


def _eof_batch(self, b, s_cols, last):
    DT = _eof_frame(b)
    keys = [self.id_col, "trial", "followup"]
    measured = _eof_measure(b, DT)
    if s_cols:
        measured = measured.join(DT.select(keys + s_cols), on=keys, how="left")

    k, w = self.end_of_fup_time, self.end_of_fup_window
    by = _eof_group_cols(self, DT)
    valid = pl.col(self.outcome_col).is_not_null()
    in_window = valid & (pl.col("followup") >= k - w) & (pl.col("followup") <= k + w)
    totals = (
        DT.group_by([self.id_col, "trial"] + by)
        .agg(
            [
                (pl.col("period") - pl.col("followup")).min().alias("_start"),
                in_window.any().alias("in_window"),
                valid.any().alias("measured"),
            ]
        )
        .filter(pl.col("_start") + k - w <= last)
        .group_by(by)
        .agg(
            [
                pl.len().alias("n_eligible"),
                (pl.col("measured") & ~pl.col("in_window")).sum().alias("n_censored"),
                (~pl.col("measured")).sum().alias("n_nomeasure"),
            ]
        )
    )
    return measured, totals


class Eof:
    """
    Exact measurements of an end-of-follow-up analysis

    :ivar measured: One row per person and trial with a measurement
    :ivar totals: Trial-period counts per arm
    """

    def __init__(self, analysis, parts, s_cols):
        self.analysis = analysis
        self.s_cols = s_cols
        self.measured = pl.concat([m for m, _ in parts])
        self.by = [
            col
            for col in (
                f"{analysis.treatment_col}{analysis.indicator_baseline}",
                analysis.subgroup_colname,
            )
            if col is not None and col in self.measured.columns
        ]
        self.totals = (
            pl.concat([t for _, t in parts])
            .group_by(self.by)
            .agg(pl.all().sum())
            .sort(self.by)
        )

    def summary(self):
        # the package's eof_summary, from every measurement
        return _eof_summary_of(self.measured, self.by)

    def estimate(self, delta=None, counts=None):
        """
        Weighted average within each arm

        :param delta: Shift of the weight model coefficients
        :param counts: Times each ID is counted
        :rtype: pl.DataFrame
        """
        a = self.analysis
        m = self.measured
        w = m["weight"].to_numpy().astype(float)
        if delta is not None and self.s_cols:
            w = w * np.exp(m.select(self.s_cols).to_numpy() @ delta)
        lower, upper = _weight_bounds(a)
        w = np.clip(w, lower, upper)
        c = np.ones(len(w))
        if counts is not None:
            c = np.array([counts.get(i, 0) for i in m[a.id_col].to_list()], float)
        est = (
            m.select(self.by + [a.id_col, "eof_value"])
            .with_columns(pl.Series("_w", w * c), pl.Series("_c", c))
            .filter(pl.col("_c") > 0)
            .group_by(self.by)
            .agg(
                [
                    (
                        (pl.col("_w") * pl.col("eof_value")).sum() / pl.col("_w").sum()
                    ).alias("estimate"),
                    pl.col("_c").sum().cast(pl.Int64).alias("n"),
                    pl.col(a.id_col).n_unique().alias("n_subjects"),
                ]
            )
        )
        return est.join(self.totals, on=self.by, how="inner").sort(self.by)

    def tables(self, estimates, CI, CI_method, sample):
        a = self.analysis
        stub = SimpleNamespace(
            treatment_col=a.treatment_col,
            indicator_baseline=a.indicator_baseline,
            subgroup_colname=a.subgroup_colname,
            end_of_fup_type=a.end_of_fup_type,
            end_of_fup_time=a.end_of_fup_time,
            bootstrap_CI=CI,
            bootstrap_CI_method=CI_method,
            bootstrap_sample=sample,
            outcome_model=[{"eof": e} for e in estimates],
        )
        return _create_endoffup(stub)
