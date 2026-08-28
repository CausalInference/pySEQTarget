"""End-of-follow-up outcomes: a single measurement taken at end_of_fup_time,
averaged within each baseline arm using the weight at that time, rather than a
time-to-event fitted with a survival outcome model. Ported from SEQTaRget's
test_end_of_fup.R (PR #168)."""

import numpy as np
import polars as pl
import pytest
from pytest import approx

from pySEQTarget import SEQopts, SEQuential
from pySEQTarget.data import load_data
from pySEQTarget.analysis._endoffup import _eof_measure


def _eof_run(data=None, outcome="outcome", method="ITT", run=True, **opts):
    s = SEQuential(
        data if data is not None else load_data("SEQdata"),
        id_col="ID",
        time_col="time",
        eligible_col="eligible",
        treatment_col="tx_init",
        outcome_col=outcome,
        time_varying_cols=["N", "L", "P"],
        fixed_cols=["sex"],
        method=method,
        parameters=SEQopts(seed=42, **opts),
    )
    if run:
        s.expand()
        if s.bootstrap_nboot > 0:
            s.bootstrap()
        s.fit()
        s.end_of_followup()
    return s


def _continuous_data(seed=42):
    rng = np.random.default_rng(seed)
    d = load_data("SEQdata")
    return d.with_columns(
        (
            10
            + 2 * pl.col("tx_init")
            + pl.col("N")
            + pl.Series(rng.standard_normal(d.height))
        ).alias("cont")
    )


def test_unweighted_estimate_is_mean_of_selected_measurements():
    k, w = 12, 3
    s = _eof_run(end_of_fup=True, end_of_fup_time=k, end_of_fup_window=w)

    # Independent re-implementation of the selection rule: the non-missing value
    # nearest to k within [k - w, k + w], ties broken toward the later one.
    manual = (
        s.DT.filter(
            pl.col("outcome").is_not_null()
            & (pl.col("followup") >= k - w)
            & (pl.col("followup") <= k + w)
        )
        .with_columns((pl.col("followup").cast(pl.Int64) - k).abs().alias("d"))
        .sort(["ID", "trial", "d", "followup"], descending=[False, False, False, True])
        .group_by(["ID", "trial"], maintain_order=True)
        .first()
        .group_by("tx_init_bas")
        .agg([pl.col("outcome").mean().alias("manual"), pl.len().alias("n")])
        .sort("tx_init_bas")
    )

    got = s.eof_data.sort("A")
    assert got["Proportion"].to_list() == approx(manual["manual"].to_list())
    assert got["Trial-periods (Analysed)"].to_list() == manual["n"].to_list()


def test_window_only_adds_trial_periods_without_measurement_at_k():
    exact = _eof_run(end_of_fup=True, end_of_fup_time=12)
    windowed = _eof_run(end_of_fup=True, end_of_fup_time=12, end_of_fup_window=3)

    n_exact = exact.eof_data.sort("A")["Trial-periods (Analysed)"].to_list()
    n_win = windowed.eof_data.sort("A")["Trial-periods (Analysed)"].to_list()
    # Widening the window can only ever add contributors, never remove them
    assert all(w >= e for w, e in zip(n_win, n_exact))
    assert any(w > e for w, e in zip(n_win, n_exact))

    # A window of 0 is the same as no window at all
    zero = _eof_run(end_of_fup=True, end_of_fup_time=12, end_of_fup_window=0)
    assert zero.eof_data.equals(exact.eof_data)


def test_window_takes_nearest_measurement_not_earliest():
    # Hand-built trial-periods where 'nearest' and 'earliest' disagree, driven
    # through the selection helper directly so the rule is tested in isolation.
    s = _eof_run(end_of_fup=True, end_of_fup_time=3, end_of_fup_window=2, run=False)

    DT = pl.DataFrame(
        {
            "ID": [1, 1, 2, 2, 3, 3],
            "trial": [0] * 6,
            "followup": [1, 4, 2, 3, 2, 4],
            "tx_init_bas": [0, 0, 1, 1, 0, 0],
            "outcome": [10.0, 20.0, 30.0, 40.0, 50.0, 60.0],
        }
        # followup is UInt32 in the real expanded frame; the distance-to-k
        # computation must not wrap for followups below k (unsigned subtraction)
    ).with_columns(pl.col("followup").cast(pl.UInt32))
    got = _eof_measure(s, DT).sort("ID")

    # ID 1: |1-3|=2 vs |4-3|=1, so the later measurement is nearer — the
    #       earliest rule would have taken followup 1
    # ID 2: measured at exactly k, which always wins
    # ID 3: |2-3|=|4-3|=1, an equidistant tie broken toward the later, so that
    #       at least k of follow-up has elapsed
    assert got["followup"].to_list() == [4, 3, 4]
    assert got["eof_value"].to_list() == [20.0, 40.0, 60.0]


def test_continuous_outcome_reported_as_mean_not_clamped():
    s = _eof_run(
        data=_continuous_data(),
        outcome="cont",
        end_of_fup=True,
        end_of_fup_time=12,
        end_of_fup_type="continuous",
        end_of_fup_window=3,
        bootstrap_nboot=5,
    )
    assert "Mean" in s.eof_data.columns
    assert "Proportion" not in s.eof_data.columns
    assert (s.eof_data["Mean"] > 1).all()
    # A continuous mean is unbounded, so its interval must not be clamped to [0, 1]
    lci_col = next(c for c in s.eof_data.columns if "LCI" in c)
    assert (s.eof_data[lci_col] > 1).all()


def test_binary_intervals_clamped_to_unit_range():
    s = _eof_run(
        end_of_fup=True, end_of_fup_time=12, end_of_fup_window=3, bootstrap_nboot=5
    )
    lci = next(c for c in s.eof_data.columns if "LCI" in c)
    uci = next(c for c in s.eof_data.columns if "UCI" in c)
    assert (s.eof_data[lci] >= 0).all()
    assert (s.eof_data[uci] <= 1).all()


def test_weighted_censoring_end_of_fup_runs():
    s = _eof_run(
        method="censoring",
        end_of_fup=True,
        end_of_fup_time=12,
        end_of_fup_window=3,
        weighted=True,
        weight_preexpansion=True,
    )
    assert s.eof_data.height == 2
    assert s.eof_data["Proportion"].is_between(0, 1).all()
    # Weighted and unweighted estimates should differ (weights are not all 1)
    unw = _eof_run(
        method="censoring", end_of_fup=True, end_of_fup_time=12, end_of_fup_window=3
    )
    assert s.eof_data["Proportion"].to_list() != approx(
        unw.eof_data["Proportion"].to_list(), rel=1e-12
    )


def test_bootstrap_gives_per_arm_and_paired_ci_with_requested_level():
    s = _eof_run(
        end_of_fup=True,
        end_of_fup_time=12,
        end_of_fup_window=3,
        bootstrap_nboot=5,
        bootstrap_CI=0.9,
    )
    assert "90% LCI" in s.eof_data.columns and "90% UCI" in s.eof_data.columns
    assert "SE" in s.eof_data.columns
    comp = s.eof_comparison
    for col in ("Difference 90% LCI", "Difference 90% UCI", "Difference SE"):
        assert col in comp.columns
    # Paired contrasts are antisymmetric
    comp = comp.sort(["A_x", "A_y"])
    assert comp["Difference"][0] == approx(-comp["Difference"][1])


def test_binary_reports_ratio_continuous_does_not():
    b = _eof_run(
        end_of_fup=True, end_of_fup_time=12, end_of_fup_window=3, bootstrap_nboot=5
    )
    assert "Ratio" in b.eof_comparison.columns
    assert "log(Ratio) SE" in b.eof_comparison.columns

    c = _eof_run(
        data=_continuous_data(),
        outcome="cont",
        end_of_fup=True,
        end_of_fup_time=12,
        end_of_fup_type="continuous",
        end_of_fup_window=3,
        bootstrap_nboot=5,
    )
    assert "Difference" in c.eof_comparison.columns
    assert "Ratio" not in c.eof_comparison.columns


def test_subgroups_produce_one_estimate_set_each():
    s = _eof_run(
        end_of_fup=True,
        end_of_fup_time=12,
        end_of_fup_window=3,
        subgroup_colname="sex",
    )
    assert "sex" in s.eof_data.columns
    # One row per (subgroup, arm)
    assert s.eof_data.height == 4
    assert "sex" in s.eof_comparison.columns
    assert s.eof_comparison.height == 4  # both directions x two subgroups


def test_incompatible_options_raise():
    with pytest.raises(ValueError, match="km_curves or hazard"):
        _eof_run(end_of_fup=True, end_of_fup_time=12, km_curves=True, run=False)
    with pytest.raises(ValueError, match="km_curves or hazard"):
        _eof_run(end_of_fup=True, end_of_fup_time=12, hazard_estimate=True, run=False)
    with pytest.raises(ValueError, match="dose-response"):
        _eof_run(
            end_of_fup=True, end_of_fup_time=12, method="dose-response", run=False
        )
    with pytest.raises(ValueError, match="compevent"):
        _eof_run(
            end_of_fup=True,
            end_of_fup_time=12,
            compevent_colname="outcome",
            run=False,
        )


def test_seqopts_validates_end_of_fup_arguments():
    with pytest.raises(ValueError, match="end_of_fup_time"):
        SEQopts(end_of_fup=True)
    with pytest.raises(ValueError, match="end_of_fup_type"):
        SEQopts(end_of_fup=True, end_of_fup_time=12, end_of_fup_type="count")
    with pytest.raises(ValueError, match="end_of_fup_window"):
        SEQopts(end_of_fup=True, end_of_fup_time=12, end_of_fup_window=-1)
    with pytest.raises(ValueError, match="non-negative"):
        SEQopts(end_of_fup=True, end_of_fup_time=-3)


def test_requested_time_must_lie_within_expanded_followup():
    with pytest.raises(ValueError, match="exceeds the maximum follow-up"):
        _eof_run(end_of_fup=True, end_of_fup_time=10_000, run=False)
    with pytest.raises(ValueError, match="exceeds the maximum follow-up"):
        _eof_run(
            end_of_fup=True,
            end_of_fup_time=12,
            end_of_fup_window=3,
            followup_max=13,
            run=False,
        )


def test_expansion_not_truncated_at_first_event():
    # The survival path cuts each trial at its first outcome row. An
    # end-of-follow-up outcome is a status measured repeatedly and read at a
    # fixed time, so that truncation must not apply or the measurement at k
    # would be discarded for anyone whose status was ever 1 earlier. SEQdata
    # cannot show this — every subject's series already ends at their event —
    # so this uses a dataset where measurement genuinely continues afterwards.
    rng = np.random.default_rng(7)
    n_id, n_t = 40, 20
    d = pl.DataFrame(
        {
            "ID": np.repeat(np.arange(1, n_id + 1), n_t),
            "time": np.tile(np.arange(n_t), n_id),
            "eligible": np.ones(n_id * n_t, dtype=int),
            "tx_init": rng.binomial(1, 0.5, n_id * n_t),
            "outcome": rng.binomial(1, 0.3, n_id * n_t),  # recurring status
            "N": rng.standard_normal(n_id * n_t),
            "L": rng.standard_normal(n_id * n_t),
            "P": rng.standard_normal(n_id * n_t),
            "sex": np.repeat(rng.binomial(1, 0.5, n_id), n_t),
        }
    )

    eof = _eof_run(data=d, end_of_fup=True, end_of_fup_time=5, run=False)
    eof.expand()
    surv = _eof_run(data=d, run=False)
    surv.expand()

    assert eof.DT.height > surv.DT.height
    # Truncation leaves at most one outcome row per trial; without it a trial
    # keeps several
    events_per_trial = pl.col("outcome").fill_null(0).sum()
    surv_max = (
        surv.DT.group_by(["ID", "trial"]).agg(events_per_trial.alias("n"))["n"].max()
    )
    eof_max = (
        eof.DT.group_by(["ID", "trial"]).agg(events_per_trial.alias("n"))["n"].max()
    )
    assert surv_max == 1
    assert eof_max > 1

    # And the estimate is still readable at k for trials whose status was 1 earlier
    eof.fit()
    eof.end_of_followup()
    assert eof.eof_data["Proportion"].is_finite().all()


def test_estimates_table_reports_censored_share_and_partition():
    s = _eof_run(end_of_fup=True, end_of_fup_time=12, end_of_fup_window=3)
    d = s.eof_data
    assert "% Censored" in d.columns
    # Eligible = Analysed + Censored + No measurement, per arm
    total = (
        d["Trial-periods (Analysed)"]
        + d["Trial-periods (Censored)"]
        + d["Trial-periods (No measurement)"]
    )
    assert total.to_list() == d["Trial-periods (Eligible)"].to_list()
    assert d["% Censored"].to_list() == approx(
        (
            100 * d["Trial-periods (Censored)"] / d["Trial-periods (Eligible)"]
        ).to_list()
    )


def test_counts_table_accounts_for_every_trial_period():
    s = _eof_run(end_of_fup=True, end_of_fup_time=12, end_of_fup_window=3)
    counts = s.diagnostics["nonunique_eof"].sort("tx_init_bas")
    levels = [
        "At k",
        "In window",
        "Excluded (outside window)",
        "Excluded (no measurement)",
    ]
    # Trial-period categories partition Eligible
    total = sum(counts[lv] for lv in levels)
    assert total.to_list() == counts["Eligible"].to_list()
    # At k + In window equals the analysed trial-periods in the estimate table
    analysed = (counts["At k"] + counts["In window"]).to_list()
    assert analysed == s.eof_data.sort("A")["Trial-periods (Analysed)"].to_list()
    # The subject-level table exists too
    assert s.diagnostics["unique_eof"].height == 2


def test_zero_window_puts_every_contributor_at_k():
    s = _eof_run(end_of_fup=True, end_of_fup_time=12)
    counts = s.diagnostics["nonunique_eof"]
    assert (counts["In window"] == 0).all()
    assert (counts["At k"] > 0).all()


def test_counts_tables_absent_unless_end_of_fup():
    s = _eof_run(run=False)
    s.expand()
    s.fit()
    assert "unique_eof" not in s.diagnostics
    assert "nonunique_eof" not in s.diagnostics


def test_missing_outcomes_permitted_only_in_end_of_fup_mode():
    rng = np.random.default_rng(0)
    d = load_data("SEQdata")
    mask = pl.Series(rng.random(d.height) < 0.2)
    d = d.with_columns(
        pl.when(mask).then(None).otherwise(pl.col("outcome")).alias("outcome")
    )

    # Permitted (null = "not measured at this time") in eof mode
    s = _eof_run(data=d, end_of_fup=True, end_of_fup_time=12, end_of_fup_window=3)
    assert s.eof_data.height == 2

    # Rejected in ordinary survival mode
    with pytest.raises(ValueError, match="missing"):
        _eof_run(data=d, run=False)


def test_nonbinary_outcome_error_hints_at_end_of_fup_type():
    with pytest.raises(ValueError, match="continuous"):
        _eof_run(
            data=_continuous_data(),
            outcome="cont",
            end_of_fup=True,
            end_of_fup_time=12,
            run=False,
        )


def test_continuous_outcome_diagnostics():
    s = _eof_run(
        data=_continuous_data(),
        outcome="cont",
        end_of_fup=True,
        end_of_fup_time=12,
        end_of_fup_type="continuous",
        end_of_fup_window=3,
    )
    # Outcome event tables are suppressed for a continuous outcome only...
    assert "unique_outcomes" not in s.diagnostics
    # ...replaced by the N/Mean/SD summary of the analysed measurements
    summary = s.diagnostics["eof_summary"]
    assert set(summary.columns) >= {"A", "N", "Mean", "SD"}
    assert (summary["Mean"] > 1).all()
    # Follow-up tables survive
    assert "unique_followup" in s.diagnostics

    # Binary eof keeps the outcome tables
    b = _eof_run(end_of_fup=True, end_of_fup_time=12)
    assert "unique_outcomes" in b.diagnostics


def test_collect_and_retrieve_data_expose_eof_tables():
    s = _eof_run(
        end_of_fup=True, end_of_fup_time=12, end_of_fup_window=3, bootstrap_nboot=3
    )
    out = s.collect()
    assert out.outcome_models is None
    assert out.retrieve_data("eof_data").height == 2
    assert out.retrieve_data("eof_comparison").height == 2
    assert out.retrieve_data("nonunique_eof").height == 2
    with pytest.raises(ValueError, match="not created"):
        out.retrieve_data("eof_summary")  # binary outcome: no summary table


def test_survival_and_hazard_blocked_in_eof_mode():
    s = _eof_run(end_of_fup=True, end_of_fup_time=12)
    with pytest.raises(ValueError, match="end_of_followup"):
        s.survival()
    with pytest.raises(ValueError, match="end_of_followup"):
        s.hazard()


def test_end_of_followup_requires_eof_mode_and_fit():
    plain = _eof_run(run=False)
    plain.expand()
    with pytest.raises(ValueError, match="end_of_fup=False"):
        plain.end_of_followup()

    s = _eof_run(end_of_fup=True, end_of_fup_time=12, run=False)
    s.expand()
    with pytest.raises(ValueError, match="fit"):
        s.end_of_followup()
