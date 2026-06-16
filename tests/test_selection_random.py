import polars as pl

from pySEQTarget import SEQopts, SEQuential
from pySEQTarget.data import load_data


def _build(**opts):
    opts.setdefault("seed", 1)
    s = SEQuential(
        load_data("SEQdata"),
        id_col="ID",
        time_col="time",
        eligible_col="eligible",
        treatment_col="tx_init",
        outcome_col="outcome",
        time_varying_cols=["N", "L", "P"],
        fixed_cols=["sex"],
        method="ITT",
        parameters=SEQopts(**opts),
    )
    s.expand()
    return s


def _arm_trial_starts(dt):
    """Trial-starts (followup == 0) per baseline-treatment arm."""
    counts = (
        dt.filter(pl.col("followup") == 0)
        .group_by("tx_init_bas")
        .len()
        .sort("tx_init_bas")
    )
    return dict(zip(counts["tx_init_bas"].to_list(), counts["len"].to_list()))


def test_selection_random_keeps_all_treated_and_subsamples_controls():
    # With selection_random=True, treated trial-starts (tx_init_bas == 1) are
    # all retained, while control trial-starts (tx_init_bas == 0) are
    # subsampled to int(selection_sample * N_controls).
    prob = 0.5

    base = _build()
    sel = _build(selection_random=True, selection_sample=prob)

    base_c = _arm_trial_starts(base.DT)
    sel_c = _arm_trial_starts(sel.DT)

    assert sel_c[1] == base_c[1]
    assert sel_c[0] < base_c[0]
    assert sel_c[0] == int(prob * base_c[0])


def test_selection_random_is_reproducible_with_fixed_seed():
    a = _build(selection_random=True, selection_sample=0.5, seed=7)
    b = _build(selection_random=True, selection_sample=0.5, seed=7)
    assert a.DT.equals(b.DT)


def test_selection_random_nonzero_control_level():
    # Regression: the filter used `is_in(sample) | col != level`, which parses
    # as `(is_in | col) != level` and silently dropped every sampled control
    # trial whenever treatment_level[0] != 0 (e.g. [1, 2]) — the whole control
    # arm vanished. Sampled controls must be retained.
    prob = 0.5

    def build(**opts):
        opts.setdefault("seed", 1)
        s = SEQuential(
            load_data("SEQdata_multitreatment"),
            id_col="ID",
            time_col="time",
            eligible_col="eligible",
            treatment_col="tx_init",
            outcome_col="outcome",
            time_varying_cols=["N", "L", "P"],
            fixed_cols=["sex"],
            method="ITT",
            parameters=SEQopts(treatment_level=[1, 2], **opts),
        )
        s.expand()
        return s

    base = _arm_trial_starts(build().DT)
    sel = _arm_trial_starts(build(selection_random=True, selection_sample=prob).DT)

    # Non-control arm (level 2) fully retained; control arm (level 1)
    # subsampled to the requested fraction — not dropped entirely.
    assert sel[2] == base[2]
    assert sel[1] == int(prob * base[1])
