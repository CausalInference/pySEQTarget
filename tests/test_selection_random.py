import numpy as np
import polars as pl
import pytest

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


def _final_risks(**opts):
    s = _build(km_curves=True, **opts)
    s.fit()
    s.survival()
    risk = s.km_data.filter(pl.col("estimate") == "risk")
    risk = risk.filter(pl.col("followup") == risk["followup"].max())
    return dict(zip(risk["tx_init"].to_list(), risk["pred"].to_list()))


def test_selection_random_standardizes_risks_over_all_trials():
    # Unweighted standardization over the kept trials under-represented the
    # subsampled arm's covariate distribution and shifted both risks by ~0.009
    # (selection_sample = 0.3, 5 seeds); weighting the kept control trials by
    # 1 / selection_sample removes the shift.
    full = _final_risks()
    selected = [
        _final_risks(selection_random=True, selection_sample=0.3, seed=s)
        for s in range(1, 6)
    ]
    for tx in (0, 1):
        mean = sum(r[tx] for r in selected) / len(selected)
        assert abs(mean - full[tx]) < 0.005


def test_selection_weight_none_without_selection():
    from pySEQTarget.expansion import _selection_weight

    assert _selection_weight(_build()) is None
    w = _selection_weight(_build(selection_random=True, selection_sample=0.25))
    out = pl.DataFrame({"tx_init_bas": [0, 1]}).select(w.alias("w"))
    assert out["w"].to_list() == [4.0, 1.0]


def test_cox_log_hr_uses_weights_on_both_backends():
    # scikit-survival's Cox model takes no weights, so a weighted fit uses
    # lifelines whichever backend is selected
    from types import SimpleNamespace

    import pandas as pd

    from pySEQTarget.analysis._hazard import _cox_log_hr

    df = pd.DataFrame(
        {
            "followup": [1, 2, 3, 4, 5, 6, 2, 3, 5, 6, 7, 8],
            "event": [1, 1, 0, 1, 1, 0, 1, 0, 1, 1, 0, 1],
            "tx": [0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1],
            "w": [4.0, 1.0, 4.0, 1.0, 4.0, 1.0] + [1.0] * 6,
        }
    )
    fit = {
        pkg: lambda weights_col, pkg=pkg: _cox_log_hr(
            SimpleNamespace(cox_package=pkg), df, "followup", "event", "tx", weights_col
        )
        for pkg in ("lifelines", "scikit-survival")
    }
    weighted = fit["lifelines"]("w")
    assert weighted != pytest.approx(fit["lifelines"](None), abs=1e-3)
    assert fit["scikit-survival"]("w") == pytest.approx(weighted)


def test_selection_random_hazard_runs_on_both_cox_backends():
    for pkg in ("lifelines", "scikit-survival"):
        s = _build(
            hazard_estimate=True,
            cox_package=pkg,
            selection_random=True,
            selection_sample=0.5,
        )
        s.fit()
        s.hazard()
        assert np.isfinite(s.hazard_ratio["Hazard ratio"][0])
