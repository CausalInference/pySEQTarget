import os

import polars as pl
import pytest

from pySEQTarget import SEQopts, SEQuential
from pySEQTarget.data import load_data


def _final_followup(s):
    return s.km_data.filter(pl.col("estimate") == "risk")["followup"].max()


def test_risk_times_reports_requested_followups():
    data = load_data("SEQdata")

    s = SEQuential(
        data,
        id_col="ID",
        time_col="time",
        eligible_col="eligible",
        treatment_col="tx_init",
        outcome_col="outcome",
        time_varying_cols=["N", "L", "P"],
        fixed_cols=["sex"],
        method="ITT",
        parameters=SEQopts(
            km_curves=True, risk_times=[2, 5], bootstrap_nboot=3, seed=42
        ),
    )
    s.expand()
    s.bootstrap()
    s.fit()
    s.survival()

    final = _final_followup(s)
    rd = s.risk_estimates["risk_difference"]
    rr = s.risk_estimates["risk_ratio"]

    assert "Followup" in rd.columns
    assert set(rd["Followup"].to_list()) == {2, 5, final}
    assert set(rr["Followup"].to_list()) == {2, 5, final}
    for col in ["RD 95% LCI", "RD 95% UCI"]:
        assert rd[col].null_count() == 0


def test_risk_times_default_reports_only_final():
    data = load_data("SEQdata")

    s = SEQuential(
        data,
        id_col="ID",
        time_col="time",
        eligible_col="eligible",
        treatment_col="tx_init",
        outcome_col="outcome",
        time_varying_cols=["N", "L", "P"],
        fixed_cols=["sex"],
        method="ITT",
        parameters=SEQopts(km_curves=True),
    )
    s.expand()
    s.fit()
    s.survival()

    assert set(s.risk_estimates["risk_difference"]["Followup"].to_list()) == {
        _final_followup(s)
    }


def test_risk_times_snaps_to_grid():
    data = load_data("SEQdata")

    s = SEQuential(
        data,
        id_col="ID",
        time_col="time",
        eligible_col="eligible",
        treatment_col="tx_init",
        outcome_col="outcome",
        time_varying_cols=["N", "L", "P"],
        fixed_cols=["sex"],
        method="ITT",
        parameters=SEQopts(km_curves=True, risk_times=[2.5]),
    )
    s.expand()
    s.fit()
    s.survival()

    # 2.5 snaps down to 2; final followup is always included
    assert set(s.risk_estimates["risk_difference"]["Followup"].to_list()) == {
        2,
        _final_followup(s),
    }


def test_risk_times_exceeding_max_raises():
    data = load_data("SEQdata")

    s = SEQuential(
        data,
        id_col="ID",
        time_col="time",
        eligible_col="eligible",
        treatment_col="tx_init",
        outcome_col="outcome",
        time_varying_cols=["N", "L", "P"],
        fixed_cols=["sex"],
        method="ITT",
        parameters=SEQopts(km_curves=True, risk_times=[1e6]),
    )
    s.expand()
    s.fit()
    with pytest.raises(ValueError, match="maximum followup"):
        s.survival()


def test_risk_times_negative_rejected():
    with pytest.raises(ValueError, match="non-negative"):
        SEQopts(km_curves=True, risk_times=[-1])


def test_regular_survival():
    data = load_data("SEQdata")

    s = SEQuential(
        data,
        id_col="ID",
        time_col="time",
        eligible_col="eligible",
        treatment_col="tx_init",
        outcome_col="outcome",
        time_varying_cols=["N", "L", "P"],
        fixed_cols=["sex"],
        method="ITT",
        parameters=SEQopts(km_curves=True),
    )
    s.expand()
    s.fit()
    s.survival()
    return


def test_bootstrapped_survival():
    data = load_data("SEQdata")

    s = SEQuential(
        data,
        id_col="ID",
        time_col="time",
        eligible_col="eligible",
        treatment_col="tx_init",
        outcome_col="outcome",
        time_varying_cols=["N", "L", "P"],
        fixed_cols=["sex"],
        method="ITT",
        parameters=SEQopts(km_curves=True, bootstrap_nboot=2, seed=42),
    )
    s.expand()
    s.bootstrap()
    s.fit()
    s.survival()
    return


def test_subgroup_survival():
    data = load_data("SEQdata")

    s = SEQuential(
        data,
        id_col="ID",
        time_col="time",
        eligible_col="eligible",
        treatment_col="tx_init",
        outcome_col="outcome",
        time_varying_cols=["N", "L", "P"],
        fixed_cols=["sex"],
        method="ITT",
        parameters=SEQopts(km_curves=True, subgroup_colname="sex"),
    )
    s.expand()
    s.fit()
    s.survival()
    return


def test_subgroup_bootstrapped_survival():
    data = load_data("SEQdata")

    s = SEQuential(
        data,
        id_col="ID",
        time_col="time",
        eligible_col="eligible",
        treatment_col="tx_init",
        outcome_col="outcome",
        time_varying_cols=["N", "L", "P"],
        fixed_cols=["sex"],
        method="ITT",
        parameters=SEQopts(
            km_curves=True, subgroup_colname="sex", bootstrap_nboot=2, seed=42
        ),
    )
    s.expand()
    s.bootstrap()
    s.fit()
    s.survival()
    return


@pytest.mark.skipif(
    os.getenv("CI") == "true", reason="Compevent dying in CI environment"
)
def test_compevent():
    data = load_data("SEQdata_LTFU")

    s = SEQuential(
        data,
        id_col="ID",
        time_col="time",
        eligible_col="eligible",
        treatment_col="tx_init",
        outcome_col="outcome",
        time_varying_cols=["N", "L", "P"],
        fixed_cols=["sex"],
        method="ITT",
        parameters=SEQopts(
            km_curves=True, compevent_colname="LTFU", plot_type="incidence"
        ),
    )
    s.expand()
    s.fit()
    s.survival()
    return


@pytest.mark.skipif(
    os.getenv("CI") == "true", reason="Compevent dying in CI environment"
)
def test_bootstrapped_compevent():
    data = load_data("SEQdata_LTFU")

    s = SEQuential(
        data,
        id_col="ID",
        time_col="time",
        eligible_col="eligible",
        treatment_col="tx_init",
        outcome_col="outcome",
        time_varying_cols=["N", "L", "P"],
        fixed_cols=["sex"],
        method="ITT",
        parameters=SEQopts(
            km_curves=True,
            compevent_colname="LTFU",
            plot_type="incidence",
            bootstrap_nboot=2,
            seed=42,
        ),
    )
    s.expand()
    s.bootstrap()
    s.fit()
    s.survival()
    return


@pytest.mark.skipif(
    os.getenv("CI") == "true", reason="Compevent dying in CI environment"
)
def test_subgroup_compevent():
    data = load_data("SEQdata_LTFU")

    s = SEQuential(
        data,
        id_col="ID",
        time_col="time",
        eligible_col="eligible",
        treatment_col="tx_init",
        outcome_col="outcome",
        time_varying_cols=["N", "L", "P"],
        fixed_cols=["sex"],
        method="ITT",
        parameters=SEQopts(
            km_curves=True,
            compevent_colname="LTFU",
            plot_type="incidence",
            subgroup_colname="sex",
        ),
    )
    s.expand()
    s.bootstrap()
    s.fit()
    s.survival()
    return


def _risk_estimates(ci, method, paired_subgroup=None):
    opts = dict(
        km_curves=True,
        bootstrap_nboot=4,
        seed=42,
        bootstrap_CI=ci,
        bootstrap_CI_method=method,
    )
    if paired_subgroup:
        opts["subgroup_colname"] = paired_subgroup
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
    s.bootstrap()
    s.fit()
    s.survival()
    return s.risk_estimates


@pytest.mark.parametrize("method", ["se", "percentile"])
def test_risk_ci_columns_labelled_with_requested_level(method):
    # The CI columns must be labelled with the requested level, not a hardcoded
    # 95% (the interval itself was already computed at the right level).
    est = _risk_estimates(0.9, method)
    rd, rr = est["risk_difference"], est["risk_ratio"]
    assert "RD 90% LCI" in rd.columns and "RD 90% UCI" in rd.columns
    assert "RR 90% LCI" in rr.columns and "RR 90% UCI" in rr.columns
    assert "RD 95% LCI" not in rd.columns
    assert "RR 95% LCI" not in rr.columns


@pytest.mark.parametrize("method", ["se", "percentile"])
def test_risk_se_columns_present_for_both_methods(method):
    # Bootstrap SEs are reported regardless of CI method: RD SE (natural scale)
    # and log(RR) SE (log scale) for inverse-variance meta-analysis pooling.
    est = _risk_estimates(0.95, method)
    rd, rr = est["risk_difference"], est["risk_ratio"]
    assert "RD SE" in rd.columns
    assert "log(RR) SE" in rr.columns
    assert rd["RD SE"].null_count() == 0
    assert (rd["RD SE"] >= 0).all()


def test_risk_se_columns_present_subgroup_delta_path():
    # Subgroups use the independent delta-method fallback (_compute_rd_rr),
    # which must also emit the SE columns and the requested CI label.
    est = _risk_estimates(0.9, "se", paired_subgroup="sex")
    rd, rr = est["risk_difference"], est["risk_ratio"]
    assert "RD SE" in rd.columns
    assert "log(RR) SE" in rr.columns
    assert "RD 90% LCI" in rd.columns
    assert "RR 90% LCI" in rr.columns
