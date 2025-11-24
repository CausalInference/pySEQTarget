from pySEQTarget import SEQopts, SEQuential
from pySEQTarget.data import load_data


def test_ITT_hazard():
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
        parameters=SEQopts(hazard_estimate=True),
    )
    s.expand()
    s.fit()
    s.hazard()


def test_bootstrap_hazard():
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
        parameters=SEQopts(hazard_estimate=True, bootstrap_nboot=2),
    )
    s.expand()
    s.bootstrap()
    s.fit()
    s.hazard()


def test_subgroup_hazard():
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
        parameters=SEQopts(hazard_estimate=True, subgroup_colname="sex"),
    )
    s.expand()
    s.bootstrap()
    s.fit()
    s.hazard()
