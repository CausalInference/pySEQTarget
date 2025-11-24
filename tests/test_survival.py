from pySEQTarget import SEQopts, SEQuential
from pySEQTarget.data import load_data


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
        parameters=SEQopts(km_curves=True, bootstrap_nboot=2),
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
        parameters=SEQopts(km_curves=True, subgroup_colname="sex", bootstrap_nboot=2),
    )
    s.expand()
    s.bootstrap()
    s.fit()
    s.survival()
    return


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
        ),
    )
    s.expand()
    s.bootstrap()
    s.fit()
    s.survival()
    return


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
