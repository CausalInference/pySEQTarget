"""The _bas/_sq indicators are matched only as name suffixes."""

import polars as pl

from pySEQTarget import SEQopts, SEQuential
from pySEQTarget.data import load_data


def test_indicator_text_mid_name_is_a_regular_column():
    data = load_data("SEQdata").with_columns(
        (pl.col("N") + 1).alias("bmi_baseline"), (pl.col("L") + 1).alias("bmi_sqrt")
    )
    s = SEQuential(
        data,
        id_col="ID",
        time_col="time",
        eligible_col="eligible",
        treatment_col="tx_init",
        outcome_col="outcome",
        time_varying_cols=["N", "L", "P"],
        fixed_cols=["sex", "bmi_baseline", "bmi_sqrt"],
        method="ITT",
    )
    s.expand()
    assert {"bmi_baseline", "bmi_sqrt", "N_bas"} <= set(s.DT.columns)
    s.fit()
    names = s.outcome_model[0]["outcome"].params.index
    assert {"bmi_baseline", "bmi_sqrt"} <= set(names)


def test_suffixed_columns_are_still_derived():
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
        parameters=SEQopts(covariates="tx_init_bas+followup+N_bas+L_sq"),
    )
    s.expand()
    assert {"N_bas", "L_sq"} <= set(s.DT.columns)
    assert (s.DT["L_sq"] == s.DT["L"] ** 2).all()
