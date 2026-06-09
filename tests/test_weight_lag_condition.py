from pySEQTarget import SEQopts, SEQuential
from pySEQTarget.data import load_data


def _nobs_per_arm(weight_lag_condition):
    s = SEQuential(
        load_data("SEQdata"),
        id_col="ID",
        time_col="time",
        eligible_col="eligible",
        treatment_col="tx_init",
        outcome_col="outcome",
        time_varying_cols=["N", "L", "P"],
        fixed_cols=["sex"],
        method="censoring",
        parameters=SEQopts(
            weighted=True, weight_lag_condition=weight_lag_condition, seed=1
        ),
    )
    s.expand()
    s.fit()
    return [int(m.nobs) for m in s.denominator_model]


def test_weight_lag_condition_conditions_each_arm_on_its_treatment_lag_stratum():
    # weight_lag_condition=True (default): each arm's weight model is fit only on
    # rows where tx_lag matches that arm (per-arm row counts differ but partition
    # the full data). =False: both arms fit on the full data (equal counts).
    on = _nobs_per_arm(weight_lag_condition=True)
    off = _nobs_per_arm(weight_lag_condition=False)

    # FALSE: both arms see the full data -> equal observation counts
    assert off[0] == off[1]
    # TRUE: arms fit on disjoint treatment-lag strata that partition that full data
    assert on[0] != on[1]
    assert on[0] + on[1] == off[0]
