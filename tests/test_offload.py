import warnings

from pySEQTarget import SEQopts, SEQuential
from pySEQTarget.data import load_data


def test_compevent_offload():
    data = load_data("SEQdata_LTFU")
    options = SEQopts(
        bootstrap_nboot=2,
        cense_colname="LTFU",
        excused=True,
        excused_colnames=["excusedZero", "excusedOne"],
        km_curves=True,
        selection_random=True,
        selection_sample=0.30,
        weighted=True,
        weight_lag_condition=False,
        weight_p99=True,
        weight_preexpansion=True,
        offload=True,
    )

    model = SEQuential(
        data,
        id_col="ID",
        time_col="time",
        eligible_col="eligible",
        treatment_col="tx_init",
        outcome_col="outcome",
        time_varying_cols=["N", "L", "P"],
        fixed_cols=["sex"],
        method="censoring",
        parameters=options,
    )
    model.expand()
    model.bootstrap()
    # Warnings from statsmodels about overflow in some bootstraps
    warnings.filterwarnings("ignore")
    model.fit()
    model.survival()
