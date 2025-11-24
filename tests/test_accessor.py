import pytest

from pySEQTarget import SEQopts, SEQuential
from pySEQTarget.data import load_data


def test_ITT_collector():
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
        parameters=SEQopts(),
    )
    s.expand()
    s.fit()
    collector = s.collect()
    collector.retrieve_data("unique_outcomes")
    with pytest.raises(ValueError):
        collector.retrieve_data("km_data")
