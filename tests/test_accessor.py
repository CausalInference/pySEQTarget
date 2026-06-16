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
    # ITT produces no switch diagnostics: a clean ValueError, not the
    # Python-2 dict.has_key AttributeError this used to raise.
    with pytest.raises(ValueError, match="not created"):
        collector.retrieve_data("unique_switches")


def test_collect_before_fit():
    # collect() without fit() must return an SEQoutput with None models rather
    # than raising UnboundLocalError. Diagnostics from expand() still come
    # through.
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
        parameters=SEQopts(),
    )
    s.expand()
    collector = s.collect()
    assert collector.outcome_models is None
    assert collector.compevent_models is None
    assert collector.retrieve_data("unique_outcomes").height > 0


def test_censoring_collector_switch_diagnostics():
    # Under method="censoring" the switch diagnostics exist and must be
    # retrievable (regression for dict.has_key).
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
        parameters=SEQopts(),
    )
    s.expand()
    s.fit()
    collector = s.collect()
    assert collector.retrieve_data("unique_switches").height > 0
    assert collector.retrieve_data("nonunique_switches").height > 0
