import numpy as np
import pytest

from pySEQTarget import SEQopts, SEQuential
from pySEQTarget.data import load_data


def test_followup_class():
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
        parameters=SEQopts(followup_class=True, followup_include=False, followup_max=5),
    )
    s.expand()
    s.fit()
    matrix = s.outcome_model[0]["outcome"].summary2().tables[1]["Coef."].to_list()
    expected = [
        -6.6000834193414635,
        0.36024705241286203,
        0.04326409573404126,
        0.07627958175273072,
        0.11375627612408938,
        0.14496108664292745,
        0.1798424095611678,
        0.015738693166264354,
        0.0009258560187318309,
        0.09066206802273916,
        0.011267393559242982,
        0.022194521411244304,
        0.19115237121222872,
    ]
    assert [round(x, 3) for x in matrix] == [round(x, 3) for x in expected]


def test_followup_spline():
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
        parameters=SEQopts(followup_spline=True, followup_include=False),
    )
    s.expand()
    s.fit()
    matrix = s.outcome_model[0]["outcome"].summary2().tables[1]["Coef."].to_list()
    expected = [
        -6.275803730636,
        0.190835581131,
        0.044181701547,
        0.000583701185,
        0.127096501708,
        0.003280255156,
        -0.013270653552,
        0.199471860366,
        0.281342694425,
        0.453405473706,
        0.758640425731,
        0.966164355851,
    ]
    assert [round(x, 3) for x in matrix] == [round(x, 3) for x in expected]
    X = s.outcome_model[0]["outcome"].model.exog
    assert np.linalg.matrix_rank(X) == X.shape[1]


def test_no_followup():
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
        parameters=SEQopts(followup_include=False),
    )
    s.expand()
    s.fit()
    matrix = s.outcome_model[0]["outcome"].summary2().tables[1]["Coef."].to_list()
    expected = [
        -6.062350570326165,
        0.17748844870984498,
        0.03344595751001804,
        0.0005457002039545119,
        0.11209431124681817,
        0.0032236473201563585,
        -0.014463448024337773,
        0.20398559747503964,
    ]
    assert [round(x, 3) for x in matrix] == [round(x, 3) for x in expected]


@pytest.mark.parametrize("method", ["ITT", "dose-response"])
def test_followup_spline_with_km_curves_interactions(method):
    # km_curves adds treatment- or dose-by-followup interactions, which must
    # interact with the spline rather than lose its variable
    s = SEQuential(
        load_data("SEQdata"),
        id_col="ID",
        time_col="time",
        eligible_col="eligible",
        treatment_col="tx_init",
        outcome_col="outcome",
        time_varying_cols=["N", "L", "P"],
        fixed_cols=["sex"],
        method=method,
        parameters=SEQopts(
            followup_spline=True, followup_include=False, km_curves=True
        ),
    )
    s.expand()
    s.fit()
    s.survival()
    names = s.outcome_model[0]["outcome"].params.index
    assert any("cr(followup" in n and ":" in n for n in names)
    assert s.km_data.height > 0
