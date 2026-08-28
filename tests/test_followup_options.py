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
        0.09066206802273916,
        0.015738693166264354,
        0.0009258560187318309,
        0.011267393559242982,
        0.022194521411244304,
        0.19115237121222872,
    ]
    assert [round(x, 3) for x in matrix] == [round(x, 3) for x in expected]


# patsy's natural-cubic cr() basis spans the constant function, so cr() plus
# the model Intercept is always collinear by exactly one dimension. Fitted
# values (all the spline path is used for) are unique regardless; only the
# individual spline coefficients are non-identified, resolved deterministically
# by statsmodels' pinv. statsmodels >= 0.15 reports the redundancy as a
# SingularMatrixWarning.
@pytest.mark.filterwarnings(
    "ignore::statsmodels.tools.sm_exceptions.SingularMatrixWarning"
)
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
        -4.804282252748607,
        0.19115933860001255,
        0.12717121164606823,
        0.044310717515918724,
        0.0005814999431447507,
        0.0032948355025455216,
        -0.013371824500839971,
        0.19972467861548412,
        -2.027245615586753,
        -1.395729861856384,
        -0.9397731941281695,
        -0.4415335811772879,
    ]
    assert [round(x, 3) for x in matrix] == [round(x, 3) for x in expected]


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
        0.11209431124681817,
        0.03344595751001804,
        0.0005457002039545119,
        0.0032236473201563585,
        -0.014463448024337773,
        0.20398559747503964,
    ]
    assert [round(x, 3) for x in matrix] == [round(x, 3) for x in expected]
