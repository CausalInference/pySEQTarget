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
        -6.264817962084417,
        0.20125056343026881,
        0.12568743032952776,
        0.03823426390103046,
        0.0006607691746414019,
        0.003343365539743267,
        -0.01319460158923785,
        0.19601796921732118,
        -0.5186462478511427,
        0.37598656666756103,
        1.6553848469346044,
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
