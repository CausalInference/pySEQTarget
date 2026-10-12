import pytest
from pytest import approx

from pySEQTarget import SEQopts, SEQuential
from pySEQTarget.data import load_data


def test_ITT_coefs():
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
    matrix = s.outcome_model[0]["outcome"].summary2().tables[1]["Coef."].to_list()
    expected = [
        -6.828506035553407,
        0.18935003090041902,
        0.033715156987629266,
        -0.00014691202235029346,
        0.044566165558944326,
        0.0005787770439053261,
        0.12717241010542563,
        0.0032906669395295026,
        -0.01339242049205771,
        0.20072409918428052,
    ]
    assert matrix == approx(expected, rel=1e-3)


def test_PreE_dose_response_coefs():
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
        method="dose-response",
        parameters=SEQopts(weighted=True, weight_preexpansion=True),
    )
    s.expand()
    s.fit()
    matrix = s.outcome_model[0]["outcome"].summary2().tables[1]["Coef."].to_list()
    expected = [
        -4.838361606118745,
        0.0555200193349395,
        -0.000588987903925482,
        -0.008977364567147181,
        0.00022025518488454974,
        0.01046828606393553,
        0.0007774475011663997,
        0.13972199132132426,
    ]
    assert matrix == approx(expected, rel=1e-3)


def test_PostE_dose_response_coefs():
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
        method="dose-response",
        parameters=SEQopts(weighted=True, weight_preexpansion=False),
    )

    s.expand()
    s.fit()
    matrix = s.outcome_model[0]["outcome"].summary2().tables[1]["Coef."].to_list()
    expected = [
        -6.265901713761531,
        0.048626017624679704,
        -0.0004688287307505834,
        -0.003975906839775267,
        0.00016676441745740924,
        0.03866279977096397,
        0.0005928449623613982,
        0.14065954021957594,
        0.0030001459817949844,
        -0.02106338184559446,
        0.14867250693140854,
    ]
    assert matrix == approx(expected, rel=1e-3)


def test_PreE_censoring_coefs():
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
        method="censoring",
        parameters=SEQopts(weighted=True, weight_preexpansion=True),
    )
    s.expand()
    s.fit()
    matrix = s.outcome_model[0]["outcome"].summary2().tables[1]["Coef."].to_list()
    expected = [
        -4.866811032893474,
        0.48052384773425627,
        0.029219605971331802,
        4.518185255729768e-05,
        -0.013811946311246277,
        0.0011309893093644648,
        0.04763133686352373,
    ]

    assert matrix == approx(expected, rel=1e-3)


def test_PostE_censoring_coefs():
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
        method="censoring",
        parameters=SEQopts(weighted=True, weight_preexpansion=False),
    )
    s.expand()
    s.fit()
    matrix = s.outcome_model[0]["outcome"].summary2().tables[1]["Coef."].to_list()
    expected = [
        -9.172266785106519,
        0.4707554720116625,
        0.029021087196430605,
        7.8937226861939e-05,
        0.06700192286925702,
        0.0005834323664644191,
        0.08162617232478116,
        0.004870212765388434,
        0.013503198983327514,
        0.4466573801510379,
    ]
    assert matrix == approx(expected, rel=1e-3)


def test_PreE_censoring_excused_coefs():
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
        method="censoring",
        parameters=SEQopts(
            weighted=True,
            weight_preexpansion=True,
            excused=True,
            excused_colnames=["excusedZero", "excusedOne"],
        ),
    )
    s.expand()
    s.fit()
    matrix = s.outcome_model[0]["outcome"].summary2().tables[1]["Coef."].to_list()
    expected = [
        -4.804650815594313,
        0.15964610209528257,
        0.029097252181086327,
        -4.654535663714614e-05,
        0.007817954062830172,
        0.0008537393051722129,
    ]
    assert matrix == approx(expected, rel=1e-3)


def test_PostE_censoring_excused_coefs():
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
        method="censoring",
        parameters=SEQopts(
            weighted=True,
            weight_preexpansion=False,
            excused=True,
            excused_colnames=["excusedZero", "excusedOne"],
        ),
    )
    s.expand()
    s.fit()
    matrix = s.outcome_model[0]["outcome"].summary2().tables[1]["Coef."].to_list()
    expected = [
        -8.311960526025656,
        0.1820521596172769,
        0.03422386974709779,
        -0.00013723383158569053,
        0.06646241563319619,
        0.0004999277868749847,
        0.15032828057125242,
        0.0012618850925656703,
        0.0028819214525432277,
        0.3590674018732999,
    ]
    assert matrix == approx(expected, rel=1e-3)


def test_PreE_LTFU_ITT():
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
            weighted=True, weight_preexpansion=True, cense_colname="LTFU"
        ),
    )
    s.expand()
    s.fit()
    matrix = s.outcome_model[0]["outcome"].summary2().tables[1]["Coef."].to_list()
    expected = [
        -21.636346991788276,
        0.06813705852786496,
        0.02874152772603635,
        -0.0005734047013500563,
        0.2854740212699898,
        -0.0013729662310668182,
        -0.1939555961858531,
        0.006501915963316852,
        -0.4467079969655381,
        1.3870473474960576,
    ]
    assert matrix == approx(expected, rel=1e-3)


def test_PostE_LTFU_ITT():
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
            weighted=True, weight_preexpansion=False, cense_colname="LTFU"
        ),
    )
    s.expand()
    s.fit()
    matrix = s.outcome_model[0]["outcome"].summary2().tables[1]["Coef."].to_list()
    expected = [
        -21.847198431385877,
        0.07786703138967718,
        0.030140057462437704,
        -0.0006287338029348562,
        0.287393206037481,
        -0.0013719595115633126,
        -0.15461370944416225,
        0.007295485861066434,
        -0.42797049565882755,
        1.4082102322835948,
    ]
    assert matrix == approx(expected, rel=1e-3)


def test_ITT_multinomial():
    data = load_data("SEQdata_multitreatment")

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
        parameters=SEQopts(treatment_level=[1, 2]),
    )
    s.expand()
    s.fit()
    matrix = s.outcome_model[0]["outcome"].summary2().tables[1]["Coef."].to_list()
    expected = [
        -47.505262164163625,
        1.76628017234151,
        0.14473536056627245,
        -0.003725499516376173,
        0.2893070991930884,
        -0.004266608123938117,
        22.79205044396338,
        0.05574429164512122,
        0.7847862691929901,
        1.4703411759229423,
    ]
    assert matrix == approx(expected, rel=1e-3)


@pytest.mark.filterwarnings("ignore:overflow encountered in exp:RuntimeWarning")
def test_weighted_multinomial():
    data = load_data("SEQdata_multitreatment")

    s = SEQuential(
        data,
        id_col="ID",
        time_col="time",
        eligible_col="eligible",
        treatment_col="tx_init",
        outcome_col="outcome",
        time_varying_cols=["N", "L", "P"],
        fixed_cols=["sex"],
        method="censoring",
        parameters=SEQopts(
            weighted=True, weight_preexpansion=True, treatment_level=[1, 2]
        ),
    )
    s.expand()
    s.fit()
    # The outcome model has no events in arm 2, so its fit diverges (and
    # overflows); pin the multinomial weights instead
    stats = s.weight_stats.row(0, named=True)
    expected = {
        "weight_min": 0.6247271882268575,
        "weight_max": 1.3120226973615323,
        "weight_mean": 0.9947273760015201,
        "weight_std": 0.05493659250756025,
        "weight_p01": 0.8251337794242161,
        "weight_p25": 0.972097606970619,
        "weight_p50": 0.9968156083037403,
        "weight_p75": 1.0164134507964089,
        "weight_p99": 1.1626440219480307,
    }
    assert stats == approx(expected, rel=1e-3)


def test_ITT_visit():
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
            weighted=True, weight_preexpansion=True, visit_colname="LTFU"
        ),
    )
    s.expand()
    s.fit()
    matrix = s.outcome_model[0]["outcome"].summary2().tables[1]["Coef."].to_list()
    expected = [
        -21.636346991788276,
        0.06813705852786496,
        0.02874152772603635,
        -0.0005734047013500563,
        0.2854740212699898,
        -0.0013729662310668182,
        -0.1939555961858531,
        0.006501915963316852,
        -0.4467079969655381,
        1.3870473474960576,
    ]
    assert matrix == approx(expected, rel=1e-3)
