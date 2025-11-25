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
        0.12717241010542563,
        0.033715156987629266,
        -0.00014691202235029346,
        0.044566165558944326,
        0.0005787770439053261,
        0.0032906669395295026,
        -0.01339242049205771,
        0.20072409918428052,
    ]
    assert [round(x, 3) for x in matrix] == [round(x, 3) for x in expected]


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
        -4.842735939069144,
        0.14286755770151904,
        0.055221018477671927,
        -0.000581657931537684,
        -0.008484541900408258,
        0.00021073328759912806,
        0.010537967151467553,
        0.0007772316818101141,
    ]
    assert [round(x, 3) for x in matrix] == [round(x, 3) for x in expected]


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
        parameters=SEQopts(weighted=True),
    )

    s.expand()
    s.fit()
    matrix = s.outcome_model[0]["outcome"].summary2().tables[1]["Coef."].to_list()
    expected = [
        -6.265901713761531,
        0.14065954021957594,
        0.048626017624679704,
        -0.0004688287307505834,
        -0.003975906839775267,
        0.00016676441745740924,
        0.03866279977096397,
        0.0005928449623613982,
        0.0030001459817949844,
        -0.02106338184559446,
        0.14867250693140854,
    ]
    assert [round(x, 3) for x in matrix] == [round(x, 3) for x in expected]


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
        -4.872373936951975,
        0.48389186624295133,
        0.0477349453301334,
        0.029127276869076173,
        4.784054961154824e-05,
        -0.013614654772205668,
        0.0011281734101133744,
    ]

    assert [round(x, 3) for x in matrix] == [round(x, 3) for x in expected]


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
        parameters=SEQopts(weighted=True),
    )
    s.expand()
    s.fit()
    matrix = s.outcome_model[0]["outcome"].summary2().tables[1]["Coef."].to_list()
    expected = [
        -9.172266785106519,
        0.4707554720116625,
        0.08162617232478116,
        0.029021087196430605,
        7.8937226861939e-05,
        0.06700192286925702,
        0.0005834323664644191,
        0.004870212765388434,
        0.013503198983327514,
        0.4466573801510379,
    ]
    assert [round(x, 3) for x in matrix] == [round(x, 3) for x in expected]


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
        -6.460912082691973,
        1.309708035546933,
        0.10853511682679658,
        -0.0038913520688693823,
        0.08849129909709463,
        -0.000647578869153453,
    ]
    print(matrix)
    assert [round(x, 3) for x in matrix] == [round(x, 3) for x in expected]


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
            excused=True,
            excused_colnames=["excusedZero", "excusedOne"],
            weight_max=1,
        ),
    )
    s.expand()
    s.fit()
    matrix = s.outcome_model[0]["outcome"].summary2().tables[1]["Coef."].to_list()
    expected = [
        -6.782732929102242,
        0.26371172100905477,
        0.13625528598217598,
        0.040580427030886,
        -0.000343018323531494,
        0.031185150775465315,
        0.000784356550754563,
        0.004338417236024277,
        -0.013052187516528172,
        0.20402680950820007,
    ]
    assert [round(x, 3) for x in matrix] == [round(x, 3) for x in expected]


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
        -21.640523091572796,
        0.0685235184372898,
        -0.19006360662228572,
        0.028750950193838918,
        -0.0005762057433736666,
        0.28554312978583757,
        -0.001373044229623057,
        0.006589141394458155,
        -0.44898959259422394,
        1.3875089788036237,
    ]
    assert [round(x, 3) for x in matrix] == [round(x, 3) for x in expected]


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
        parameters=SEQopts(weighted=True, cense_colname="LTFU"),
    )
    s.expand()
    s.fit()
    matrix = s.outcome_model[0]["outcome"].summary2().tables[1]["Coef."].to_list()
    expected = [
        -21.640523091572796,
        0.0685235184372898,
        -0.19006360662228572,
        0.028750950193838918,
        -0.0005762057433736666,
        0.28554312978583757,
        -0.001373044229623057,
        0.006589141394458155,
        -0.44898959259422394,
        1.3875089788036237,
    ]
    assert [round(x, 3) for x in matrix] == [round(x, 3) for x in expected]


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
        22.79205044396338,
        0.14473536056627245,
        -0.003725499516376173,
        0.2893070991930884,
        -0.004266608123938117,
        0.05574429164512122,
        0.7847862691929901,
        1.4703411759229423,
    ]
    assert [round(x, 3) for x in matrix] == [round(x, 3) for x in expected]


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
    matrix = s.outcome_model[0]["outcome"].summary2().tables[1]["Coef."].to_list()
    expected = [
        -109.99715622379995,
        -12.536816769546702,
        9.22013733949143,
        -0.6129380297017852,
        0.01597877250531723,
        5.743984176710672,
        -0.08478678955657822,
    ]
    assert [round(x, 3) for x in matrix] == [round(x, 3) for x in expected]
