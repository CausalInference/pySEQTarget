from pySEQTarget import SEQopts, SEQuential
from pySEQTarget.data import load_data


def test_ITT_covariates():
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

    assert (
        s.covariates
        == "tx_init_bas+followup+followup_sq+trial+trial_sq+sex+N_bas+L_bas+P_bas"
    )
    assert s.numerator is None
    assert s.denominator is None
    return


def test_PreE_dose_response_covariates():
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

    assert s.covariates == "dose+dose_sq+followup+followup_sq+trial+trial_sq+sex"
    assert s.numerator == "sex+time+time_sq"
    assert s.denominator == "sex+N+L+P+time+time_sq"
    return


def test_PostE_dose_response_covariates():
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
    assert (
        s.covariates
        == "dose+dose_sq+followup+followup_sq+trial+trial_sq+sex+N_bas+L_bas+P_bas"
    )
    assert s.numerator == "sex+N_bas+L_bas+P_bas+followup+followup_sq+trial+trial_sq"
    assert (
        s.denominator
        == "sex+N+L+P+N_bas+L_bas+P_bas+followup+followup_sq+trial+trial_sq"
    )
    return


def test_PreE_censoring_covariates():
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
    assert s.covariates == "tx_init_bas+followup+followup_sq+trial+trial_sq+sex"
    assert s.numerator == "sex+time+time_sq"
    assert s.denominator == "sex+N+L+P+time+time_sq"
    return


def test_PostE_censoring_covariates():
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
    assert (
        s.covariates
        == "tx_init_bas+followup+followup_sq+trial+trial_sq+sex+N_bas+L_bas+P_bas"
    )
    assert s.numerator == "sex+N_bas+L_bas+P_bas+followup+followup_sq+trial+trial_sq"
    assert (
        s.denominator
        == "sex+N+L+P+N_bas+L_bas+P_bas+followup+followup_sq+trial+trial_sq"
    )

    return


def test_PreE_censoring_excused_covariates():
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
    assert s.covariates == "tx_init_bas+followup+followup_sq+trial+trial_sq"
    assert s.numerator is None
    assert s.denominator == "sex+N+L+P+time+time_sq"
    return


def test_PostE_censoring_excused_covariates():
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
            weighted=True, excused=True, excused_colnames=["excusedZero", "excusedOne"]
        ),
    )
    assert (
        s.covariates
        == "tx_init_bas+followup+followup_sq+trial+trial_sq+sex+N_bas+L_bas+P_bas"
    )
    assert s.numerator == "sex+N_bas+L_bas+P_bas+followup+followup_sq+trial+trial_sq"
    assert (
        s.denominator
        == "sex+N+L+P+N_bas+L_bas+P_bas+followup+followup_sq+trial+trial_sq"
    )
    return
