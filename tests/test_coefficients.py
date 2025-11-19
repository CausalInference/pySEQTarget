from pySEQ import SEQuential, SEQopts
from pySEQ.data import load_data

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
        method = "ITT",
        parameters=SEQopts()
    )
    s.expand()
    s.fit()
    matrix = s.outcome_model[0].summary2().tables[1]['Coef.'].to_list()
    assert matrix == [-6.828506035553367, 0.12717241010543864, 0.1893500309004178, 
                      0.03371515698762837, -0.00014691202235021713, 0.044566165558946304, 
                      0.000578777043905276, 0.0032906669395291782, -0.013392420492057825, 
                      0.20072409918428197]
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
        method = "dose-response",
        parameters=SEQopts(weighted=True,
                           weight_preexpansion=True)
    )
    s.expand()
    s.fit()
    print(s.weight_stats)
    print(s.numerator_models[0].summary())
    print(s.numerator_models[1].summary())
    
    matrix = s.outcome_model[0].summary2().tables[1]
    return print(matrix)
test_PreE_dose_response_covariates()
'''
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
        method = "dose-response",
        parameters=SEQopts(weighted=True)
    )
    assert s.covariates == "dose+dose_sq+followup+followup_sq+trial+trial_sq+sex+N_bas+L_bas+P_bas+followup*dose+followup*dose_sq"
    assert s.numerator == "sex+N_bas+L_bas+P_bas+followup+followup_sq+trial+trial_sq"
    assert s.denominator == "sex+N+L+P+N_bas+L_bas+P_bas+followup+followup_sq+trial+trial_sq"
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
        method = "censoring",
        parameters=SEQopts(weighted=True,
                           weight_preexpansion=True)
    )
    assert s.covariates == "tx_init_bas+followup+followup_sq+trial+trial_sq+sex+tx_init_bas*followup"
    assert s.numerator == "sex+time+time_sq"
    assert s.denominator == "ssex+N+L+P+time+time_sq"
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
        method = "censoring",
        parameters=SEQopts(weighted=True)
    )
    assert s.covariates == "tx_init_bas+followup+followup_sq+trial+trial_sq+sex+N_bas+L_bas+P_bas+tx_init_bas*followup"
    assert s.numerator == "sex+N_bas+L_bas+P_bas+followup+followup_sq+trial+trial_sq"
    assert s.denominator == "sex+N+L+P+N_bas+L_bas+P_bas+followup+followup_sq+trial+trial_sq"
    
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
        method = "censoring",
        parameters=SEQopts(weighted=True,
                           weight_preexpansion=True)
    )
    assert s.covariates == "tx_init_bas+followup+followup_sq+trial+trial_sq+sex+tx_init_bas*followup"
    assert s.numerator == "sex+time+time_sq"
    assert s.denominator == "sex+N+L+P+time+time_sq"
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
        method = "censoring",
        parameters=SEQopts(weighted=True,
                           weight_preexpansion=True,
                           excused=True,
                           excused_colnames=["ExcusedZero", "ExcusedOne"])
    )
    assert s.covariates == "tx_init_bas+followup+followup_sq+trial+trial_sq+tx_init_bas*followup"
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
        method = "censoring",
        parameters=SEQopts(weighted=True,
                           excused=True,
                           excused_colnames=["ExcusedZero", "ExcusedOne"])
    )
    assert s.covariates == "tx_init_bas+followup+followup_sq+trial+trial_sq+sex+N_bas+L_bas+P_bas+tx_init_bas*followup"
    assert s.numerator == "sex+N_bas+L_bas+P_bas+followup+followup_sq+trial+trial_sq"
    assert s.denominator == "sex+N+L+P+N_bas+L_bas+P_bas+followup+followup_sq+trial+trial_sq"
    return
'''