from pySEQ import SEQuential, SEQopts
from pySEQ.data import load_data

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
        method = "dose-response",
        parameters=SEQopts(weighted=True,
                           weight_preexpansion=True)
    )
    s.expand()
    s.fit()
    matrix = s.outcome_model[0].summary2().tables[1]['Coef.'].to_list()
    assert matrix == [-4.842735939069144, 0.14286755770151904, 0.055221018477671927, 
                      -0.000581657931537684, -0.008484541900408258, 0.00021073328759912806, 
                      0.010537967151467553, 0.0007772316818101141]
    return

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
        method = "dose-response",
        parameters=SEQopts(weighted=True)
    )
    
    s.expand()
    s.fit()
    matrix = s.outcome_model[0].summary2().tables[1]['Coef.'].to_list()
    assert matrix == [-6.378405714539014, 0.1783781183733739, 0.04468084245850419, 
                      -0.00028721095405992604, -0.0001680250387661585, -1.6465184245665544e-05, 
                      0.03796880915870554, 0.0006587394643895277, 0.002530895897349293, 
                      -0.039757502333589184, 0.1638394382909944]
    
    return print(matrix)

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
        method = "censoring",
        parameters=SEQopts(weighted=True,
                           weight_preexpansion=True)
    )
    s.expand()
    s.fit()
    matrix = s.outcome_model[0].summary2().tables[1]['Coef.'].to_list()
    assert matrix == [-4.661102616366661, 0.06322831388844205, 0.5000738277717721, 
                      0.007974580521778882, 0.0005337038990034418, -0.011577561000157839, 
                      0.0010459271332870575]
    return

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
        method = "censoring",
        parameters=SEQopts(weighted=True)
    )
    s.expand()
    s.fit()
    matrix = s.outcome_model[0].summary2().tables[1]["Coef."].to_list()
    assert matrix == [-7.552997975919737, 0.09676401101585111, 0.4731499690921231, 
                      0.009424470533477319, 0.0005314170238427062, 0.04111388864102163, 
                      0.0007102010905924766, 0.003667143725614561, 0.007220844654484709, 
                      0.30098248859104154]
'''
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