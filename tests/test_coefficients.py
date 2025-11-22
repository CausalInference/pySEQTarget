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
    assert matrix == [-6.265901713761531, 0.14065954021957594, 0.048626017624679704, 
                      -0.0004688287307505834, -0.003975906839775267, 0.00016676441745740924, 
                      0.03866279977096397, 0.0005928449623613982, 0.0030001459817949844, 
                      -0.02106338184559446, 0.14867250693140854]

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
    assert matrix == [-4.818288687908983, 0.06202831678835523, 0.5116656068909778, 
                      0.025489681857267917, 0.00018215948440049318, -0.014019017637919164, 
                      0.0011102389266667307]

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
    assert matrix == [-7.911317932628025, 0.08903087485404401, 0.4909219070145824, 
                      0.026160806382874355, 0.0001907814850356967, 0.04445697224986894, 
                      0.0007051968052006822, 0.00431623909529477, 0.013762799304812941, 
                      0.3196331024454667]
    return print(matrix)

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
        method = "censoring",
        parameters=SEQopts(weighted=True,
                           weight_preexpansion=True,
                           excused=True,
                           excused_colnames=["excusedZero", "excusedOne"])
    )
    s.expand()
    s.fit()
    matrix = s.outcome_model[0].summary2().tables[1]['Coef.'].to_list()
    assert matrix == [-6.175691049418418, 1.3493634846413598, 0.1072284696749134, 
                      -0.003977965364113033, 0.06959432825811135, -0.00034297574787048573]

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
        method = "censoring",
        parameters=SEQopts(weighted=True,
                           excused=True,
                           excused_colnames=["excusedZero", "excusedOne"],
                           weight_max=1)
    )
    s.expand()
    s.fit()
    matrix = s.outcome_model[0].summary2().tables[1]["Coef."].to_list()
    # Doesn't converge on test data (have to input max weight)!
    assert matrix == [-7.126398786875212, 0.13345454814736768, 0.2632047482928211, 
                      0.03967181206032499, -0.0003308944679339907, 0.03763545026332593, 
                      0.0007588725152627008, 0.0036793093608787847, -0.022372677571544725, 
                      0.24418426175207003]
    return print(matrix)
