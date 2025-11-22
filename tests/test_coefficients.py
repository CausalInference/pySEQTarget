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
    assert matrix == [-6.828506035553407, 0.18935003090041902, 0.12717241010542563, 
                      0.033715156987629266, -0.00014691202235029346, 0.044566165558944326, 
                      0.0005787770439053261, 0.0032906669395295026, -0.01339242049205771, 
                      0.20072409918428052]

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
    assert matrix == [-4.818288687908951, 0.511665606890965, 0.062028316788368384, 
                      0.025489681857269905, 0.00018215948440046585, -0.014019017637918164, 
                      0.001110238926667272]

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
    assert matrix == [-7.9113179326280445, 0.49092190701455873, 0.08903087485402544, 
                      0.026160806382879903, 0.00019078148503570062, 0.04445697224987294, 
                      0.0007051968052005897, 0.004316239095295115, 0.013762799304812959, 
                      0.3196331024454665]

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
    assert matrix == [-7.126398786875262, 0.2632047482928519, 0.13345454814736696,
                      0.03967181206032395, -0.00033089446793392585, 0.03763545026332514, 
                      0.0007588725152627089, 0.0036793093608788923, -0.022372677571544992, 
                      0.2441842617520696]
    
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
        method = "ITT",
        parameters=SEQopts(weighted=True,
                           weight_preexpansion=True,
                           cense_colname="LTFU")
    )
    s.expand()
    s.fit()
    matrix = s.outcome_model[0].summary2().tables[1]["Coef."].to_list()
    assert matrix == [-21.640523091572675, 0.06852351843717125, -0.19006360662233904,
                      0.02875095019383619, -0.0005762057433737245, 0.28554312978583674,
                      -0.001373044229622937, 0.00658914139445824, -0.44898959259422067, 
                      1.387508978803619]

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
        method = "ITT",
        parameters=SEQopts(weighted=True,
                           cense_colname="LTFU")
    )
    s.expand()
    s.fit()
    matrix = s.outcome_model[0].summary2().tables[1]["Coef."].to_list()
    matrix == [-21.640523091572796, -0.19006360662228572, 0.0685235184372898, 
               0.028750950193838918, -0.0005762057433736666, 0.28554312978583757, 
               -0.001373044229623057, 0.006589141394458155, -0.44898959259422394, 
               1.3875089788036237]

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
        method = "ITT",
        parameters=SEQopts(treatment_level=[1,2])
    )
    s.expand()
    s.fit()
    matrix = s.outcome_model[0].summary2().tables[1]["Coef."].to_list()
    assert matrix == [-47.505262164163625, 1.76628017234151, 22.79205044396338, 
                      0.14473536056627245, -0.003725499516376173, 0.2893070991930884, 
                      -0.004266608123938117, 0.05574429164512122, 0.7847862691929901, 
                      1.4703411759229423]
    
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
        method = "censoring",
        parameters=SEQopts(weighted = True,
                           weight_preexpansion=True,
                           treatment_level=[1,2])
    )
   s.expand()
   s.fit()
   matrix = s.outcome_model[0].summary2().tables[1]["Coef."].to_list()
   assert matrix == [-111.35419661939163, -12.571187230338328, 9.234157699403015, 
                     -0.6336774763031923, 0.016754692338530056, 5.8240772329087225, 
                     -0.08598454090661659]
