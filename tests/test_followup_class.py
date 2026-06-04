import re

from pySEQTarget import SEQopts, SEQuential
from pySEQTarget.data import load_data


def test_followup_class_encodes_followup_as_a_factor():
    # followup_class=True makes the outcome model treat follow-up as categorical
    # (cast to category in _cast_categories), so it gains one patsy dummy
    # 'followup[T.<n>]' per non-reference follow-up level and loses the linear
    # followup / followup_sq pair. It is exclusive with followup_include, so that
    # is switched off here.
    s = SEQuential(
        load_data("SEQdata"),
        id_col="ID",
        time_col="time",
        eligible_col="eligible",
        treatment_col="tx_init",
        outcome_col="outcome",
        time_varying_cols=["N", "L", "P"],
        fixed_cols=["sex"],
        method="ITT",
        parameters=SEQopts(followup_class=True, followup_include=False),
    )
    s.expand()
    s.fit()
    names = list(s.outcome_model[0]["outcome"].params.index)

    # Categorical, not continuous: no linear follow-up terms
    assert "followup" not in names
    assert "followup_sq" not in names

    # One dummy per non-reference follow-up level
    dummies = [n for n in names if re.fullmatch(r"followup\[T\.\d+\]", n)]
    assert len(dummies) > 2
    assert len(dummies) == s.DT["followup"].n_unique() - 1
