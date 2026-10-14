"""Censoring weights use the probability of remaining under follow-up."""

import numpy as np
import pytest

from pySEQTarget import SEQopts, SEQuential
from pySEQTarget.data import load_data
from pySEQTarget.helpers._predict_model import _predict_model


@pytest.mark.filterwarnings("ignore")
def test_ltfu_weight_is_the_ratio_of_probabilities_of_remaining():
    s = SEQuential(
        load_data("SEQdata_LTFU"),
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
    # Under ITT the treatment weights are 1, so each weight is the LTFU factor
    rows = s.DT.filter(s.DT["followup"] > 0).head(500)
    lost_num = _predict_model(s, s.cense_numerator_model, rows)
    lost_den = _predict_model(s, s.cense_denominator_model, rows)
    expected = (1 - lost_num) / (1 - lost_den)
    assert np.allclose(rows["weight"].to_numpy(), expected)
