"""Treatment weights do not depend on the order of treatment_level."""

import numpy as np
import pytest

from pySEQTarget import SEQopts, SEQuential
from pySEQTarget.data import load_data


def _weights(dataset, treatment_level, preexpansion):
    s = SEQuential(
        load_data(dataset),
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
            weight_preexpansion=preexpansion,
            treatment_level=treatment_level,
        ),
    )
    s.expand()
    s.fit()
    return s.DT.sort(["ID", "trial", "followup"])["weight"].to_numpy()


@pytest.mark.filterwarnings("ignore")
@pytest.mark.parametrize("preexpansion", [False, True])
@pytest.mark.parametrize(
    "dataset, levels",
    [("SEQdata", [0, 1]), ("SEQdata_multitreatment", [1, 2])],
)
def test_weights_match_with_treatment_level_reversed(dataset, levels, preexpansion):
    forward = _weights(dataset, levels, preexpansion)
    reverse = _weights(dataset, levels[::-1], preexpansion)
    assert np.allclose(forward, reverse)
