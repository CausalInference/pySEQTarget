import os

import pytest

from pySEQTarget import SEQopts, SEQuential
from pySEQTarget.data import load_data


@pytest.mark.skipif(
    os.getenv("CI") == "true", reason="Parallelism test hangs in CI environment"
)
def test_parallel_ITT():
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
        parameters=SEQopts(parallel=True, bootstrap_nboot=2, ncores=1),
    )
    s.expand()
    s.bootstrap()
    s.fit()
    matrix = s.outcome_model[0]["outcome"].summary2().tables[1]["Coef."].to_list()
    assert matrix == [
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
