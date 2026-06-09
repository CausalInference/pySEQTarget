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
        parameters=SEQopts(parallel=True, bootstrap_nboot=2, ncores=1, seed=42),
    )
    s.expand()
    s.bootstrap()
    s.fit()
    matrix = s.outcome_model[0]["outcome"].summary2().tables[1]["Coef."].to_list()
    assert matrix == pytest.approx(
        [
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
        ],
        abs=1e-6,
    )


def _hazard_run(parallel, glm_package):
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
        parameters=SEQopts(
            glm_package=glm_package,
            hazard_estimate=True,
            bootstrap_nboot=4,
            ncores=2,
            parallel=parallel,
            seed=42,
        ),
    )
    s.expand()
    s.bootstrap()
    s.fit()
    s.hazard()
    hr = s.hazard_ratio
    return (hr["Hazard ratio"][0], hr["LCI"][0], hr["UCI"][0])


@pytest.mark.skipif(
    os.getenv("CI") == "true", reason="Parallelism test hangs in CI environment"
)
@pytest.mark.parametrize("glm_package", ["statsmodels", "glum"])
def test_parallel_hazard_matches_serial(glm_package):
    # The process-pool bootstrap must produce the same hazard ratio + CI as the
    # serial loop. Locks two fixes: (1) glum's _GlumFit is now picklable so the
    # fitted models survive crossing the process boundary, and (2) the worker
    # calls the raw fit body, so outcome_model[i] is a model dict (not a list)
    # for the hazard consumer to index. Previously crashed for both backends.
    serial = _hazard_run(parallel=False, glm_package=glm_package)
    parallel = _hazard_run(parallel=True, glm_package=glm_package)
    assert parallel == pytest.approx(serial, rel=1e-9, abs=1e-12)
