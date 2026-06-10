import os

import pytest
from pytest import approx

from pySEQTarget import SEQopts, SEQuential
from pySEQTarget.data import load_data


def _fit_weighted(bootstrap_nboot, parallel=False, glm_package="statsmodels", seed=42):
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
            glm_package=glm_package,
            weighted=True,
            # post-expansion weights are refit on every (resampled) replicate, so
            # this is the case where an in-process bootstrap clobbers the stored
            # main-fit weight models.
            weight_preexpansion=False,
            bootstrap_nboot=bootstrap_nboot,
            parallel=parallel,
            ncores=2,
            seed=seed,
        ),
    )
    s.expand()
    if bootstrap_nboot > 0:
        s.bootstrap()
    s.fit()
    return s


def _weight_params(s):
    return {
        "numerator": [list(m.params) for m in s.numerator_model if m is not None],
        "denominator": [list(m.params) for m in s.denominator_model if m is not None],
    }


def _assert_same(a, b):
    for kind in ("numerator", "denominator"):
        assert len(a[kind]) == len(b[kind])
        for pa, pb in zip(a[kind], b[kind]):
            assert pb == approx(pa, rel=1e-9, abs=1e-9)


@pytest.mark.parametrize("glm_package", ["statsmodels", "glum"])
def test_main_weight_models_preserved_after_bootstrap(glm_package):
    # _fit_numerator/_fit_denominator overwrite self.X_model on every fit, and a
    # serial bootstrap runs replicates in-process — so without preservation the
    # stored weight model (used by the numerator/denominator summary) would be
    # the last resampled replicate, not the main fit. The main-fit models must
    # survive the bootstrap loop.
    main = _weight_params(_fit_weighted(0, glm_package=glm_package))
    booted = _weight_params(_fit_weighted(3, glm_package=glm_package))
    _assert_same(main, booted)


@pytest.mark.skipif(
    os.getenv("CI") == "true", reason="Parallelism test hangs in CI environment"
)
def test_main_weight_models_match_across_serial_and_parallel():
    main = _weight_params(_fit_weighted(0, glm_package="glum"))
    serial = _weight_params(_fit_weighted(3, parallel=False, glm_package="glum"))
    parallel = _weight_params(_fit_weighted(3, parallel=True, glm_package="glum"))
    _assert_same(main, serial)
    _assert_same(main, parallel)
