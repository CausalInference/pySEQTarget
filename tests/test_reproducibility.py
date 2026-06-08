import os

import numpy as np
import pytest

from pySEQTarget import SEQopts, SEQuential
from pySEQTarget.data import load_data


def _make_seq(seed, **extra_opts):
    data = load_data("SEQdata")
    return SEQuential(
        data,
        id_col="ID",
        time_col="time",
        eligible_col="eligible",
        treatment_col="tx_init",
        outcome_col="outcome",
        time_varying_cols=["N", "L", "P"],
        fixed_cols=["sex"],
        method="ITT",
        parameters=SEQopts(seed=seed, **extra_opts),
    )


def test_unseeded_run_assigns_stable_concrete_seed():
    # With no seed, the hazard simulation must not fall back to the global,
    # never-reseeded np.random. Instead a concrete seed is drawn once, recorded
    # on self.seed, and held fixed for the life of the object so the seed is the
    # same before and after a run (and can be reported to reproduce it).
    s = _make_seq(seed=None, hazard_estimate=True)

    before = s.seed
    assert before is not None
    assert isinstance(before, int)
    assert 0 <= before < 2**32
    assert isinstance(s._rng, np.random.RandomState)

    s.expand()
    s.fit()
    s.hazard()

    assert s.seed == before


def test_two_unseeded_runs_are_deterministic():
    # With no seed supplied, runs use a fixed default seed (mirroring R), so two
    # otherwise identical unseeded runs produce the same hazard ratio.
    results = []
    for _ in range(2):
        s = _make_seq(seed=None, hazard_estimate=True)
        s.expand()
        s.fit()
        s.hazard()
        results.append(s.hazard_ratio["Hazard ratio"][0])

    assert results[0] == results[1]


def test_unseeded_captured_seed_reproduces_hazard():
    # The seed recorded on an unseeded run is the one actually used, so feeding
    # it back as an explicit seed reproduces the hazard ratio exactly.
    s1 = _make_seq(seed=None, hazard_estimate=True)
    captured = s1.seed
    s1.expand()
    s1.fit()
    s1.hazard()

    s2 = _make_seq(seed=captured, hazard_estimate=True)
    s2.expand()
    s2.fit()
    s2.hazard()

    assert s1.hazard_ratio["Hazard ratio"][0] == s2.hazard_ratio["Hazard ratio"][0]


def test_hazard_reproducible_with_seed():
    results = []
    for _ in range(2):
        s = _make_seq(seed=42, hazard_estimate=True)
        s.expand()
        s.fit()
        s.hazard()
        results.append(s.hazard_ratio)

    assert results[0]["Hazard ratio"][0] == results[1]["Hazard ratio"][0]


def test_hazard_bootstrap_se_reproducible_with_seed():
    results = []
    for _ in range(2):
        s = _make_seq(seed=42, hazard_estimate=True, bootstrap_nboot=3)
        s.expand()
        s.bootstrap()
        s.fit()
        s.hazard()
        results.append(s.hazard_ratio)

    assert results[0]["Hazard ratio"][0] == results[1]["Hazard ratio"][0]
    assert results[0]["LCI"][0] == results[1]["LCI"][0]
    assert results[0]["UCI"][0] == results[1]["UCI"][0]


@pytest.mark.skipif(
    os.getenv("CI") == "true", reason="Bootstrap reproducibility test hangs in CI"
)
def test_hazard_bootstrap_percentile_reproducible_with_seed():
    results = []
    for _ in range(2):
        s = _make_seq(
            seed=42,
            hazard_estimate=True,
            bootstrap_nboot=3,
            bootstrap_CI_method="percentile",
        )
        s.expand()
        s.bootstrap()
        s.fit()
        s.hazard()
        results.append(s.hazard_ratio)

    assert results[0]["Hazard ratio"][0] == results[1]["Hazard ratio"][0]
    assert results[0]["LCI"][0] == results[1]["LCI"][0]
    assert results[0]["UCI"][0] == results[1]["UCI"][0]


@pytest.mark.skipif(
    os.getenv("CI") == "true", reason="Reproducibility test hangs in CI"
)
def test_survival_reproducible_with_seed():
    results = []
    for _ in range(2):
        s = _make_seq(seed=42, km_curves=True)
        s.expand()
        s.fit()
        s.survival()
        results.append(s.km_data)

    np.testing.assert_allclose(
        results[0]["pred"].to_numpy(), results[1]["pred"].to_numpy(), atol=1e-14
    )


@pytest.mark.skipif(
    os.getenv("CI") == "true", reason="Bootstrap reproducibility test hangs in CI"
)
def test_survival_bootstrap_reproducible_with_seed():
    results = []
    for _ in range(2):
        s = _make_seq(seed=42, km_curves=True, bootstrap_nboot=3)
        s.expand()
        s.bootstrap()
        s.fit()
        s.survival()
        results.append(s.km_data)

    for col in ["pred", "SE", "LCI", "UCI"]:
        np.testing.assert_allclose(
            results[0][col].to_numpy(), results[1][col].to_numpy(), atol=1e-14
        )
