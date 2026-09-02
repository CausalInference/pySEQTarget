import numpy as np
import polars as pl
import pytest
from statsmodels.tools.sm_exceptions import SingularMatrixWarning

from pySEQTarget import SEQopts, SEQuential
from pySEQTarget.data import load_data


def _model(data, **opts):
    return SEQuential(
        data,
        id_col="ID",
        time_col="time",
        eligible_col="eligible",
        treatment_col="tx_init",
        outcome_col="outcome",
        time_varying_cols=["N", "L", "P", "grp"],
        fixed_cols=["sex"],
        method="ITT",
        parameters=SEQopts(km_curves=True, **opts),
    )


def test_string_time_varying_covariate_bootstrap():
    """A categorical (string) time-varying covariate should run through the full
    bootstrap pipeline and produce risk estimates."""
    data = load_data("SEQdata")
    rng = np.random.RandomState(1)
    data = data.with_columns(
        pl.Series("grp", rng.choice(["a", "b", "c"], size=data.height))
    )

    s = _model(data, bootstrap_nboot=3, seed=42)
    s.expand()
    s.bootstrap()
    s.fit()
    s.survival()

    assert "grp_bas" in s.DT.columns
    rd = s.risk_estimates["risk_difference"]
    assert rd.height > 0
    assert rd.select(["RD 95% LCI", "RD 95% UCI"]).null_count().to_series().sum() == 0


def test_rare_level_not_dropped_by_bootstrap_resample():
    """A rare categorical level absent from some bootstrap resamples must not
    crash counterfactual prediction. The full-data level set is fixed at fit
    time so every resample shares a stable factor encoding."""
    data = load_data("SEQdata")
    ids = data["ID"].unique().to_list()
    rng = np.random.RandomState(0)
    # Level "c" appears for a single ID only, so aggressive subsampling will
    # produce resamples that omit it entirely.
    grp = pl.Series(
        "grp",
        np.where(
            np.isin(data["ID"].to_numpy(), [ids[0]]),
            "c",
            rng.choice(["a", "b"], size=data.height),
        ),
    )
    data = data.with_columns(grp)

    s = _model(data, bootstrap_nboot=8, bootstrap_sample=0.5, seed=7)
    s.expand()
    s.bootstrap()
    # The frozen level set makes resamples lacking level "c" carry an all-zero
    # design column, so statsmodels (>= 0.15) reports the deliberate rank
    # deficiency; pinv resolves it with a zero coefficient. The warning is the
    # expected signature of the mechanism under test.
    with pytest.warns(SingularMatrixWarning):
        s.fit()
    s.survival()  # previously raised ValueError on NaN predictions

    rd = s.risk_estimates["risk_difference"]
    assert rd.height > 0
    assert not rd["RD 95% LCI"].is_nan().any()
    assert not rd["RD 95% UCI"].is_nan().any()
