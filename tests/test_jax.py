import numpy as np
import pandas as pd
import pytest
from pytest import approx

from pySEQTarget import SEQopts, SEQuential
from pySEQTarget.data import load_data
from pySEQTarget.helpers._jax_fit import _JaxFit


def _fit(method, glm_package, dataset="SEQdata", **opts):
    data = load_data(dataset)
    s = SEQuential(
        data,
        id_col="ID",
        time_col="time",
        eligible_col="eligible",
        treatment_col="tx_init",
        outcome_col="outcome",
        time_varying_cols=["N", "L", "P"],
        fixed_cols=["sex"],
        method=method,
        parameters=SEQopts(glm_package=glm_package, **opts),
    )
    s.expand()
    s.fit()
    return s


def _outcome_coefs(s):
    return list(s.outcome_model[0]["outcome"].params)


def test_jax_matches_statsmodels_ITT():
    sm = _outcome_coefs(_fit("ITT", "statsmodels"))
    jx = _outcome_coefs(_fit("ITT", "jax"))
    assert jx == approx(sm, rel=1e-2, abs=2e-3)


def test_jax_matches_statsmodels_censoring_preexpansion():
    opts = dict(weighted=True, weight_preexpansion=True)
    sm = _outcome_coefs(_fit("censoring", "statsmodels", **opts))
    jx = _outcome_coefs(_fit("censoring", "jax", **opts))
    assert jx == approx(sm, rel=1e-2, abs=2e-3)


def test_jax_matches_statsmodels_censoring_postexpansion():
    opts = dict(weighted=True, weight_preexpansion=False)
    sm = _outcome_coefs(_fit("censoring", "statsmodels", **opts))
    jx = _outcome_coefs(_fit("censoring", "jax", **opts))
    assert jx == approx(sm, rel=1e-2, abs=2e-3)


def test_jax_standard_errors_match_statsmodels():
    sm_model = _fit("ITT", "statsmodels").outcome_model[0]["outcome"]
    jx_model = _fit("ITT", "jax").outcome_model[0]["outcome"]
    assert list(jx_model.bse) == approx(list(sm_model.bse), rel=1e-2, abs=1e-3)


def test_jax_bootstrap_survival_matches_statsmodels():
    common = dict(bootstrap_nboot=2, seed=1636, km_curves=True)

    def risk_diff(pkg):
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
            parameters=SEQopts(glm_package=pkg, **common),
        )
        s.expand()
        s.bootstrap()
        s.fit()
        s.survival()
        rd = s.risk_estimates["risk_difference"]
        assert rd["RD 95% LCI"].null_count() == 0
        return rd["Risk Difference"].to_list()

    assert risk_diff("jax") == approx(risk_diff("statsmodels"), rel=1e-2, abs=2e-3)


def _binary_frame(seed=1, n=2000):
    rng = np.random.default_rng(seed)
    x1, x2 = rng.normal(size=n), rng.normal(size=n)
    eta = -1.0 + 0.8 * x1 - 0.5 * x2
    y = rng.binomial(1, 1 / (1 + np.exp(-eta)))
    return pd.DataFrame({"y": y, "x1": x1, "x2": x2})


def test_jax_multinomial_label_detection():
    rng = np.random.default_rng(0)
    n = 900
    x = rng.normal(size=n)
    df = pd.DataFrame({"y": rng.integers(0, 3, size=n), "x": x})

    model = _JaxFit("y ~ x", df, num_epochs=300)
    assert list(model.classes_) == [0, 1, 2]

    probs = model.predict(df)
    assert probs.shape == (n, 3)
    assert probs.sum(axis=1) == approx(np.ones(n), abs=1e-5)


def test_jax_warm_start_reaches_same_optimum():
    df = _binary_frame()
    cold = _JaxFit("y ~ x1 + x2", df)
    warm = _JaxFit(
        "y ~ x1 + x2",
        df,
        start_params=(cold.params.values, list(cold.model.exog_names)),
    )
    assert list(warm.params) == approx(list(cold.params), rel=1e-3, abs=1e-3)
