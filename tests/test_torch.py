import numpy as np
import pandas as pd
import pytest
from pytest import approx

# torch is optional so skip the module without it
pytest.importorskip("torch")

from pySEQTarget import SEQopts, SEQuential  # noqa: E402
from pySEQTarget.data import load_data  # noqa: E402
from pySEQTarget.helpers._torch_fit import _TorchFit  # noqa: E402


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


def test_torch_matches_statsmodels_ITT():
    sm = _outcome_coefs(_fit("ITT", "statsmodels"))
    tc = _outcome_coefs(_fit("ITT", "torch"))
    assert tc == approx(sm, rel=1e-2, abs=2e-3)


def test_torch_matches_statsmodels_censoring_preexpansion():
    opts = dict(weighted=True, weight_preexpansion=True)
    sm = _outcome_coefs(_fit("censoring", "statsmodels", **opts))
    tc = _outcome_coefs(_fit("censoring", "torch", **opts))
    assert tc == approx(sm, rel=1e-2, abs=2e-3)


def test_torch_matches_statsmodels_censoring_postexpansion():
    opts = dict(weighted=True, weight_preexpansion=False)
    sm = _outcome_coefs(_fit("censoring", "statsmodels", **opts))
    tc = _outcome_coefs(_fit("censoring", "torch", **opts))
    assert tc == approx(sm, rel=1e-2, abs=2e-3)


def test_torch_standard_errors_match_statsmodels():
    sm_model = _fit("ITT", "statsmodels").outcome_model[0]["outcome"]
    tc_model = _fit("ITT", "torch").outcome_model[0]["outcome"]
    assert list(tc_model.bse) == approx(list(sm_model.bse), rel=1e-2, abs=1e-3)


def test_torch_bootstrap_survival_matches_statsmodels():
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

    assert risk_diff("torch") == approx(risk_diff("statsmodels"), rel=1e-2, abs=2e-3)


def _binary_frame(seed=1, n=2000):
    rng = np.random.default_rng(seed)
    x1, x2 = rng.normal(size=n), rng.normal(size=n)
    eta = -1.0 + 0.8 * x1 - 0.5 * x2
    y = rng.binomial(1, 1 / (1 + np.exp(-eta)))
    return pd.DataFrame({"y": y, "x1": x1, "x2": x2})


def test_torch_multinomial_label_detection():
    rng = np.random.default_rng(0)
    n = 900
    x = rng.normal(size=n)
    df = pd.DataFrame({"y": rng.integers(0, 3, size=n), "x": x})

    model = _TorchFit("y ~ x", df)
    assert list(model.classes_) == [0, 1, 2]

    probs = model.predict(df)
    assert probs.shape == (n, 3)
    assert probs.sum(axis=1) == approx(np.ones(n), abs=1e-5)


def test_torch_warm_start_reaches_same_optimum():
    df = _binary_frame()
    cold = _TorchFit("y ~ x1 + x2", df)
    warm = _TorchFit(
        "y ~ x1 + x2",
        df,
        start_params=(cold.params.values, list(cold.model.exog_names)),
    )
    assert list(warm.params) == approx(list(cold.params), rel=1e-3, abs=1e-3)


def test_torch_fit_pickle_roundtrip():
    # offload and parallel bootstrap pickle fitted models
    import pickle

    df = _binary_frame()
    m = _TorchFit("y ~ x1 + x2", df)

    m2 = pickle.loads(pickle.dumps(m))

    assert list(m2.params) == approx(list(m.params), rel=1e-12, abs=1e-12)
    assert m2.predict(df) == approx(m.predict(df), rel=1e-10, abs=1e-12)
    assert list(m2.bse) == approx(list(m.bse), rel=1e-10, abs=1e-12)
    assert str(m2.summary())


def test_torch_offload_bootstrap_survival():
    # End-to-end: offload=True pickles every fitted model to disk via joblib.
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
            glm_package="torch",
            bootstrap_nboot=2,
            seed=7,
            km_curves=True,
            offload=True,
        ),
    )
    s.expand()
    s.bootstrap()
    s.fit()
    s.survival()
    assert s.km_data.height > 0


def test_torch_parallel_bootstrap():
    # End-to-end: parallel=True pickles the SEQuential object into worker
    # processes and the fitted models back.
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
            glm_package="torch", bootstrap_nboot=2, seed=7, parallel=True, ncores=2
        ),
    )
    s.expand()
    s.bootstrap()
    s.fit()
    assert len(s.outcome_model) == 3  # main + 2 replicates
