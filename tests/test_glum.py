import numpy as np
import pytest
from pytest import approx

from pySEQTarget import SEQopts, SEQuential
from pySEQTarget.data import load_data


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


def test_glm_package_invalid_raises():
    with pytest.raises(ValueError, match="glm_package"):
        SEQopts(glm_package="sklearn")


def test_glum_matches_statsmodels_ITT():
    sm = _outcome_coefs(_fit("ITT", "statsmodels"))
    gl = _outcome_coefs(_fit("ITT", "glum"))
    assert gl == approx(sm, rel=1e-2, abs=2e-3)


def test_glum_matches_statsmodels_censoring_preexpansion():
    opts = dict(weighted=True, weight_preexpansion=True)
    sm = _outcome_coefs(_fit("censoring", "statsmodels", **opts))
    gl = _outcome_coefs(_fit("censoring", "glum", **opts))
    assert gl == approx(sm, rel=1e-2, abs=2e-3)


def test_glum_matches_statsmodels_censoring_postexpansion():
    opts = dict(weighted=True, weight_preexpansion=False)
    sm = _outcome_coefs(_fit("censoring", "statsmodels", **opts))
    gl = _outcome_coefs(_fit("censoring", "glum", **opts))
    assert gl == approx(sm, rel=1e-2, abs=2e-3)


def test_glum_weight_models_are_glum_backed():
    # The numerator/denominator binary logit models should be fit by glum,
    # not statsmodels, when glm_package="glum".
    from pySEQTarget.helpers._glum_fit import _GlumFit

    s = _fit("censoring", "glum", weighted=True, weight_preexpansion=True)
    assert all(isinstance(m, _GlumFit) for m in s.numerator_model if m is not None)
    assert all(isinstance(m, _GlumFit) for m in s.denominator_model if m is not None)


def test_glum_LTFU_runs():
    # Near-separation model (large coefficients): the two optimizers diverge,
    # so this is a smoke test that the glum censoring-weight path runs and
    # yields finite coefficients rather than an exact-equivalence check.
    s = _fit(
        "ITT",
        "glum",
        dataset="SEQdata_LTFU",
        weighted=True,
        weight_preexpansion=True,
        cense_colname="LTFU",
    )
    coefs = np.array(_outcome_coefs(s))
    assert np.all(np.isfinite(coefs))
    assert s.cense_numerator_model is not None
    assert s.cense_denominator_model is not None


def test_glum_summary_is_printable_and_consistent():
    # SEQoutput.summary() calls model.summary(); the glum wrapper must support
    # it (regression for the short-course render). tables[1] should be the coef
    # table, matching statsmodels' summary2 layout.
    s = _fit("ITT", "glum")
    model = s.outcome_model[0]["outcome"]

    smry = model.summary()
    assert str(smry)  # renders without error

    coef_col = model.summary().tables[1]["Coef."].to_list()
    assert coef_col == approx(list(model.params), rel=1e-9, abs=1e-9)


def test_glum_standard_errors_match_statsmodels():
    sm_model = _fit("ITT", "statsmodels").outcome_model[0]["outcome"]
    gl_model = _fit("ITT", "glum").outcome_model[0]["outcome"]
    assert list(gl_model.bse) == approx(list(sm_model.bse), rel=1e-2, abs=1e-3)


def test_glum_summary_via_seqoutput():
    # Mirrors the short-course usage: results.summary("numerator"/"outcome").
    s = _fit("censoring", "glum", weighted=True, weight_preexpansion=True)
    out = s.collect()
    for kind in ("numerator", "denominator", "outcome"):
        summaries = out.summary(kind)
        assert len(summaries) >= 1
        assert all(str(sm) for sm in summaries)


def test_glum_bootstrap_survival_matches_statsmodels():
    # Exercises the prediction-caching path in _survival_pred (transform=False
    # and .model.data.design_info on the glum wrapper) and confirms the point
    # risk estimates match the statsmodels backend.
    common = dict(bootstrap_nboot=3, seed=42, km_curves=True)

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

    assert risk_diff("glum") == approx(risk_diff("statsmodels"), rel=1e-2, abs=2e-3)


def _run_bootstrap_glum_outcome_coefs(monkeypatch, disable_warm_start):
    """Run a small bootstrap fit with glum and return each replicate's outcome coefs."""
    import pySEQTarget.helpers._glum_fit as glm_mod

    if disable_warm_start:
        original = glm_mod._fit_glum

        def patched(*args, **kwargs):
            kwargs["start_params"] = None
            return original(*args, **kwargs)

        monkeypatch.setattr(glm_mod, "_fit_glum", patched)

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
        parameters=SEQopts(glm_package="glum", bootstrap_nboot=3, seed=42),
    )
    s.expand()
    s.bootstrap()
    s.fit()
    return [list(m["outcome"].params.values) for m in s.outcome_model]


def test_glum_warm_start_is_wired_through_bootstrap(monkeypatch):
    # The main fit gets start_params=None (cold); every bootstrap replicate gets
    # the cached (values, names) tuple. Capture start_params on each call to
    # _fit_glum and assert the wiring.
    import pySEQTarget.helpers._glum_fit as glm_mod

    seen = []
    original = glm_mod._fit_glum

    def spy(*args, **kwargs):
        seen.append(kwargs.get("start_params"))
        return original(*args, **kwargs)

    monkeypatch.setattr(glm_mod, "_fit_glum", spy)

    _run_bootstrap_glum_outcome_coefs(monkeypatch, disable_warm_start=False)

    assert len(seen) >= 4  # main + 3 bootstrap replicates
    assert seen[0] is None  # main fit: cold start
    for sp in seen[1:]:
        assert sp is not None
        values, names = sp
        assert len(values) == len(names)
        assert names[0] == "Intercept"


def test_glum_warm_start_matches_cold_start_outcome_coefs(monkeypatch):
    # Warm-start is a pure convergence optimisation: it must reach the same
    # optimum as a cold start. Compare per-replicate outcome coefficients with
    # warm-start enabled (default) vs forcibly disabled.
    warm = _run_bootstrap_glum_outcome_coefs(monkeypatch, disable_warm_start=False)
    cold = _run_bootstrap_glum_outcome_coefs(monkeypatch, disable_warm_start=True)

    # Both runs reach an optimum within glum's coordinate-descent tolerance, but
    # the warm-start path stops at a point a few iterations earlier and so
    # differs from the cold-start path within that tolerance. A genuine wiring
    # regression would shift coefficients by several percent or more.
    assert len(warm) == len(cold)
    for w, c in zip(warm, cold):
        assert w == approx(c, rel=1e-2, abs=1e-4)


def test_glum_design_cache_avoids_reparsing_on_bootstrap(monkeypatch):
    # The main fit calls patsy.dmatrices to build the design info; every
    # bootstrap replicate should hit the cache and reuse it via
    # patsy.build_design_matrices instead of re-parsing the formula.
    import patsy

    import pySEQTarget.helpers._glum_fit as glm_mod

    real_dmatrices = patsy.dmatrices
    real_build = patsy.build_design_matrices
    parsed_formulas = []
    build_calls = [0]

    def spy_dmatrices(formula, *a, **kw):
        parsed_formulas.append(formula)
        return real_dmatrices(formula, *a, **kw)

    def spy_build(*a, **kw):
        build_calls[0] += 1
        return real_build(*a, **kw)

    monkeypatch.setattr(glm_mod.patsy, "dmatrices", spy_dmatrices)
    monkeypatch.setattr(glm_mod.patsy, "build_design_matrices", spy_build)

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
        parameters=SEQopts(glm_package="glum", bootstrap_nboot=3, seed=42),
    )
    s.expand()
    s.bootstrap()
    s.fit()

    # ITT has no weight models, so the only formula reaching _fit_glum is the
    # outcome formula. It must be parsed exactly once (the main fit); the three
    # bootstrap replicates must reuse the cached dinfo via build_design_matrices.
    outcome_formula = f"{s.outcome_col} ~ {s.covariates}"
    assert parsed_formulas.count(outcome_formula) == 1
    assert build_calls[0] >= 3
    # The cache survives onto self for inspection / future replicates.
    assert outcome_formula in s._patsy_design_cache


def _run_bootstrap_outcome_coefs_with_cache_disabled(monkeypatch):
    """Run a small bootstrap fit with the design cache forcibly disabled."""
    import pySEQTarget.helpers._glum_fit as glm_mod

    original = glm_mod._fit_glum

    def patched(*args, **kwargs):
        kwargs["design_cache"] = None
        return original(*args, **kwargs)

    monkeypatch.setattr(glm_mod, "_fit_glum", patched)

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
        parameters=SEQopts(glm_package="glum", bootstrap_nboot=3, seed=42),
    )
    s.expand()
    s.bootstrap()
    s.fit()
    return [list(m["outcome"].params.values) for m in s.outcome_model]


def test_glum_design_cache_matches_no_cache_outcome_coefs(monkeypatch):
    # The design cache freezes the categorical column structure to the main
    # fit's columns. On SEQdata both arms appear in every bootstrap resample, so
    # a no-cache run encodes the same columns from scratch each time and must
    # converge to the same coefficients (within glum's tolerance).
    cached = _run_bootstrap_glum_outcome_coefs(monkeypatch, disable_warm_start=False)
    no_cache = _run_bootstrap_outcome_coefs_with_cache_disabled(monkeypatch)

    assert len(cached) == len(no_cache)
    for c, nc in zip(cached, no_cache):
        assert c == approx(nc, rel=1e-2, abs=1e-4)


def test_glum_design_cache_handles_categorical_level_reordering():
    # A bootstrap resample can realise the same categorical levels in a
    # different ORDER than the cached design_info froze (e.g. polars->pandas
    # appearance order on the full data vs sorted order on a resample). The
    # cached build_design_matrices path must re-align the categories instead of
    # raising "mismatching levels". Regression for the short-course render.
    import numpy as np
    import pandas as pd

    from pySEQTarget.helpers._glum_fit import _fit_glum

    rng = np.random.default_rng(0)
    n = 2000
    levels = ["16-29", "30-39", "40-49", "50+"]
    formula = "y ~ age_grp + x"
    cache = {}

    # Main fit: categories in a NON-sorted order; this freezes the cache.
    main = pd.DataFrame(
        {
            "age_grp": pd.Categorical(
                rng.choice(levels, n), categories=["30-39", "16-29", "40-49", "50+"]
            ),
            "x": rng.standard_normal(n),
            "y": (rng.random(n) < 0.4).astype(int),
        }
    )
    m = _fit_glum(formula, main, design_cache=cache)

    # Bootstrap: same levels, sorted order — the crash trigger.
    boot = pd.DataFrame(
        {
            "age_grp": pd.Categorical(
                rng.choice(levels, n), categories=["16-29", "30-39", "40-49", "50+"]
            ),
            "x": rng.standard_normal(n),
            "y": (rng.random(n) < 0.4).astype(int),
        }
    )
    mb = _fit_glum(formula, boot, design_cache=cache)

    assert np.all(np.isfinite(mb.params.values))
    # The cached column structure is preserved (categories re-aligned, not reparsed).
    assert list(mb.params.index) == list(m.params.index)


def test_glum_model_pickle_roundtrip_preserves_predictions():
    # _GlumFit holds a patsy DesignInfo, which cannot be pickled (patsy #26).
    # It must rebuild the DesignInfo on unpickle from the stored formula +
    # reference frame so fitted models can cross a process boundary (parallel
    # bootstrap, offload). The roundtrip must preserve params and predictions
    # exactly and yield a usable design_info.
    import pickle

    import numpy as np
    import pandas as pd

    from pySEQTarget.helpers._glum_fit import _fit_glum

    rng = np.random.default_rng(0)
    n = 500
    df = pd.DataFrame(
        {
            "y": (rng.random(n) < 0.4).astype(int),
            "x": rng.standard_normal(n),
            "g": pd.Categorical(rng.choice(["a", "b", "c"], n), categories=["a", "b", "c"]),
        }
    )

    m = _fit_glum("y ~ x + g", df)
    pred = m.predict(df)

    m2 = pickle.loads(pickle.dumps(m))

    assert list(m2.params.index) == list(m.params.index)
    assert list(m2.params.values) == approx(list(m.params.values), rel=1e-12, abs=1e-12)
    assert list(m2.exog_names) == list(m.exog_names)
    # design_info is rebuilt and reproduces the frozen column structure
    assert list(m2.model.data.design_info.column_names) == list(
        m.model.data.design_info.column_names
    )
    # predictions are bit-identical through both predict paths
    np.testing.assert_array_equal(m2.predict(df), pred)
    np.testing.assert_array_equal(
        m2.predict(m2._X_design, transform=False),
        m.predict(m._X_design, transform=False),
    )


def test_glum_warm_start_dropped_when_design_columns_mismatch():
    # The defensive guard in _fit_glum: a (values, names) tuple whose names
    # don't line up with the patsy design matrix must be ignored, falling back
    # to the cold-start init and producing the same fit as start_params=None.
    import numpy as np
    import pandas as pd

    from pySEQTarget.helpers._glum_fit import _fit_glum

    rng = np.random.default_rng(0)
    n = 1000
    df = pd.DataFrame(
        {
            "x1": rng.standard_normal(n),
            "x2": rng.standard_normal(n),
            "y": (rng.random(n) < 0.4).astype(int),
        }
    )

    ref = _fit_glum("y ~ x1 + x2", df)
    bogus = (np.zeros(5), ["Intercept", "wrong", "names", "here", "extra"])
    bogus_fit = _fit_glum("y ~ x1 + x2", df, start_params=bogus)

    assert list(bogus_fit.params.values) == approx(
        list(ref.params.values), rel=1e-8, abs=1e-12
    )
