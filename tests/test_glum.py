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
