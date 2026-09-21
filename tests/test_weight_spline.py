"""weight_spline: model the baseline hazard in the weight models as a flexible
function of time, and fix the knots of every cr(x, df=N) term — in the weight
models or the outcome model — from the data that model is fit on, so the basis
is constant across bootstrap resamples.
"""

import re

import numpy as np
import patsy
import polars as pl
import pytest

from pySEQTarget import SEQopts, SEQuential
from pySEQTarget.data import load_data
from pySEQTarget.helpers._spline import _bake_spline_knots
from pySEQTarget.initialization import _cense_denominator, _cense_numerator
from pySEQTarget.initialization._denominator import _denominator
from pySEQTarget.initialization._numerator import _numerator

# Two expected warnings, neither marking a bad fit: _check_separation trips on the
# large (but precise, z ~ 1.8) coefficient of a small-scale cr() basis column, and
# an unconstrained hand-written cr() is collinear with the intercept, which pinv
# resolves after newton reports it.
pytestmark = [
    pytest.mark.filterwarnings("ignore:Possible perfect or quasi-complete"),
    pytest.mark.filterwarnings("ignore:Maximum Likelihood optimization failed"),
]

BAKED_PLAIN = r"cr\({var}, knots=\[[^]]*\], lower_bound=[-\d.]+, upper_bound=[-\d.]+"
BAKED = BAKED_PLAIN + r', constraints="center"\)'


def _model(**opts):
    s = SEQuential(
        load_data("SEQdata"),
        id_col="ID",
        time_col="time",
        eligible_col="eligible",
        treatment_col="tx_init",
        outcome_col="outcome",
        time_varying_cols=["N", "L", "P"],
        fixed_cols=["sex"],
        method="censoring",
        parameters=SEQopts(weighted=True, seed=42, **opts),
    )
    s.expand()
    s.fit()
    return s


class _Opts:
    """Minimal stand-in for the formula-building attributes of SEQuential."""

    def __init__(self, **kwargs):
        defaults = dict(
            method="censoring",
            excused=False,
            time_col="time",
            time_varying_cols=["N", "L", "P"],
            fixed_cols=["sex"],
            indicator_baseline="_bas",
            indicator_squared="_sq",
            trial_include=True,
            followup_include=True,
            weight_preexpansion=True,
            weight_spline=False,
            weight_spline_df=4,
        )
        defaults.update(kwargs)
        for key, value in defaults.items():
            setattr(self, key, value)


def _default_formulas(**kwargs):
    opts = _Opts(**kwargs)
    return [
        _numerator(opts),
        _denominator(opts),
        _cense_numerator(opts),
        _cense_denominator(opts),
    ]


def test_weight_spline_replaces_the_time_quadratics():
    post = _default_formulas(weight_spline=True, weight_preexpansion=False)
    # Post-expansion weight models run on two time axes: both become splines
    assert all('cr(followup, df=4, constraints="center")' in f for f in post)
    assert all('cr(trial, df=4, constraints="center")' in f for f in post)
    assert not any(re.search(r"followup_sq|trial_sq", f) for f in post)

    pre = _default_formulas(weight_spline=True, weight_preexpansion=True)
    # Pre-expansion weight models only see the subject's own time column
    assert all('cr(time, df=4, constraints="center")' in f for f in pre)
    assert not any("time_sq" in f for f in pre)


def test_weight_spline_off_leaves_the_weight_models_quadratic():
    covs = _default_formulas(weight_preexpansion=False)[1]
    assert "cr(" not in covs
    assert {"followup", "followup_sq", "trial", "trial_sq"} <= set(covs.split("+"))


def test_weight_spline_df_is_passed_through():
    covs = _default_formulas(weight_spline=True, weight_spline_df=6)[1]
    assert 'cr(time, df=6, constraints="center")' in covs


def test_weight_spline_df_validation():
    for df in (1, 0, -1):
        with pytest.raises(ValueError, match="weight_spline_df"):
            _model(weight_spline=True, weight_spline_df=df)


def test_weight_model_knots_are_baked_from_the_data_fit_on():
    s = _model(weight_preexpansion=False, weight_spline=True)

    for covs in (s.numerator, s.denominator):
        assert re.search(BAKED.format(var="followup"), covs)
        assert re.search(BAKED.format(var="trial"), covs)
        assert "df=4" not in covs

    # The percentiles patsy would place over the post-expansion data the weight
    # models are fit on, df - 1 = 3 of them once the basis is centred
    followup = s.DT["followup"].to_numpy()
    expected = np.percentile(np.unique(followup), [25, 50, 75]).tolist()
    knots = re.search(r"cr\(followup, knots=\[([^]]*)\]", s.denominator).group(1)
    assert [float(k) for k in knots.split(",")] == pytest.approx(expected)
    assert f"lower_bound={float(followup.min())}" in s.denominator
    assert f"upper_bound={float(followup.max())}" in s.denominator


def test_preexpansion_knots_come_from_the_preexpansion_data():
    s = _model(weight_preexpansion=True, weight_spline=True)
    time = s.data["time"].to_numpy()
    assert f"upper_bound={float(time.max())}" in s.denominator
    # followup tops out one below the pre-expansion time column, so the two
    # would be indistinguishable if the wrong frame were used
    assert float(time.max()) != float(s.DT["followup"].max())


def test_user_supplied_df_form_terms_are_baked():
    s = _model(
        weight_preexpansion=False,
        numerator="sex+cr(followup, df=3)",
        denominator="sex+N+L+P+cr(followup, df=3)",
    )
    # A hand-written term is baked as written — the centering constraint is
    # only added to the ones weight_spline generates
    assert re.search(BAKED_PLAIN.format(var="followup") + r"\)", s.denominator)
    assert "df=3" not in s.denominator


def test_arm_specific_weight_models_each_get_their_knots_baked():
    s = _model(
        weight_preexpansion=False,
        numerator=["sex+cr(followup, df=3)", "sex+cr(followup, df=5)"],
        denominator=["sex+N+L+P+cr(followup, df=3)", "sex+N+L+P+cr(followup, df=5)"],
    )
    assert len(s.denominator) == 2
    assert not any("df=" in covs for covs in s.denominator)
    # unconstrained: df = 3 gives 1 interior knot, df = 5 gives 3
    n_knots = [
        len(re.search(r"knots=\[([^]]*)\]", covs).group(1).split(","))
        for covs in s.denominator
    ]
    assert n_knots == [1, 3]


def test_baked_basis_matches_the_df_form_on_the_full_data():
    # Baking has to reproduce patsy's own knot placement exactly, or the
    # weights would shift purely because the term was rewritten.
    s = _model(weight_preexpansion=False, weight_spline=True)
    DT = s.DT.to_pandas()
    baked = patsy.dmatrix(
        re.search(BAKED.format(var="followup"), s.denominator).group(0), DT
    )
    df_form = patsy.dmatrix('cr(followup, df=4, constraints="center")', DT)
    assert np.allclose(np.asarray(baked), np.asarray(df_form))


def test_weights_are_unchanged_by_the_default_options():
    # weight_spline is opt-in: the default weight models must be untouched
    baseline = _model(weight_preexpansion=False)
    assert "cr(" not in baseline.denominator
    spline = _model(weight_preexpansion=False, weight_spline=True)
    assert not np.allclose(
        baseline.DT["weight"].to_numpy(), spline.DT["weight"].to_numpy()
    )


def test_bootstrap_replicates_share_the_main_fit_basis():
    s = SEQuential(
        load_data("SEQdata"),
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
            weight_preexpansion=False,
            weight_spline=True,
            bootstrap_nboot=3,
            bootstrap_sample=0.8,
            seed=42,
        ),
    )
    s.expand()
    s.bootstrap()
    s.fit()
    baked = s.denominator
    assert "df=" not in baked
    # Refitting (the replicates run through the same body) leaves the basis
    # fixed at the one the main fit chose
    s.fit()
    assert s.denominator == baked


def test_expansion_keeps_the_columns_a_spline_term_wraps():
    # The expansion works out which columns to carry from the model formulas;
    # a cr() term has to contribute its variable, not fragments of the call.
    s = _model(
        weight_preexpansion=False,
        numerator="sex+cr(N_bas, df=3)",
        denominator="sex+N+L+P+cr(N_bas, df=3)",
    )
    assert "N_bas" in s.DT.columns


def test_bake_leaves_terms_it_cannot_bake_alone():
    DT = pl.DataFrame(
        {"followup": range(101), "grp": ["a", "b"] * 50 + ["a"]},
    )
    # Variable absent from the data
    assert _bake_spline_knots("cr(missing, df=4)", DT) == "cr(missing, df=4)"
    # Non-numeric variable
    assert _bake_spline_knots("cr(grp, df=4)", DT) == "cr(grp, df=4)"
    # Already-explicit knots
    explicit = "cr(followup, knots=[1.0, 2.0], lower_bound=0.0, upper_bound=100.0)"
    assert _bake_spline_knots(explicit, DT) == explicit
    # None and per-arm lists pass through in the same shape
    assert _bake_spline_knots(None, DT) is None
    assert _bake_spline_knots(["sex", "cr(followup, df=3)"], DT)[0] == "sex"
    # Several terms in one formula, each baked against its own variable
    baked = _bake_spline_knots("cr(followup, df=3)+cr(followup, df=5)", DT)
    assert baked.count("lower_bound") == 2
    assert "df=" not in baked


def test_bake_rejects_a_term_with_too_few_distinct_values():
    # Every knot lands on the single value, which patsy would reject with an
    # error naming neither the term nor the reason
    DT = pl.DataFrame({"x": [5.0] * 50})
    with pytest.raises(ValueError, match="too few distinct values"):
        _bake_spline_knots("cr(x, df=4)", DT)


@pytest.mark.parametrize("glm_package", ["statsmodels", "glum"])
def test_weight_spline_fits_on_either_glm_backend(glm_package):
    # The basis is centred, which keeps it full rank alongside the model
    # intercept — glum's solver, unlike statsmodels' pinv, needs that
    s = _model(weight_preexpansion=False, weight_spline=True, glm_package=glm_package)
    assert 'constraints="center"' in s.denominator
    assert s.DT["weight"].to_numpy().min() > 0


def test_user_supplied_outcome_spline_is_baked():
    s = SEQuential(
        load_data("SEQdata"),
        id_col="ID",
        time_col="time",
        eligible_col="eligible",
        treatment_col="tx_init",
        outcome_col="outcome",
        time_varying_cols=["N", "L", "P"],
        fixed_cols=["sex"],
        method="ITT",
        parameters=SEQopts(
            covariates=(
                "tx_init_bas+trial+trial_sq+sex+N_bas+L_bas+P_bas"
                '+cr(followup, df=4, constraints="center")'
            ),
            km_curves=True,
            seed=42,
        ),
    )
    s.expand()
    s.fit()
    s.survival()
    assert re.search(BAKED.format(var="followup"), s.covariates)
    assert "df=4" not in s.covariates
    # df = 4 centred contributes 4 columns to the outcome model
    coef_names = list(s.outcome_model[0]["outcome"].params.index)
    assert len([n for n in coef_names if n.startswith("cr(followup,")]) == 4
