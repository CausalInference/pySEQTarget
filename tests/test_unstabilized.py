"""Unstabilized IP weights: weight_stabilized=False fixes the treatment-weight
numerator at 1, so the weights are cumulative products of 1/denominator."""

import numpy as np
import polars as pl
import pytest

from pySEQTarget import SEQopts, SEQuential
from pySEQTarget.data import load_data

KEY = ["ID", "trial", "followup"]


def _run(data="SEQdata", method="censoring", fit=True, **kwargs):
    s = SEQuential(
        load_data(data),
        id_col="ID",
        time_col="time",
        eligible_col="eligible",
        treatment_col="tx_init",
        outcome_col="outcome",
        time_varying_cols=["N", "L", "P"],
        fixed_cols=["sex"],
        method=method,
        parameters=SEQopts(**kwargs),
    )
    if fit:
        s.expand()
        s.fit()
    return s


def _merge(stab, unstab):
    return stab.DT.select(KEY + ["numerator", "denominator"]).join(
        unstab.DT.select(KEY + ["numerator", "denominator"]),
        on=KEY,
        suffix="_unstab",
    )


@pytest.mark.filterwarnings("ignore")
def test_unstabilized_weights_use_the_same_denominators():
    opts = dict(weighted=True, denominator="N+L+P+sex", seed=1636)
    stab = _run(numerator="sex", **opts)
    unstab = _run(weight_stabilized=False, **opts)

    merged = _merge(stab, unstab)
    assert (merged["numerator_unstab"] == 1.0).all()
    # Same denominator model, so identical denominators; only the numerator differs
    np.testing.assert_allclose(merged["denominator"], merged["denominator_unstab"])

    # Weights are the running product of 1/denominator within each trial
    DT = unstab.DT.sort(KEY)
    wt = DT["wt"].to_numpy()
    inv_denom = 1.0 / DT["denominator"].to_numpy()
    assert np.all(np.isclose(wt, 1.0) | np.isclose(wt, inv_denom))
    cum = DT.select(pl.col("wt").cum_prod().over(["ID", "trial"]))["wt"]
    np.testing.assert_allclose(DT["weight"], cum)

    # No numerator model is fit, and the reported numerator formula is "1"
    assert unstab.numerator == "1"
    assert all(m is None for m in unstab.numerator_model)
    assert unstab.collect().summary("numerator") == []


def test_unstabilized_warns_where_it_cannot_apply():
    with pytest.warns(UserWarning, match="has no effect"):
        _run(method="ITT", fit=False, weight_stabilized=False)
    with pytest.warns(UserWarning, match="has no effect"):
        _run(fit=False, weight_stabilized=False)


def test_unstabilized_warns_when_numerator_supplied():
    with pytest.warns(UserWarning, match="'numerator' is ignored"):
        s = _run(
            fit=False,
            weighted=True,
            weight_stabilized=False,
            numerator="sex",
            denominator="N+L+P+sex",
        )
    assert s.numerator == "1"


def test_weight_stabilized_must_be_bool():
    with pytest.raises(TypeError, match="weight_stabilized"):
        SEQopts(weight_stabilized="no")


@pytest.mark.filterwarnings("ignore")
@pytest.mark.parametrize(
    "data, extra",
    [
        (
            "SEQdata",
            dict(excused=True, excused_colnames=["excusedZero", "excusedOne"]),
        ),
        ("SEQdata_multitreatment", dict(treatment_level=[0, 1, 2])),
    ],
)
def test_unstabilized_keeps_denominators_excused_and_multinomial(data, extra):
    opts = dict(weighted=True, weight_preexpansion=False, seed=1636, **extra)
    stab = _run(data, **opts)
    unstab = _run(data, weight_stabilized=False, **opts)

    merged = _merge(stab, unstab)
    assert (merged["numerator_unstab"] == 1.0).all()
    np.testing.assert_allclose(merged["denominator"], merged["denominator_unstab"])
