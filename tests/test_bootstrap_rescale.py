from types import SimpleNamespace

import numpy as np
import polars as pl
import pytest
from pytest import approx

from pySEQTarget.analysis import _create_endoffup, _risk_estimates
from pySEQTarget.helpers import _rescale_boot

PROBS = [0.025, 0.975]


def test_rescale_boot_corrects_m_out_of_n_spread():
    # Bootstrap the mean drawing half the sample each time: the raw draws are
    # sqrt(2) times too spread out; the rescaled ones match the full-sample SE
    rng = np.random.default_rng(1)
    x = rng.normal(size=4000)
    draws = np.array([rng.choice(x, 2000, replace=True).mean() for _ in range(2000)])
    rescaled = _rescale_boot(draws, x.mean(), 0.5)
    assert np.std(rescaled, ddof=1) == approx(np.std(x) / np.sqrt(4000), rel=0.05)
    assert np.std(rescaled, ddof=1) == approx(np.std(draws, ddof=1) * np.sqrt(0.5))
    # A bounded quantity stays within its bounds
    bounded = _rescale_boot(np.array([0.0, 1.0]), 0.3, 0.5)
    assert np.all((bounded >= 0) & (bounded <= 1))
    # No-op for the standard bootstrap
    assert _rescale_boot(draws, x.mean(), 1.0) is draws


def test_rescale_boot_polars_expression_keeps_draw_column():
    df = pl.DataFrame({"est": [0.5, 0.5], "draw": [0.3, 0.9]})
    out = df.with_columns(
        _rescale_boot(pl.col("draw"), pl.col("est"), 0.25).alias("draw")
    )
    assert out["draw"].to_list() == approx([0.4, 0.7])
    assert out["est"].to_list() == [0.5, 0.5]


def _shrink(q, est):
    # bootstrap_sample = 0.25 shrinks each draw's distance from the estimate by
    # sqrt(0.25) = 0.5; with an affine map the percentiles shrink the same way
    return est + 0.5 * (q - est)


@pytest.fixture
def draws():
    rng = np.random.default_rng(1)
    return rng.uniform(0.2, 0.4, 200), rng.uniform(0.3, 0.6, 200)


def test_risk_estimates_percentile_ci_rescaled(draws):
    r0, r1 = draws
    boot = lambda r: pl.DataFrame(  # noqa: E731
        {"followup": [10] * 200, "risk": r, "boot_idx": list(range(1, 201))}
    )
    obj = SimpleNamespace(
        km_data=pl.DataFrame(
            {
                "followup": [10, 10],
                "tx_init": [0, 1],
                "pred": [0.3, 0.45],
                "estimate": ["risk", "risk"],
            }
        ),
        risk_times=None,
        subgroup_colname=None,
        bootstrap_nboot=200,
        bootstrap_sample=0.25,
        bootstrap_CI=0.95,
        bootstrap_CI_method="percentile",
        treatment_level=[0, 1],
        treatment_col="tx_init",
        _boot_risks={0: boot(r0), 1: boot(r1)},
    )
    out = _risk_estimates(obj)
    rd = out["risk_difference"].filter(pl.col("A_x") == 1)
    rr = out["risk_ratio"].filter(pl.col("A_x") == 1)
    rd_draws = pl.Series(r1 - r0)
    log_rr_draws = pl.Series(np.log(r1 / r0))
    assert [rd["RD 95% LCI"][0], rd["RD 95% UCI"][0]] == approx(
        [_shrink(rd_draws.quantile(p), 0.15) for p in PROBS]
    )
    assert [rr["RR 95% LCI"][0], rr["RR 95% UCI"][0]] == approx(
        [np.exp(_shrink(log_rr_draws.quantile(p), np.log(1.5))) for p in PROBS]
    )
    assert rd["RD SE"][0] == approx(0.5 * rd_draws.std())
    assert rr["log(RR) SE"][0] == approx(0.5 * log_rr_draws.std())


def test_endoffup_percentile_ci_rescaled(draws):
    r0, r1 = draws

    def eof(e0, e1):
        return {
            "eof": pl.DataFrame(
                {
                    "tx_init_bas": [0, 1],
                    "estimate": [e0, e1],
                    "n": [10, 10],
                    "n_subjects": [10, 10],
                    "n_eligible": [10, 10],
                    "n_censored": [0, 0],
                    "n_nomeasure": [0, 0],
                }
            )
        }

    obj = SimpleNamespace(
        treatment_col="tx_init",
        indicator_baseline="_bas",
        subgroup_colname=None,
        end_of_fup_type="binary",
        end_of_fup_time=5,
        bootstrap_CI=0.95,
        bootstrap_CI_method="percentile",
        bootstrap_sample=0.25,
        outcome_model=[eof(0.3, 0.45)] + [eof(a, b) for a, b in zip(r0, r1)],
    )
    out = _create_endoffup(obj)
    est = out["eof_data"].filter(pl.col("A") == 0)
    assert [est["95% LCI"][0], est["95% UCI"][0]] == approx(
        [_shrink(np.quantile(r0, p), 0.3) for p in PROBS]
    )
    cmp = out["eof_comparison"].filter(pl.col("A_x") == 0)
    assert [cmp["Difference 95% LCI"][0], cmp["Difference 95% UCI"][0]] == approx(
        [_shrink(np.quantile(r1 - r0, p), 0.15) for p in PROBS]
    )
    assert [cmp["Ratio 95% LCI"][0], cmp["Ratio 95% UCI"][0]] == approx(
        [np.exp(_shrink(np.quantile(np.log(r1 / r0), p), np.log(1.5))) for p in PROBS]
    )
