from types import SimpleNamespace

import numpy as np
import pandas as pd

from pySEQTarget.weighting._weight_fit import (_fit_denominator,
                                               _fit_numerator, _fit_pair)


def _mock_self(**overrides):
    attrs = {
        "weight_preexpansion": False,
        "weight_stabilized": True,
        "excused": False,
        "method": "censoring",
        "treatment_col": "tx",
        "indicator_baseline": "_bas",
        "numerator": "1",
        "denominator": "1",
        "treatment_level": [0, 1],
        "weight_fit_method": "newton",
        "weight_lag_condition": True,
        "weight_eligible_colnames": [None, None],
        "excused_colnames": [None, None],
        "cense_colname": "ltfu",
    }
    attrs.update(overrides)
    return SimpleNamespace(**attrs)


def _make_data(n=60):
    """Both treatment levels have variation in the predictor."""
    block = np.array([0] * (n // 2) + [1] * (n // 2))
    return pd.concat(
        [
            pd.DataFrame(
                {
                    "tx": block,
                    "tx_lag": np.zeros(n, int),
                    "followup": np.arange(1, n + 1),
                }
            ),
            pd.DataFrame(
                {
                    "tx": block,
                    "tx_lag": np.ones(n, int),
                    "followup": np.arange(1, n + 1),
                }
            ),
        ],
        ignore_index=True,
    )


def _make_data_no_variation_level0(n=60):
    """Level-0 subset (tx_lag=0) has all tx=0; level-1 subset has mixed tx."""
    block = np.array([0] * (n // 2) + [1] * (n // 2))
    return pd.concat(
        [
            pd.DataFrame(
                {
                    "tx": np.zeros(n, int),
                    "tx_lag": np.zeros(n, int),
                    "followup": np.arange(1, n + 1),
                }
            ),
            pd.DataFrame(
                {
                    "tx": block,
                    "tx_lag": np.ones(n, int),
                    "followup": np.arange(1, n + 1),
                }
            ),
        ],
        ignore_index=True,
    )


# ── _fit_numerator ────────────────────────────────────────────────────────────


def test_fit_numerator_stores_none_when_no_variation():
    obj = _mock_self()
    _fit_numerator(obj, _make_data_no_variation_level0())
    assert obj.numerator_model[0] is None
    assert obj.numerator_model[1] is not None


def test_fit_numerator_fits_when_variation_exists():
    obj = _mock_self()
    _fit_numerator(obj, _make_data())
    assert obj.numerator_model[0] is not None
    assert obj.numerator_model[1] is not None


# ── _fit_denominator ──────────────────────────────────────────────────────────


def test_fit_denominator_stores_none_when_no_variation():
    obj = _mock_self()
    _fit_denominator(obj, _make_data_no_variation_level0())
    assert obj.denominator_model[0] is None
    assert obj.denominator_model[1] is not None


def test_fit_denominator_fits_when_variation_exists():
    obj = _mock_self()
    _fit_denominator(obj, _make_data())
    assert obj.denominator_model[0] is not None
    assert obj.denominator_model[1] is not None


# ── _fit_pair (cense / visit models) ─────────────────────────────────────────


def test_fit_pair_stores_none_when_no_variation():
    n = 60
    df = pd.DataFrame({"ltfu": np.zeros(n, int), "followup": np.arange(n)})
    obj = _mock_self()
    _fit_pair(
        obj,
        df,
        "cense_colname",
        ["1", "1"],
        ["cense_numerator_model", "cense_denominator_model"],
    )
    assert obj.cense_numerator_model is None
    assert obj.cense_denominator_model is None


def test_fit_pair_fits_when_variation_exists():
    n = 60
    df = pd.DataFrame(
        {
            "ltfu": np.array([0] * (n // 2) + [1] * (n // 2)),
            "followup": np.arange(n),
        }
    )
    obj = _mock_self()
    _fit_pair(
        obj,
        df,
        "cense_colname",
        ["1", "1"],
        ["cense_numerator_model", "cense_denominator_model"],
    )
    assert obj.cense_numerator_model is not None
    assert obj.cense_denominator_model is not None
