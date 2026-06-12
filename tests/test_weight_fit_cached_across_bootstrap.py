"""With weight_preexpansion=True the weight models are fit on the un-resampled
pre-expansion data, so bootstrap replicates would refit bit-identical models
and re-predict identical weights every iteration. The main fit's predicted
weight frame is cached and replicates only redo the join onto their resampled
DT — results must be unchanged, the weight fitters must run exactly once.
"""

import sys

import numpy as np

from pySEQTarget import SEQopts, SEQuential
from pySEQTarget.data import load_data


def _run(monkeypatch=None, disable_cache=False):
    seq_mod = sys.modules["pySEQTarget.SEQuential"]
    fit_calls = []

    real_fit_denominator = seq_mod._fit_denominator

    def spy_fit_denominator(self, WDT):
        fit_calls.append(getattr(self, "_current_boot_idx", None))
        return real_fit_denominator(self, WDT)

    if monkeypatch is not None:
        monkeypatch.setattr(seq_mod, "_fit_denominator", spy_fit_denominator)
        if disable_cache:
            real_bind = seq_mod._weight_bind

            def bind_no_cache(self, WDT):
                result = real_bind(self, WDT)
                # Drop the cache after every bind so each replicate refits.
                self._main_weight_WDT = None
                return result

            monkeypatch.setattr(seq_mod, "_weight_bind", bind_no_cache)

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
            weight_preexpansion=True,
            bootstrap_nboot=3,
            seed=42,
        ),
    )
    s.expand()
    s.bootstrap()
    s.fit()
    coefs = np.concatenate([np.asarray(m["outcome"].params) for m in s.outcome_model])
    return fit_calls, coefs


def test_weight_models_fit_once_across_bootstrap(monkeypatch):
    fit_calls, _ = _run(monkeypatch)
    assert fit_calls == [None]  # main fit only, no replicate refits


def test_cached_weights_match_refit_weights(monkeypatch):
    _, cached = _run(monkeypatch)
    fit_calls, refit = _run(monkeypatch, disable_cache=True)
    assert len(fit_calls) == 4  # cache disabled: main + 3 replicate refits
    assert np.array_equal(cached, refit)  # bit-identical outcome coefficients
