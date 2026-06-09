import numpy as np

from pySEQTarget import SEQopts, SEQuential
from pySEQTarget.data import load_data


def _fit(**opts):
    opts.setdefault("seed", 1)
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
        parameters=SEQopts(weighted=True, **opts),
    )
    s.expand()
    s.fit()
    return s.outcome_model[0]["outcome"].params.values


def test_weight_min_max_truncate_the_weights_used_in_the_outcome_fit():
    # Truncation is applied to the weight vector passed to the outcome GLM (in
    # _outcome_fit.py via pl.col(weight_col).clip(weight_min, weight_max)). It
    # doesn't change self.DT or weight_stats, so we check it via the fitted
    # coefficients. SEQdata weights span ~0.5-2, so a band entirely above that
    # range collapses every weight to the lower bound. A GLM is invariant to a
    # uniform scaling of its weights, so two all-constant clamps must give
    # identical coefficients, while a genuinely varying-weight fit must differ.
    varying = _fit()
    clamp3 = _fit(weight_min=3, weight_max=4)
    clamp10 = _fit(weight_min=10, weight_max=11)

    # Both clamps collapse weights to a constant => identical fit (scale-invariant)
    assert np.allclose(clamp3, clamp10, atol=1e-6)
    # Clamping away the real weight variation changes the fit
    assert not np.allclose(clamp3, varying, atol=1e-6)
