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
    return s


def _coefs(s):
    return s.outcome_model[0]["outcome"].params.values


def test_weight_p99_truncates_at_p01_p99_percentile_weights():
    # weight_p99=True overrides weight_min/weight_max with the p01/p99 of the
    # (untruncated) weights -- these are reported in weight_stats. So it must be
    # equivalent to passing those percentile values as explicit bounds, and must
    # differ from an untruncated weighted fit.
    p99 = _fit(weight_p99=True)
    ws = p99.weight_stats
    p01_val = float(ws["weight_p01"][0])
    p99_val = float(ws["weight_p99"][0])

    explicit = _fit(weight_min=p01_val, weight_max=p99_val)
    untruncated = _fit()

    assert np.allclose(_coefs(p99), _coefs(explicit), atol=1e-8)
    assert not np.allclose(_coefs(p99), _coefs(untruncated), atol=1e-6)
