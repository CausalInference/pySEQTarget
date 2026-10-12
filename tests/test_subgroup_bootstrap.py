"""Subgroup bootstrap models are matched by subgroup value, not position."""

from collections import Counter

import polars as pl
import pytest

from pySEQTarget import SEQopts, SEQuential
from pySEQTarget.data import load_data


@pytest.mark.parametrize("estimate", ["survival", "hazard"])
def test_replicate_missing_a_subgroup_is_left_out_of_it(estimate):
    data = load_data("SEQdata")
    opts = dict(subgroup_colname="sex", bootstrap_nboot=3, seed=42)
    opts["km_curves" if estimate == "survival" else "hazard_estimate"] = True
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
        parameters=SEQopts(**opts),
    )
    s.expand()
    s.bootstrap()
    # The last replicate resamples only sex == 1, so it has no sex == 0 rows
    ids = data.filter(pl.col("sex") == 1)["ID"].unique().to_list()
    s._boot_samples[-1] = Counter(ids)
    s.fit()

    assert s._unique_subgroups == [0, 1]
    assert set(s.outcome_model[-1]) == {1}
    with pytest.warns(UserWarning, match="1 bootstrap replicate.*sex = 0"):
        getattr(s, estimate)()
    out = s.km_data if estimate == "survival" else s.hazard_ratio
    assert set(out["sex"].unique()) == {0, 1}
