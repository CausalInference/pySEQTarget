"""Post-expansion excused weights use P(uncensored) in every arm."""

import polars as pl
import pytest

from pySEQTarget import SEQopts, SEQuential
from pySEQTarget.data import load_data


@pytest.mark.filterwarnings("ignore")
def test_uncensored_rows_get_the_probability_of_staying_uncensored():
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
            excused=True,
            excused_colnames=["excusedZero", "excusedOne"],
        ),
    )
    s.expand()
    s.fit()
    # Uncensored rows on their baseline arm, which that arm's models predict
    own_col = (
        pl.when(pl.col("tx_init_bas") == 0)
        .then(pl.col("excusedZero"))
        .otherwise(pl.col("excusedOne"))
    )
    rows = s.DT.filter(
        (pl.col("switch") == 0)
        & (pl.col("followup") > 0)
        & (pl.col("tx_init") == pl.col("tx_init_bas"))
        & (own_col == 0)
    )
    medians = rows.group_by("tx_init_bas").agg(
        pl.col("numerator").median(), pl.col("denominator").median()
    )
    # Switching is rare, so P(uncensored) is near 1 in both arms
    assert medians.height == 2
    assert (medians["numerator"] > 0.5).all()
    assert (medians["denominator"] > 0.5).all()
