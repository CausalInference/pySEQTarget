"""excused_colnames[i] excuses a switch away from treatment_level[i]."""

import polars as pl

from pySEQTarget import SEQopts, SEQuential
from pySEQTarget.data import load_data


def test_expansion_excuses_switches_away_from_each_level():
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
            excused=True, excused_colnames=["excusedZero", "excusedOne"]
        ),
    )
    s.expand()
    excused = s.DT.filter(pl.col("isExcused"))
    assert excused.height > 0
    # A switch away from 0 lands on 1 (excusedZero), and away from 1 on 0 (excusedOne)
    away = ((pl.col("excusedZero") == 1) & (pl.col("tx_init") == 1)) | (
        (pl.col("excusedOne") == 1) & (pl.col("tx_init") == 0)
    )
    assert excused.select(away.all()).item()
