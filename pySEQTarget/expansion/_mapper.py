import math

import polars as pl


def _mapper(data, id_col, time_col, min_followup=-math.inf, max_followup=math.inf):
    """
    Internal function to create the expanded map to bind data to.
    """

    DT = (
        data.select([pl.col(id_col), pl.col(time_col)])
        .with_columns([pl.col(id_col).cum_count().over(id_col).sub(1).alias("trial")])
        .with_columns(
            [
                pl.int_ranges(
                    pl.col(time_col),
                    pl.col(time_col).max().over(id_col) + 1,
                ).alias("period")
            ]
        )
        .explode("period", empty_as_null=True)
        .drop(pl.col(time_col))
        .with_columns(
            [
                pl.col(id_col)
                .cum_count()
                .over([id_col, "trial"])
                .sub(1)
                .alias("followup")
            ]
        )
        .filter(
            (pl.col("followup") >= min_followup) & (pl.col("followup") <= max_followup)
        )
    )
    return DT
