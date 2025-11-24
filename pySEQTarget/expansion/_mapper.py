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
                pl.struct(
                    [
                        pl.col(time_col),
                        pl.col(time_col).max().over(id_col).alias("max_time"),
                    ]
                )
                .map_elements(
                    lambda x: list(range(x[time_col], x["max_time"] + 1)),
                    return_dtype=pl.List(pl.Int64),
                )
                .alias("period")
            ]
        )
        .explode("period")
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
