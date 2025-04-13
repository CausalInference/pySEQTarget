import polars as pl
def __dynamic(DT, id_col, time_col, treatment_col, method,
              excused_col0, excused_col1,
              squared_indicator, baseline_indicator):
    """
    Handles special cases for the data from the __mapper -> __binder pipeline
    """
    if method == "dose-response":
        DT = DT.with_columns(
            pl.col(treatment_col).cum_count().over([id_col, "trial"]).alias("dose")
        ).with_columns([
            (pl.col("dose") ** 2).alias(f"dose{squared_indicator}")
        ])
    elif method == "censoring":
        DT = DT.with_columns(
            pl.col(treatment_col)
            .first()
            .over([id_col, "trial"])
            .alias("temp")
        ).with_columns(
            pl.col(treatment_col)
            .shift(1)
            .over([id_col, "trial"])
            .alias("tx_lag")
        ).with_columns(
            pl.when(pl.col("temp").is_null())
            .then(pl.col("temp"))
            .otherwise(pl.col("tx_lag"))
            .alias("tx_lag")
        ).with_columns(
            (pl.col(treatment_col) != pl.col("tx_lag")).alias("switch")
        )