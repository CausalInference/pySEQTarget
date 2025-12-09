import polars as pl


def _weight_bind(self, WDT):
    if self.weight_preexpansion:
        join = "inner"
        on = [self.id_col, "period"]
        WDT = WDT.rename({self.time_col: "period"})
    else:
        join = "left"
        on = [self.id_col, "trial", "followup"]

    WDT = self.DT.join(WDT, on=on, how=join)

    if self.visit_colname is not None:
        visit = pl.col(self.visit_colname) == 0
    else:
        visit = pl.lit(False)

    if self.weight_preexpansion and self.excused:
        trial = (pl.col("trial") == 0) & (pl.col("period") == 0)
        excused = (
            pl.col("isExcused").fill_null(False).cum_sum().over([self.id_col, "trial"])
            > 0
        )
        override = (
            trial
            | excused
            | visit
            | pl.col(self.outcome_col).is_null()
            | (pl.col("denominator") < 1e-7)
        )
    elif not self.weight_preexpansion and self.excused:
        trial = pl.col("followup") == 0
        excused = (
            pl.col("isExcused").fill_null(False).cum_sum().over([self.id_col, "trial"])
            > 0
        )
        override = (
            trial
            | excused
            | visit
            | pl.col(self.outcome_col).is_null()
            | (pl.col("denominator") < 1e-7)
            | (pl.col("numerator") < 1e-7)
        )
    else:
        trial = (pl.col("trial") == pl.col("trial").min().over(self.id_col)) & (
            pl.col("followup") == 0
        )
        excused = pl.lit(False)
        override = (
            trial
            | excused
            | visit
            | pl.col(self.outcome_col).is_null()
            | (pl.col("denominator") < 1e-15)
            | pl.col("numerator").is_null()
        )

    self.DT = (
        (
            WDT.with_columns(
                pl.when(override)
                .then(pl.lit(1.0))
                .otherwise(pl.col("numerator") / pl.col("denominator"))
                .alias("wt")
            )
            .sort([self.id_col, "trial", "followup"])
            .with_columns(
                pl.col("wt")
                .fill_null(1.0)
                .cum_prod()
                .over([self.id_col, "trial"])
                .alias("weight")
            )
        )
        .with_columns(
            (
                pl.col("weight")
                * pl.col("_cense").fill_null(1.0)
                * pl.col("_visit").fill_null(1.0)
            ).alias("weight")
        )
        .drop(["_cense", "_visit"])
    )
