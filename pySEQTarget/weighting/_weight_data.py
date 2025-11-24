import polars as pl


def _weight_setup(self):
    DT = self.DT
    data = self.data
    if not self.weight_preexpansion:
        baseline_lag = (
            data.select([self.treatment_col, self.id_col, self.time_col])
            .sort([self.id_col, self.time_col])
            .with_columns(
                pl.col(self.treatment_col)
                .shift(fill_value=self.treatment_level[0])
                .over(self.id_col)
                .alias("tx_lag")
            )
            .drop(self.treatment_col)
            .rename({self.time_col: "period"})
        )

        fup0 = DT.filter(pl.col("followup") == 0).join(
            baseline_lag, on=[self.id_col, "period"], how="inner"
        )

        fup = (
            DT.sort([self.id_col, "trial", "followup"])
            .with_columns(
                pl.col(self.treatment_col)
                .shift(fill_value=self.treatment_level[0])
                .over([self.id_col, "trial"])
                .alias("tx_lag")
            )
            .filter(pl.col("followup") != 0)
        )

        WDT = pl.concat([fup0, fup]).sort([self.id_col, "trial", "followup"])
    else:
        WDT = data.with_columns(
            pl.col(self.treatment_col)
            .shift(fill_value=self.treatment_level[0])
            .over(self.id_col)
            .alias("tx_lag"),
            (pl.col(self.time_col) ** 2).alias(
                f"{self.time_col}{self.indicator_squared}"
            ),
        )
    return WDT
