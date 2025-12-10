import polars as pl


def _random_selection(self):
    """
    Handles the case where random selection is applied for data from
    the __mapper -> __binder -> optionally __dynamic pipeline
    """
    UIDs = (
        self.DT.select(
            [self.id_col, "trial", f"{self.treatment_col}{self.indicator_baseline}"]
        )
        .with_columns(
            (
                pl.col(self.id_col).cast(pl.Utf8) + "_" + pl.col("trial").cast(pl.Utf8)
            ).alias("trialID")
        )
        .filter(
            pl.col(f"{self.treatment_col}{self.indicator_baseline}")
            == self.treatment_level[0]
        )
        .unique("trialID")
        .get_column("trialID")
        .to_list()
    )

    NIDs = len(UIDs)
    sample = self._rng.choice(
        UIDs, size=int(self.selection_sample * NIDs), replace=False
    )

    self.DT = (
        self.DT.with_columns(
            (
                pl.col(self.id_col).cast(pl.Utf8) + "_" + pl.col("trial").cast(pl.Utf8)
            ).alias("trialID")
        )
        .filter(
            pl.col("trialID").is_in(sample)
            | pl.col(f"{self.treatment_col}{self.indicator_baseline}")
            != self.treatment_level[0]
        )
        .drop("trialID")
    )
