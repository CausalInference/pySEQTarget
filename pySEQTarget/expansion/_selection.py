import polars as pl


def _selection_weight(self):
    """Inverse-probability weights for the ID-trials kept by ``selection_random``.

    ``selection_random`` keeps a proportion ``selection_sample`` of the
    ID-trials starting on the first level of ``treatment_level`` and all other
    ID-trials. Selection depends only on baseline treatment, which the outcome
    model conditions on, so the model needs no correction, but standardizing
    over the kept ID-trials under-represents the first arm's covariate
    distribution. Each kept ID-trial of that arm stands for
    ``1 / selection_sample`` ID-trials.

    Returns a polars expression on the observed baseline treatment, or ``None``
    when no selection was made.
    """
    if not self.selection_random:
        return None
    tx_bas = f"{self.treatment_col}{self.indicator_baseline}"
    return (
        pl.when(pl.col(tx_bas) == self.treatment_level[0])
        .then(pl.lit(1 / self.selection_sample))
        .otherwise(pl.lit(1.0))
    )


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
        .sort("trialID")
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
            | (
                pl.col(f"{self.treatment_col}{self.indicator_baseline}")
                != self.treatment_level[0]
            )
        )
        .drop("trialID")
    )
