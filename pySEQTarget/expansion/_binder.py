import polars as pl

from ._mapper import _mapper


def _binder(self, kept_cols):
    """
    Internal function to bind data to the map created by __mapper
    """
    excluded = {
        "dose",
        f"dose{self.indicator_squared}",
        "followup",
        f"followup{self.indicator_squared}",
        "tx_lag",
        "trial",
        f"trial{self.indicator_squared}",
        self.time_col,
        f"{self.time_col}{self.indicator_squared}",
    }

    cols = kept_cols.union({self.eligible_col, self.outcome_col, self.treatment_col})
    cols = {col for col in cols if col is not None}

    regular = {
        col
        for col in cols
        if not (self.indicator_baseline in col or self.indicator_squared in col)
        and col not in excluded
    }

    baseline = {
        col for col in cols if self.indicator_baseline in col and col not in excluded
    }
    bas_kept = {col.replace(self.indicator_baseline, "") for col in baseline}

    squared = {
        col for col in cols if self.indicator_squared in col and col not in excluded
    }
    sq_kept = {col.replace(self.indicator_squared, "") for col in squared}

    kept = list(regular.union(bas_kept).union(sq_kept))

    if self.selection_first_trial:
        DT = (
            self.data.sort([self.id_col, self.time_col])
            .with_columns(
                [
                    pl.col(self.time_col).alias("period"),
                    pl.col(self.time_col).alias("followup"),
                    pl.lit(0).alias("trial"),
                ]
            )
            .drop(self.time_col)
        )
    else:
        DT = _mapper(
            self.data, self.id_col, self.time_col, self.followup_min, self.followup_max
        )
        DT = DT.join(
            self.data.select([self.id_col, self.time_col] + kept),
            left_on=[self.id_col, "period"],
            right_on=[self.id_col, self.time_col],
            how="left",
        )
    DT = DT.sort([self.id_col, "trial", "followup"]).with_columns(
        [
            (pl.col("trial") ** 2).alias(f"trial{self.indicator_squared}"),
            (pl.col("followup") ** 2).alias(f"followup{self.indicator_squared}"),
        ]
    )

    if squared:
        squares = []
        for sq in squared:
            col = sq.replace(self.indicator_squared, "")
            squares.append((pl.col(col) ** 2).alias(f"{col}{self.indicator_squared}"))
        DT = DT.with_columns(squares)

    baseline_cols = {bas.replace(self.indicator_baseline, "") for bas in baseline}
    needed = {self.eligible_col, self.treatment_col}
    baseline_cols.update({c for c in needed})

    bas = [
        pl.col(c)
        .first()
        .over([self.id_col, "trial"])
        .alias(f"{c}{self.indicator_baseline}")
        for c in baseline_cols
    ]

    DT = (
        DT.with_columns(bas)
        .filter(pl.col(f"{self.eligible_col}{self.indicator_baseline}") == 1)
        .drop([f"{self.eligible_col}{self.indicator_baseline}", self.eligible_col])
    )

    return DT
