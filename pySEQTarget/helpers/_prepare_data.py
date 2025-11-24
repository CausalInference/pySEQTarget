import polars as pl


def _prepare_data(self, DT):
    binaries = [
        self.eligible_col,
        self.outcome_col,
        self.cense_colname,
    ]  # self.excused_colnames + self.weight_eligible_colnames
    binary_colnames = [col for col in binaries if col in DT.columns and not None]

    DT = DT.with_columns(
        [
            *[pl.col(col).cast(pl.Categorical) for col in self.fixed_cols],
            *[pl.col(col).cast(pl.Int8) for col in binary_colnames],
            pl.col(self.id_col).cast(pl.Utf8),
        ]
    )
    return DT
