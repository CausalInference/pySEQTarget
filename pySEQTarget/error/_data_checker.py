import polars as pl


def _data_checker(self):
    check = self.data.group_by(self.id_col).agg(
        [pl.len().alias("row_count"), pl.col(self.time_col).max().alias("max_time")]
    )

    invalid = check.filter(pl.col("row_count") != pl.col("max_time") + 1)
    if len(invalid) > 0:
        raise ValueError(
            f"Data validation failed: {len(invalid)} ID(s) have mismatched "
            f"This suggests invalid times"
            f"Invalid IDs:\n{invalid}"
        )

    for col in self.excused_colnames:
        violations = (
            self.data.sort([self.id_col, self.time_col])
            .group_by(self.id_col)
            .agg(
                [
                    (
                        (pl.col(col).cum_sum().shift(1, fill_value=0) > 0)
                        & (pl.col(col) == 0)
                    )
                    .any()
                    .alias("has_violation")
                ]
            )
            .filter(pl.col("has_violation"))
        )

        if len(violations) > 0:
            raise ValueError(
                f"Column '{col}' violates 'once one, always one' rule for excusing treatment "
                f"{len(violations)} ID(s) have zeros after ones."
            )
