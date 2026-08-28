import polars as pl


def _check_binary(data, col):
    unique_vals = set(data[col].drop_nulls().unique().to_list())
    if not unique_vals.issubset({0, 1}):
        # Cap the listing — a continuous column can hold thousands of values
        offending = sorted(unique_vals - {0, 1})
        shown = ", ".join(str(v) for v in offending[:10])
        if len(offending) > 10:
            shown += f", ... ({len(offending)} distinct values)"
        raise ValueError(
            f"Column '{col}' must be binary (0/1) but contains values: {shown}"
        )


def _data_checker(self):
    _check_binary(self.data, self.eligible_col)

    # end_of_fup treats a null outcome as "not measured at this time"; in every
    # other mode the outcome must be complete.
    if not self.end_of_fup and self.data[self.outcome_col].null_count() > 0:
        raise ValueError(
            f"Column '{self.outcome_col}' contains missing values; missing "
            "outcome measurements are only permitted with end_of_fup=True."
        )
    # Continuous end-of-follow-up outcomes are averaged, not modelled, so
    # non-binary values are expected there.
    if not (self.end_of_fup and self.end_of_fup_type == "continuous"):
        try:
            _check_binary(self.data, self.outcome_col)
        except ValueError as e:
            if self.end_of_fup:
                raise ValueError(
                    f"{e} For an outcome that is not 0/1, set "
                    "end_of_fup_type='continuous' in SEQopts."
                ) from None
            raise

    if self.cense_eligible_colname is not None:
        _check_binary(self.data, self.cense_eligible_colname)

    for col in self.weight_eligible_colnames:
        if col is not None:
            if col not in self.data.columns:
                raise ValueError(
                    f"weight_eligible_colnames entry '{col}' not found in data columns."
                )
            _check_binary(self.data, col)

    check = self.data.group_by(self.id_col).agg(
        [pl.len().alias("row_count"), pl.col(self.time_col).max().alias("max_time")]
    )

    invalid = check.filter(pl.col("row_count") != pl.col("max_time") + 1)
    if len(invalid) > 0:
        raise ValueError(
            f"Data validation failed: {len(invalid)} ID(s) have mismatched row counts. "
            f"This suggests invalid times. "
            f"Invalid IDs:\n{invalid}"
        )

    for col in self.excused_colnames:
        # _param_checker pads the list with None up to len(treatment_level)
        # when fewer excused columns are supplied.
        if col is None:
            continue
        if col not in self.data.columns:
            raise ValueError(
                f"excused_colnames entry '{col}' not found in data columns."
            )
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
                f"Column '{col}' violates the 'once one, always one' rule: "
                f"{len(violations)} ID(s) have zeros after ones."
            )
