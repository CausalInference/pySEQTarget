import polars as pl

class SEQ:
    def SEQexpand(self):
        """
        Creates the expanded dataset for use in Sequential target trial emulation.
        The logical flow of this method is:
        1. Intake the data and ensure that all IDs have at least 1 point of eligibility
        2. Create a shell table of expanded data
            a. e.g. If ID contains time 0:10, new rows are replicated as the concatenation of 0:10, 1:10, 2:10, etc.
        3. For time varying columns, join data on ID and period
        4. For baseline columns, join data on the ID and trial
        5. Work on special cases dependent on the causal contrast
            a. ITT: no additional data processing required
            b. dose-response: 'dose' and 'dose_sq' columns created as the cumulative sum of the treatment column
            c. censoring: dependent on excused condition
                i. non-excused: if a switch (movement on/off a treatment) is observed, the ID-trial is censored at that point
                ii. excused: if a switch is observed, but excused, it is not censored 
        """
        eligible_ids = (
            self.data.filter(pl.col(self.eligible_col) == 1)
            .groupby(self.id_col)
            .agg(pl.sum(self.eligible_col).alias("sum_elig"))
            .filter(pl.col("sum_elig") != 0)[self.id_col]
        )

        data = self.data.filter(pl.col(self.id_col).is_in(eligible_ids))

        # Data expansion based on time
        data = (
            data.filter(pl.col(self.eligible_col) == 1)
            .with_columns(
                pl.col(self.time_col).alias("period"),
                pl.col(self.time_col).range_rows(1).alias("followup"),
            )
            .filter(pl.col("followup") <= self.options.get("max_followup", 100))
        )

        # Handle time-varying variables
        if self.time_varying_cols:
            data_time = (
                data.join(
                    self.data,
                    on=[self.id_col, "period"],
                    how="left",
                )
                .drop(self.eligible_col)
                .with_columns(
                    *[
                        (pl.col(col) ** 2).alias(f"{col}_sq")
                        for col in self.time_varying_cols
                    ]
                )
            )

        if self.method == "dose-response":
            data = data.with_columns(
                pl.col(self.treatment_col).cumsum().alias("dose"),
                (pl.col("dose") ** 2).alias("dose_sq"),
                (pl.col("trial") ** 2).alias("trial_sq"),
            )

        elif self.method == "censoring":
            data = data.with_columns(
                (pl.col(self.treatment_col).diff() != 0).alias("switch")
            )
            if self.options.get("excused"):
                data = data.with_columns(
                    pl.when(pl.col("switch") & (pl.col(self.treatment_col) == 0))
                    .then(pl.when(pl.col(self.options["excused_col1"]) == 1).then(1).otherwise(0))
                    .otherwise(None)
                    .alias("isExcused")
                ).with_columns(
                    pl.col("isExcused").cumsum().over([self.id_col, "trial"]).alias("excused_tmp")
                )
                data = data.filter(pl.col("excused_tmp") == 0)

        return data
