import polars as pl

def _dynamic(self):
    """
    Handles special cases for the data from the __mapper -> __binder pipeline
    """
    if self.method == "dose-response":
        DT = self.DT.with_columns(
            pl.col(self.treatment_col)
            .cum_sum()
            .over([self.id_col, "trial"])
            .alias("dose")
        ).with_columns([
            (pl.col("dose") ** 2)
            .alias(f"dose{self.indicator_squared}")
        ])
        self.DT = DT
        
    elif self.method == "censoring":
        DT = self.DT.sort([self.id_col, "trial", "followup"]).with_columns(
            pl.col(self.treatment_col)
            .shift(1)
            .over([self.id_col, "trial"])
            .alias("tx_lag")
        )
        
        switch = (
            pl.when(pl.col("followup") == 0)
            .then(pl.lit(False))
            .otherwise(
                (pl.col("tx_lag").is_not_null()) & 
                (pl.col("tx_lag") != pl.col(self.treatment_col))
            )
        )
        is_excused = pl.lit(False)
        if self.excused:
            conditions = []
            for i in range(len(self.treatment_level)):
                colname = self.excused_colnames[i]
                if colname is not None:
                    conditions.append(
                        (pl.col(colname) == 1) & 
                        (pl.col(self.treatment_col) == self.treatment_level[i])
                    )
            
            if conditions:
                excused = pl.any_horizontal(conditions)
                is_excused = switch & excused
                switch = (
                    pl.when(excused)
                    .then(pl.lit(False))
                    .otherwise(switch) 
                )
        
        DT = DT.with_columns([
            switch.alias("switch"),
            is_excused.alias("isExcused")
        ]).sort([self.id_col, "trial", "followup"]) \
            .filter(
                (
                    pl.col("switch")
                    .cum_max()
                    .shift(1, fill_value=False)
                )
                .over([self.id_col, "trial"])
                == 0
        ).with_columns(
            pl.col("switch")
            .cast(pl.Int8)
            .alias("switch")
        )
        
        self.DT = DT.drop(["tx_lag"])