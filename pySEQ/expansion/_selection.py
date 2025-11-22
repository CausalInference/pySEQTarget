import polars as pl
def _random_selection(self):
    """
    Handles the case where random selection is applied for data from 
    the __mapper -> __binder -> optionally __dynamic pipeline
    """
    UIDs = self.DT.select([
        self.id_col, 
        "trial", 
        f"{self.treatment_col}{self.indicator_baseline}"]) \
    .with_columns(
        (pl.col(self.id_col) + "_" + pl.col("trial")).alias("trialID")) \
    .filter(
        pl.col(f"{self.treatment_col}{self.indicator_baseline}") == 0) \
    .unique("trialID").to_series().to_list()
    
    NIDs = len(UIDs)
    sample = self._rng.choice(
        UIDs, 
        size=int(self.selection_probability * NIDs), 
        replace=False
    )
    
    self.DT = self.DT.with_columns(
        (pl.col(self.id_col) + "_" + pl.col("trial")).alias("trialID")
    ).filter(
        pl.col("trialID").is_in(sample)
    ).drop("trialID")
    