import polars as pl


def _weight_stats(self):
    stats = self.DT.select(
        [
            pl.col("weight").min().alias("weight_min"),
            pl.col("weight").max().alias("weight_max"),
            pl.col("weight").mean().alias("weight_mean"),
            pl.col("weight").std().alias("weight_std"),
            pl.col("weight").quantile(0.01).alias("weight_p01"),
            pl.col("weight").quantile(0.25).alias("weight_p25"),
            pl.col("weight").quantile(0.50).alias("weight_p50"),
            pl.col("weight").quantile(0.75).alias("weight_p75"),
            pl.col("weight").quantile(0.99).alias("weight_p99"),
        ]
    )

    if self.weight_p99:
        self.weight_min = stats.select("weight_p01").item()
        self.weight_max = stats.select("weight_p99").item()

    return stats
