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

    return stats


def _weight_bounds(self):
    """Truncation bounds for the current fit; with weight_p99, its 1st/99th percentiles."""
    stats = getattr(self, "weight_stats", None)
    if self.weight_p99 and stats is not None:
        return float(stats["weight_p01"][0]), float(stats["weight_p99"][0])
    return self.weight_min, self.weight_max
