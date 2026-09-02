# End-of-Follow-up Outcomes

## What this is for

The rest of pySEQTarget estimates survival outcomes: a binary event that may occur at any point during follow-up, summarised through risks, survival curves or a hazard ratio.

An end-of-follow-up outcome is different. It is measured once, at a single follow-up time chosen by the user — a biomarker at 12 months, disease status at two years, a questionnaire score at the end of the trial. There is no time-to-event to model, and the quantity of interest is simply the average outcome in each treatment arm.

The {py:class}`~pySEQTarget.SEQopts` option `end_of_fup=True` switches to that estimand. For each trial-period the outcome is read at the requested follow-up time and averaged within each baseline treatment arm, weighted by the period-trial-specific weight at the time the measurement was taken. For a binary outcome this is the weighted proportion in each arm; for a continuous outcome, the weighted mean.

Because there is no outcome model, `end_of_fup` cannot be combined with `km_curves`, `hazard_estimate`, `compevent_colname`, or the dose-response method.

## A minimal example

`end_of_fup_time` is the follow-up time `k` at which the outcome is evaluated, counted in follow-up periods since trial enrollment (not calendar time). After {py:meth}`~pySEQTarget.SEQuential.fit`, the estimates are assembled with {py:meth}`~pySEQTarget.SEQuential.end_of_followup` and land in the `eof_data` and `eof_comparison` attributes.

```python
from pySEQTarget import SEQopts, SEQuential
from pySEQTarget.data import load_data

options = SEQopts(end_of_fup=True,
                  # evaluate the outcome 12 follow-up periods after enrollment
                  end_of_fup_time=12,
                  # "binary" reports a proportion, "continuous" a mean
                  end_of_fup_type="binary",
                  bootstrap_nboot=20,
                  # fixes the bootstrap resamples, so the intervals below are
                  # reproducible
                  seed=1636)

model = SEQuential(load_data("SEQdata"),
                   id_col="ID",
                   time_col="time",
                   eligible_col="eligible",
                   treatment_col="tx_init",
                   outcome_col="outcome",
                   time_varying_cols=["N", "L", "P"],
                   fixed_cols=["sex"],
                   method="ITT",
                   parameters=options)
model.expand()
model.bootstrap()
model.fit()
model.end_of_followup()

model.eof_data.select(["A", "Proportion", "SE", "95% LCI", "95% UCI",
                       "% Censored", "Subjects"])
```

```text
┌─────┬────────────┬──────────┬──────────┬──────────┬────────────┬──────────┐
│ A   ┆ Proportion ┆ SE       ┆ 95% LCI  ┆ 95% UCI  ┆ % Censored ┆ Subjects │
╞═════╪════════════╪══════════╪══════════╪══════════╪════════════╪══════════╡
│ 0   ┆ 0.019865   ┆ 0.003014 ┆ 0.013956 ┆ 0.025773 ┆ 19.687712  ┆ 266      │
│ 1   ┆ 0.027256   ┆ 0.002342 ┆ 0.022666 ┆ 0.031847 ┆ 28.46412   ┆ 236      │
└─────┴────────────┴──────────┴──────────┴──────────┴────────────┴──────────┘
```

`eof_data` gives the weighted proportion (or mean) in each arm with its bootstrap confidence interval, and an account of how much of the arm it rests on. `Trial-periods (Eligible)` is every trial-period that reached the follow-up time, and the next three partition it: `(Analysed)` contribute to the estimate, `(Censored)` were measured at some point but not within the window, and `(No measurement)` were never measured at all — so the three sum back to the eligible total. `% Censored` gives the censored share of that total. `Subjects` counts the distinct people behind the analysed trial-periods; one subject contributes several trial-periods and can be analysed in some and censored in others.

```python
model.eof_comparison.select(["A_x", "A_y", "Difference", "Difference 95% LCI",
                             "Difference 95% UCI", "Difference SE"])
```

```text
┌─────┬─────┬────────────┬────────────────────┬────────────────────┬───────────────┐
│ A_x ┆ A_y ┆ Difference ┆ Difference 95% LCI ┆ Difference 95% UCI ┆ Difference SE │
╞═════╪═════╪════════════╪════════════════════╪════════════════════╪═══════════════╡
│ 0   ┆ 1   ┆ 0.007392   ┆ -0.000471          ┆ 0.015255           ┆ 0.004012      │
│ 1   ┆ 0   ┆ -0.007392  ┆ -0.015255          ┆ 0.000471           ┆ 0.004012      │
└─────┴─────┴────────────┴────────────────────┴────────────────────┴───────────────┘
```

`eof_comparison` gives the pairwise between-arm contrast: the difference in proportions here, or the difference in means for a continuous outcome, with its standard error and confidence interval. For a binary outcome the ratio of proportions is reported alongside it, with an interval computed on the log scale and a `log(Ratio) SE` for inverse-variance pooling. Contrasts are paired by bootstrap iteration, so their intervals account for the correlation between arms. Both directions of each arm pair are reported, so the row you want is the one whose `A_x` is your reference arm.

## Missing measurements and the time window

Outcomes measured at particular visits are rarely available for everyone at exactly time `k`. Encode "not measured at this time" as a missing value (null) in the outcome column — `end_of_fup` is the one mode that accepts missing outcomes, precisely because missingness is meaningful here.

`end_of_fup_window` sets the half-width of a window used when a trial-period has no measurement at exactly `k`:

```python
options = SEQopts(end_of_fup=True,
                  end_of_fup_time=12,
                  # accept a measurement anywhere in [9, 15] when there is none at 12
                  end_of_fup_window=3,
                  bootstrap_nboot=20,
                  seed=1636)

windowed = SEQuential(load_data("SEQdata"),
                      id_col="ID", time_col="time", eligible_col="eligible",
                      treatment_col="tx_init", outcome_col="outcome",
                      time_varying_cols=["N", "L", "P"], fixed_cols=["sex"],
                      method="ITT", parameters=options)
windowed.expand()
windowed.bootstrap()
windowed.fit()
windowed.end_of_followup()

windowed.eof_data.select(["A", "Proportion", "95% LCI", "95% UCI",
                          "Trial-periods (Eligible)", "Trial-periods (Analysed)",
                          "Trial-periods (Censored)", "% Censored"])
```

```text
┌─────┬────────────┬──────────┬──────────┬──────────────┬──────────────┬──────────────┬────────────┐
│ A   ┆ Proportion ┆ 95% LCI  ┆ 95% UCI  ┆ Trial-period ┆ Trial-period ┆ Trial-period ┆ % Censored │
│     ┆            ┆          ┆          ┆ s (Eligible) ┆ s (Analysed) ┆ s (Censored) ┆            │
╞═════╪════════════╪══════════╪══════════╪══════════════╪══════════════╪══════════════╪════════════╡
│ 0   ┆ 0.075704   ┆ 0.058602 ┆ 0.092805 ┆ 2946         ┆ 2523         ┆ 423          ┆ 14.358452  │
│ 1   ┆ 0.102475   ┆ 0.0864   ┆ 0.11855  ┆ 6257         ┆ 4889         ┆ 1368         ┆ 21.863513  │
└─────┴────────────┴──────────┴──────────┴──────────────┴──────────────┴──────────────┴────────────┘
```

```python
windowed.eof_comparison.select(["A_x", "A_y", "Difference", "Ratio",
                                "Ratio 95% LCI", "Ratio 95% UCI", "log(Ratio) SE"])
```

```text
┌─────┬─────┬────────────┬──────────┬───────────────┬───────────────┬───────────────┐
│ A_x ┆ A_y ┆ Difference ┆ Ratio    ┆ Ratio 95% LCI ┆ Ratio 95% UCI ┆ log(Ratio) SE │
╞═════╪═════╪════════════╪══════════╪═══════════════╪═══════════════╪═══════════════╡
│ 0   ┆ 1   ┆ 0.026771   ┆ 1.353635 ┆ 1.024306      ┆ 1.788848      ┆ 0.142237      │
│ 1   ┆ 0   ┆ -0.026771  ┆ 0.738752 ┆ 0.559019      ┆ 0.976271      ┆ 0.142237      │
└─────┴─────┴────────────┴──────────┴───────────────┴───────────────┴───────────────┘
```

The selection rule, applied to each trial-period independently, is:

1. If there is a measurement at exactly `k`, use it.
2. Otherwise, use the measurement *nearest* to `k` within `[k - window, k + window]`. Where two measurements are equally far either side of `k`, the *later* one is taken, so that at least `k` of follow-up has elapsed.
3. If there is no measurement anywhere in the window, the trial-period is *censored* — it contributes nothing to the average.

The weight used is always the weight at the time the chosen measurement was taken, not the weight at `k`.

A window is not free. Widening it recovers trial-periods that would otherwise be dropped, but the measurements it recovers are taken further from the time you actually care about, and the trial-periods it recovers are not a random subset — a trial-period with no measurement at `k` is often one whose follow-up ended early. Treat the window as a trade-off between precision and how literally the estimate answers "the outcome at time `k`", and check how much of the estimate rests on it using the accounting table below.

## Checking what contributed

The diagnostics report where every trial-period went. `nonunique_eof` counts trial-periods and `unique_eof` counts distinct subjects:

```python
windowed.diagnostics["nonunique_eof"]
```

```text
┌─────────────┬──────────┬──────┬───────────┬───────────────────┬───────────────────────────┐
│ tx_init_bas ┆ Eligible ┆ At k ┆ In window ┆ Excluded (outside ┆ Excluded (no measurement) │
│             ┆          ┆      ┆           ┆ window)           ┆                           │
╞═════════════╪══════════╪══════╪═══════════╪═══════════════════╪═══════════════════════════╡
│ 0           ┆ 2946     ┆ 2366 ┆ 157       ┆ 423               ┆ 0                         │
│ 1           ┆ 6257     ┆ 4476 ┆ 413       ┆ 1368              ┆ 0                         │
└─────────────┴──────────┴──────┴───────────┴───────────────────┴───────────────────────────┘
```

The four categories are mutually exclusive, so the trial-period counts partition `Eligible`:

- *At k* — contributed, using a measurement at exactly `k`.
- *In window* — contributed, having fallen back to the window.
- *Excluded (outside window)* — measured at some point, but not within the window.
- *Excluded (no measurement)* — never measured at any follow-up time. Under `method="censoring"` this also picks up trial-periods artificially censored before any measurement was taken.

`At k` plus `In window` is exactly the number of trial-periods behind the estimate, so the two tables always reconcile. The subject counts in `unique_eof` need *not* sum to `Eligible`, because one subject can fall into different categories for different trials.

If `In window` is large relative to `At k`, or `Excluded` dominates, the estimate is resting on much less — or much more indirect — data than the arm totals alone suggest.

These tables are also available from {py:meth}`~pySEQTarget.SEQoutput.retrieve_data` after {py:meth}`~pySEQTarget.SEQuential.collect`, as `"unique_eof"`, `"nonunique_eof"`, `"eof_data"` and `"eof_comparison"`.

## Continuous outcomes

Set `end_of_fup_type="continuous"` for an outcome that is not 0/1. The estimate becomes a weighted mean, reported in a `Mean` column rather than `Proportion`, and its confidence interval is not clamped to `[0, 1]`. The between-arm contrast is the difference in means; no ratio is reported, since a continuous outcome need not be bounded away from zero.

```python
import numpy as np
import polars as pl

rng = np.random.default_rng(42)
data = load_data("SEQdata")
data = data.with_columns(
    (10 + 2 * pl.col("tx_init") + pl.col("N")
     + pl.Series(rng.standard_normal(data.height))).alias("biomarker")
)

continuous = SEQuential(data,
                        id_col="ID", time_col="time", eligible_col="eligible",
                        treatment_col="tx_init", outcome_col="biomarker",
                        time_varying_cols=["N", "L", "P"], fixed_cols=["sex"],
                        method="ITT",
                        parameters=SEQopts(end_of_fup=True,
                                           end_of_fup_time=12,
                                           end_of_fup_type="continuous",
                                           end_of_fup_window=3,
                                           bootstrap_nboot=20,
                                           seed=1636))
continuous.expand()
continuous.bootstrap()
continuous.fit()
continuous.end_of_followup()

continuous.eof_data.select(["A", "Mean", "SE", "95% LCI", "95% UCI"])
```

```text
┌─────┬───────────┬──────────┬───────────┬───────────┐
│ A   ┆ Mean      ┆ SE       ┆ 95% LCI   ┆ 95% UCI   │
╞═════╪═══════════╪══════════╪═══════════╪═══════════╡
│ 0   ┆ 21.703792 ┆ 0.105694 ┆ 21.496635 ┆ 21.910949 │
│ 1   ┆ 21.767593 ┆ 0.091838 ┆ 21.587594 ┆ 21.947591 │
└─────┴───────────┴──────────┴───────────┴───────────┘
```

```python
continuous.eof_comparison
```

```text
┌──────┬─────┬─────┬────────────┬────────────────────┬────────────────────┬───────────────┐
│ Time ┆ A_x ┆ A_y ┆ Difference ┆ Difference 95% LCI ┆ Difference 95% UCI ┆ Difference SE │
╞══════╪═════╪═════╪════════════╪════════════════════╪════════════════════╪═══════════════╡
│ 12.0 ┆ 0   ┆ 1   ┆ 0.063801   ┆ -0.16125           ┆ 0.288851           ┆ 0.114824      │
│ 12.0 ┆ 1   ┆ 0   ┆ -0.063801  ┆ -0.288851          ┆ 0.16125            ┆ 0.114824      │
└──────┴─────┴─────┴────────────┴────────────────────┴────────────────────┴───────────────┘
```

Here `Difference` is the difference in means, and there is no `Ratio` column.

Note that the usual outcome diagnostic tables count `outcome == 1` rows, which has no meaning for a continuous outcome, so they are omitted. In their place the diagnostics report the N, mean and SD of the raw analysed measurements per arm in `eof_summary`; the follow-up and end-of-follow-up tables remain available.

```python
continuous.diagnostics["eof_summary"]
```

```text
┌─────┬──────┬───────────┬──────────┐
│ A   ┆ N    ┆ Mean      ┆ SD       │
╞═════╪══════╪═══════════╪══════════╡
│ 0   ┆ 2523 ┆ 21.703792 ┆ 5.067746 │
│ 1   ┆ 4889 ┆ 21.767593 ┆ 5.180225 │
└─────┴──────┴───────────┴──────────┘
```

## Per-protocol effects

`end_of_fup` composes with weighting in the usual way, so a per-protocol end-of-follow-up effect is the censoring method with `weighted=True`:

```python
perprotocol = SEQuential(load_data("SEQdata"),
                         id_col="ID", time_col="time", eligible_col="eligible",
                         treatment_col="tx_init", outcome_col="outcome",
                         time_varying_cols=["N", "L", "P"], fixed_cols=["sex"],
                         method="censoring",
                         parameters=SEQopts(weighted=True,
                                            numerator="sex",
                                            denominator="N + L + P + sex",
                                            end_of_fup=True,
                                            end_of_fup_time=12,
                                            end_of_fup_window=3,
                                            bootstrap_nboot=20,
                                            seed=1636))
perprotocol.expand()
perprotocol.bootstrap()
perprotocol.fit()
perprotocol.end_of_followup()

perprotocol.eof_data.select(["A", "Proportion", "95% LCI", "95% UCI",
                             "% Censored", "Subjects"])
```

```text
┌─────┬────────────┬──────────┬──────────┬────────────┬──────────┐
│ A   ┆ Proportion ┆ 95% LCI  ┆ 95% UCI  ┆ % Censored ┆ Subjects │
╞═════╪════════════╪══════════╪══════════╪════════════╪══════════╡
│ 0   ┆ 0.045069   ┆ 0.011689 ┆ 0.07845  ┆ 82.959946  ┆ 86       │
│ 1   ┆ 0.088733   ┆ 0.065572 ┆ 0.111894 ┆ 49.592456  ┆ 211      │
└─────┴────────────┴──────────┴──────────┴────────────┴──────────┘
```

Subjects who deviate from their assigned strategy are artificially censored at the point of deviation, and their outcome is missing from then on. A trial-period that deviates before `k` therefore has no measurement to contribute and is excluded rather than carried forward — it appears under `Excluded (outside window)` or `Excluded (no measurement)` in the accounting table, depending on what it had measured earlier. That is why `% Censored` is so much larger here than under ITT.
