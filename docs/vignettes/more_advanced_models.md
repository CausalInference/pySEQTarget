# More Advanced Analysis
In getting started, we covered some of the basics for getting up and running on a simple analysis, but there are many options stored within `SEQuential`, or more aptly, many more parameters to play with in {py:class}`~pySEQTarget.SEQopts`. Let's cover a more in-depth analysis.

In this case, let's go over a censoring analysis with excused conditions and stabilized weighting, limiting weights to the 99th percentile, and adjusting for losses-to-followup. Futhermore, we are interested in bootstrapping our results to get a risk estimate with confidence bounds and for ease of computation, we are going to randomly downsample 30% of trials which did not initiate treatment. Because we are downsampling, we are additionally going to turn off the lag condition for our adherance weights.

If you are coming from the R version, many arguments have been streamlined or inferred - take R's `bootstrap`, and `bootstrap.nboot` - these have been merged such that any `bootstrap_nboot` over 0 automatically starts the bootstrap initiation.

## Setting up our analysis

In similar fashion to our process in getting started, we begin by setting up our SEQopts

```python
from pySEQTarget import SEQopts
from pySEQTarget.data import load_data

data = load_data("SEQdata_LTFU")
my_options = SEQopts(
    bootstrap_nboot = 20,       # 20 bootstrap iterations (for demonstration only — use 500+ in practice)
    cense_colname = "LTFU",      # control for losses-to-followup as a censor
    excused = True,             # allow excused treatment swapping
    excused_colnames = ["excusedZero", "excusedOne"],
    km_curves = True,           # run survival estimates
    selection_random = True,    #  randomly sample treatment non-initiators
    selection_sample = 0.30,     # sample 30% of treatment non-initiators
    weighted = True,            # enables the weighting
    weight_lag_condition=False, # turn off lag condition when weighting for adherance
    weight_p99 = True,          # bounds weights by the 1st and 99th percentile
    weight_preexpansion = True  # weights are predicted using pre-expansion data
)
```

## Running our Analysis

Now that we have our setup, it is time to repeat the analytical pipeline. From here on, not much differs.

```python
from pySEQTarget import SEQuential

my_analysis = SEQuential(data,
                         id_col="ID",
                         time_col="time",
                         eligible_col="eligible",
                         treatment_col="tx_init",
                         outcome_col="outcome",
                         time_varying_cols=["N", "L", "P"],
                         fixed_cols=["sex"],
                         method="censoring",
                         parameters=my_options)

# Expand the data
my_analysis.expand()
```

### A quick note about bootstrapping

The key difference, when bootstrapping, is that you will additionally have to call {py:meth}`~pySEQTarget.SEQuential.bootstrap`. This initializes the underlying randomization with replacement. Note that if you've forgotten to enable bootstrapping initially in your `SEQopts` you can do this here as well.

```python
my_analysis.bootstrap()
```

## Back to our analysis

Now that the underlying bootstrap structure has been in place, we can simply continue as we would in simpler models- fit, survival, plot, collect, and dump.

You can recover all results through the collection, and interact with them as you would normally in Python; however, to dump to md or pdf you may require the optional depedencies which interact with these file formats.

These can be easily installed through the `output` optional dependencies with pySEQTarget.

```shell
pip install pySEQTarget[output]
```

```python
my_analysis.fit()
my_analysis.survival()
my_analysis.plot()

my_output = my_analysis.collect()
# Requires tabulate installation
my_output.to_md()
```

## Modelling Followup with a Natural Cubic Spline

By default, followup time enters the outcome model as a linear and quadratic term (`followup` and `followup_sq`). If you prefer a more flexible non-linear representation, you can replace these with a natural cubic spline basis by setting `followup_spline=True`. Because the spline replaces the standard followup terms, you must also set `followup_include=False` to avoid conflicting specifications.

```python
my_options = SEQopts(
    followup_spline=True,
    followup_include=False,
    km_curves=True,
)
```

The spline basis is constructed using patsy's `cr()` function. Knot positions are computed once from the full expanded dataset before any bootstrap resampling, and are held fixed across all bootstrap iterations and the survival prediction grid. This ensures the spline basis is identical at fit time and prediction time regardless of how the bootstrap sample happens to be distributed.

The degrees of freedom for the spline are controlled by `followup_spline_df`, which defaults to `4` (matching the default in the R package SEQTaRget). Increasing this allows a more flexible curve; decreasing it towards the minimum of `2` produces a stiffer fit.

```python
my_options = SEQopts(
    followup_spline=True,
    followup_include=False,
    followup_spline_df=6,   # more flexible spline
    km_curves=True,
)
```

The rest of the analytical pipeline is unchanged — `expand()`, `fit()`, `survival()`, and `collect()` all work exactly as before.

## Modelling the Baseline Hazard in the Weight Models with a Natural Cubic Spline

The same choice arises in the weight models. By default they are quadratic in time: `followup`, `followup_sq`, `trial` and `trial_sq` when `weight_preexpansion=False`, and the time column and its square when `weight_preexpansion=True`. A quadratic only lets the hazard of treatment rise or flatten off, which can fit poorly when the cumulative incidence of treatment moves through several phases over followup.

Setting `weight_spline=True` replaces each of those quadratics with a natural cubic spline basis (`cr()`) of `weight_spline_df` degrees of freedom, letting the baseline hazard take a flexible shape over time. This applies to the treatment weight models and, when `cense_colname` or `visit_colname` is set, to those models too.

```python
my_options = SEQopts(
    km_curves=True,
    weighted=True,
    weight_preexpansion=False,
    weight_spline=True,     # model time in the weight models as a natural cubic spline
    weight_spline_df=4,     # 4 degrees of freedom, i.e. 3 interior knots at percentiles
)
```

The default is unchanged (`weight_spline=False`), so existing analyses are unaffected.

For finer control — a spline in `followup` but not `trial`, a different number of knots per term, or a spline in a time-varying confounder — write the `cr()` terms into `numerator` and `denominator` yourself. Any `cr(x, df=N)` term in a model formula, wherever it came from, has its knots fixed from the full data the model is fit on before fitting, so every bootstrap resample builds the same basis as the main fit.

Pass `constraints="center"`, as the generated terms do. An unconstrained `cr()` basis spans the constant function, so it is collinear with the model intercept by exactly one dimension; statsmodels absorbs that with a pseudo-inverse (after reporting a convergence warning), but the `glum` backend cannot fit it at all.

```python
my_options = SEQopts(
    km_curves=True,
    weighted=True,
    weight_preexpansion=False,
    numerator='sex+N_bas+L_bas+P_bas+trial+trial_sq+cr(followup, df=5, constraints="center")',
    denominator='sex+N+L+P+N_bas+L_bas+P_bas+trial+trial_sq+cr(followup, df=5, constraints="center")',
)
```

After `fit()`, the `numerator` and `denominator` attributes show the formulas with their knots written out.

## That's it?

Yes! There are very few differences between the code for more straightforward and more difficult analyses using this package. Our hope is that through utilizing almost only the SEQopts to work with your analysis, that this is a streamlined process that is also easy to manipulate.
