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
    bootstrap_nboot = 20,       # 20 bootstrap iterations
    cense_colname = "LTFU",      # control for losses-to-followup as a censor
    excused = True,             # allow excused treatment swapping
    excused_colnames = ["excusedZero", "excusedOne"],
    km_curves = True,           # run survival estimates
    selection_random = True,    #  randomly sample treatment non-initiators
    selection_sample = 0.30,     # sample 30% of treatment non-initiators
    weighted = True,            # enables the weighting
    weight_lag_condition=False, # turn off lag condition when weighting for adherance
    weight_p99 = True,          # bounds weights by the 1st and 99th percentile
    weight_preexpansion = False # weights are predicted using post-expansion data as a stabilizer
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

```python
my_analysis.fit()
my_analysis.survival()
my_analysis.plot()

my_output = my_analysis.collect()
my_output.to_md()
```

## That's it?

Yes! There are very few differences between the code for more straightforward and more difficult analyses using this package. Our hope is that through utilizing almost only the SEQopts to work with your analysis, that this is a streamlined process that is also easy to manipulate.
