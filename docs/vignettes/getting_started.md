# Getting Started

Getting started with SEQuential is hopefully quite easy. The primary flow is to define your options through `SEQopts`, and then build and modify the state of the `SEQuential` class. Let's move through a basic tutorial.

## A Simple Analysis

Let's create a motivating example - we are primarily interested in a treatment's effectiveness, based on the initial treatment assignment, and how this differs between `sex` in our fabricated cohort. Assuming we already have the package installed and it is accessible to our python environment, we can dive into building our options:

A full list of options is available in the documentation under {py:class}`~pySEQtarget.SEQopts`

## Setup

```python
from pySEQTarget import SEQopts
my_options = SEQopts(subgroup_colname = "sex",
                     km_curves = True)
```

We don't have too many options available to use as we run an ITT analysis. Except in certain cases, this is an unweighted analysis, which is what many of the options interact with.

## Initializing our primary 'Driver'

Now, we begin our analysis - this amounts to creating and modifying the state of our SEQuential class. Nothing is returned until a call to {py:meth}`~pySEQTarget.SEQuential.collect` is made, which will return all results created to the point of collection. 

```python
from pySEQTarget import SEQuential
from pySEQTarget.data import load_data

# Load sample data 
data = load_data("SEQdata")

# Initialize the class
my_analysis = SEQuential(data,
                         id_col="ID",
                         time_col="time",
                         eligible_col="eligible",
                         treatment_col="tx_init",
                         outcome_col="outcome",
                         time_varying_cols=["N", "L", "P"],
                         fixed_cols=["sex"],
                         method="ITT",
                         parameters=my_options)
```

## Building our analysis

Now that we've initialized our class a few things have happened, our covariates have been created and stored, and our parameters have been checked. If there is no error, we are ready to build our analysis!

### Creating the nested target trial framework

```python
my_analysis.expand()
```

In this code snippet, we access the class method {py:meth}`~pySEQTarget.SEQuential.expand` which builds our target trial framework. This internally creates a `DT` attribute (our expanded data).

### Fitting our model

```python
my_analysis.fit()
```

Since this is a relatively simple model, we can immediately move to fitting out model. Like most other python packages, this is done by calling {py:meth}`~pySEQTarget.SEQuential.fit`. This again doesn't return anything, but will add the outcome model to our internal class state.
At this point there are results to collect, so we could inspect them; however, let's save that for after building our survival curves and risk data.

### 'Predicting' from our Model
Canonically in Python, we usually call a `predict` method. `SEQuential` handles this internally, and instead of the usual `predict`, survival, risk, and incidence rates are derived from {py:meth}`~pySEQTarget.SEQuential.survival`. Again at this point we could collect our results and have the majority of our results; however, `SEQuential` will also plot our data for us.

```python
my_analysis.survival()
my_analysis.plot()
```

### Collecting our results

Now that we've reached the end of our analysis, we can call {py:meth}`~pySEQTarget.SEQuential.collect`. To note, we can always call collect at any step of the way if you want to collect any results and check them as they are being built, but you can also do this by accessing the internal state of the class. The collection here, formally, sends all results currently made into an output class {py:class}`~pySEQTarget.SEQoutput` which has some handy tools for accessing results.

```python
my_output = my_analysis.collect()
```
Now that we have created an object with our output class, the most immediate way to recover results is to dump everything to markdown or pdf using {py:meth}`~pySEQTarget.SEQoutput.to_md` or {py:meth}`~pySEQTarget.SEQoutput.to_pdf` respectively.

```python
my_output.to_md()
```
