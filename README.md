[![PyPI version](https://badge.fury.io/py/pySEQTarget.svg)](https://pypi.org/project/pySEQTarget) [![Downloads](https://static.pepy.tech/badge/pySEQTarget)](https://pepy.tech/project/pySEQTarget)[![codecov](https://codecov.io/gh/CausalInference/pySEQTarget/graph/badge.svg?token=DMOVJJUWXP)](https://codecov.io/gh/CausalInference/pySEQTarget)[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](https://opensource.org/licenses/MIT)![versions](https://img.shields.io/pypi/pyversions/pySEQTarget.svg)
# pySEQTarget - Sequentially Nested Target Trial Emulation

Implementation of sequential trial emulation for the analysis of
observational databases. The `SEQTaRget` software accommodates
time-varying treatments and confounders, as well as binary and failure
time outcomes. `SEQTaRget` allows to compare both static and dynamic
strategies, can be used to estimate observational analogs of
intention-to-treat and per-protocol effects, and can adjust for
potential selection bias.

## Installation
You can install the development version of pySEQTarget from github with:
```shell
pip install git+https://github.com/CausalInference/pySEQTarget
```
Or from pypi iwth 
```shell
pip install pySEQTarget
```

## Setting up your Analysis
The primary API, `SEQuential` uses a dataclass system to handle function input. You can then recover elements as they are built by interacting with the `SEQuential` object you create.

From the user side, this amounts to creating a dataclass, `SEQopts`, and then feeding this into `SEQuential`. If you forgot to add something at class instantiation, you can, in some cases, add them when you call their respective class method.

```python
import polars as pl
from pySEQTarget import SEQuential, SEQopts

data = pl.from_pandas(SEQdata)
options = SEQopts(km_curves = True)

# Initiate the class
model = SEQuential(data, 
                   id_col = "ID",
                   time_col = "time",
                   eligible_col = "eligible",
                   time_varying_cols = ["N", "L", "P"],
                   fixed_cols = ["sex"],
                   method = "ITT",
                   options = options)
model.expand()  # Construct the nested structure
model.bootstrap(bootstrap_nboot = 20) # Run 20 bootstrap samples
model.fit() # Fit the model
model.survival() # Create survival curves
model.plot() # Create and show a plot of the survival curves
model.collect() # Collection of important information

```

## Assumptions
There are several key assumptions in this package -
1. User provided `time_col` begins at 0 per unique `id_col`, we also assume this column contains only integers and continues by 1 for every time step, e.g. (0, 1, 2, 3, 4, ...) is allowed and (0, 1, 2, 2.5, ...) or (0, 1, 4, 5) are not
    1. Provided `time_col` entries may be out of order at intake as a sort is enforced at expansion.
2. `eligible_col` and elements of `excused_colnames` are once 1, only 1 (with respect to `time_col`) flag variables.

