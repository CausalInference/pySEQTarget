"""Numeric fixed covariates stay continuous; non-numeric ones are categorical."""

import numpy as np
import pandas as pd
import polars as pl

from pySEQTarget import SEQopts, SEQuential
from pySEQTarget.data import load_data
from pySEQTarget.helpers._fix_categories import _cast_fixed_categories


def _with_age(data):
    rng = np.random.default_rng(1)
    ids = data["ID"].unique().sort()
    ages = pl.DataFrame({"ID": ids, "age": np.round(rng.uniform(20, 80, len(ids)), 1)})
    return data.join(ages, on="ID")


def test_cast_leaves_numeric_fixed_covariates_numeric():
    df = pd.DataFrame(
        {
            "age": [34.5, 61.2, 47.8],
            "sex": [0, 1, 1],
            "region": ["north", "south", "north"],
        }
    )
    out = _cast_fixed_categories(["age", "sex", "region"], df)
    assert out["age"].tolist() == [34.5, 61.2, 47.8]
    assert out["sex"].tolist() == [0, 1, 1]
    assert isinstance(out["region"].dtype, pd.CategoricalDtype)


def test_continuous_fixed_covariate_is_a_single_term():
    s = SEQuential(
        _with_age(load_data("SEQdata")),
        id_col="ID",
        time_col="time",
        eligible_col="eligible",
        treatment_col="tx_init",
        outcome_col="outcome",
        time_varying_cols=["N", "L", "P"],
        fixed_cols=["sex", "age"],
        method="censoring",
        parameters=SEQopts(weighted=True, weight_preexpansion=False, km_curves=True),
    )
    s.expand()
    s.fit()
    s.survival()
    names = list(s.outcome_model[0]["outcome"].params.index)
    assert [n for n in names if n.startswith("age")] == ["age"]
    assert "sex" in names
    assert "age" in s.denominator_model[0].params.index


def test_subgroup_drops_only_the_subgroup_column_from_the_outcome_model():
    s = SEQuential(
        _with_age(load_data("SEQdata")),
        id_col="ID",
        time_col="time",
        eligible_col="eligible",
        treatment_col="tx_init",
        outcome_col="outcome",
        time_varying_cols=["N", "L", "P"],
        fixed_cols=["sex", "age"],
        method="ITT",
        parameters=SEQopts(subgroup_colname="sex"),
    )
    terms = s.covariates.split("+")
    assert "age" in terms
    assert "sex" not in terms
