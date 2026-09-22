"""SEQopts formula arguments: the ones that are a single formula by
construction are checked where they are supplied, rather than failing further
downstream in the column extraction or the patsy formula build.
"""

import pytest

from pySEQTarget import SEQopts
from pySEQTarget.helpers._col_string import _col_string


@pytest.mark.parametrize("name", ["covariates", "cense_numerator", "cense_denominator"])
def test_single_formula_arguments_reject_a_sequence(name):
    with pytest.raises(TypeError, match=f"{name} must be a single patsy formula"):
        SEQopts(**{name: ["sex", "N"]})


@pytest.mark.parametrize("name", ["numerator", "denominator"])
def test_weight_formulas_reject_an_empty_sequence(name):
    with pytest.raises(ValueError, match="empty sequence"):
        SEQopts(**{name: []})


@pytest.mark.parametrize("name", ["numerator", "denominator"])
def test_weight_formulas_reject_a_non_formula(name):
    with pytest.raises(TypeError, match=f"{name} must be a patsy formula"):
        SEQopts(**{name: 3})


def test_valid_formula_arguments_are_accepted():
    # numerator/denominator may still be one formula per treatment_level,
    # validated against treatment_level in _param_checker
    opts = SEQopts(
        covariates="sex+N",
        numerator=["sex", "sex+N"],
        denominator=["sex+L", "sex+N+L"],
    )
    assert opts.covariates == "sex+N"
    assert opts.numerator == ["sex", "sex+N"]


def test_col_string_reads_through_function_call_terms():
    # The expansion works out which columns to carry from the model formulas,
    # so a term that wraps a column has to contribute the column itself
    assert _col_string(["sex+N_bas+followup+followup_sq"]) == {
        "sex",
        "N_bas",
        "followup",
        "followup_sq",
    }
    assert _col_string(["cr(followup, df=4)+sex"]) == {"followup", "sex"}
    assert _col_string(
        ["cr(followup, knots=[1.0, 2.0], lower_bound=0.0, upper_bound=60.0)"]
    ) == {"followup"}
    assert _col_string(["tx_init_bas*followup"]) == {"tx_init_bas", "followup"}
    # An intercept-only weight model refers to no columns at all
    assert _col_string(["1", "", None]) == set()
    # Per-treatment-level formulas contribute the union of their columns
    assert _col_string([["sex", "sex+N"]]) == {"sex", "N"}
