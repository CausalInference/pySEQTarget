"""Validation of excused_colnames in _data_checker.

_param_checker pads excused_colnames with None up to len(treatment_level);
the data checker used to feed that None into pl.col() and crash with a
confusing TypeError before the analysis even started.
"""

import pytest

from pySEQTarget import SEQopts, SEQuential
from pySEQTarget.data import load_data


def _build(**opts):
    return SEQuential(
        load_data("SEQdata"),
        id_col="ID",
        time_col="time",
        eligible_col="eligible",
        treatment_col="tx_init",
        outcome_col="outcome",
        time_varying_cols=["N", "L", "P"],
        fixed_cols=["sex"],
        method="censoring",
        parameters=SEQopts(**opts),
    )


def test_excused_colnames_shorter_than_treatment_level():
    # One excused column for two treatment levels: the padded None entry must
    # be skipped, not validated.
    s = _build(excused=True, excused_colnames=["excusedZero"])
    assert s.excused_colnames == ["excusedZero", None]


def test_excused_colnames_missing_column_raises_clearly():
    with pytest.raises(ValueError, match="not found in data columns"):
        _build(excused=True, excused_colnames=["nonexistent_col"])
