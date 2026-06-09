import warnings

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
        method=opts.pop("method", "censoring"),
        parameters=SEQopts(**opts),
    )


def test_warns_when_numerator_and_denominator_are_identical():
    # Identical num/denom -> stabilized weights all 1 -> usually a typo.
    formula = "sex"
    with pytest.warns(UserWarning, match="identical"):
        _build(weighted=True, numerator=formula, denominator=formula)


def test_no_warning_when_numerator_and_denominator_differ():
    with warnings.catch_warnings():
        warnings.simplefilter("error", UserWarning)
        _build(weighted=True, numerator="sex", denominator="sex+N+L+P")


def test_no_warning_under_ITT_even_if_identical():
    # ITT doesn't fit treatment-weight models, so the warning is gated on method.
    with warnings.catch_warnings():
        warnings.simplefilter("error", UserWarning)
        # weighted ITT requires LTFU/visit; use LTFU dataset
        SEQuential(
            load_data("SEQdata_LTFU"),
            id_col="ID",
            time_col="time",
            eligible_col="eligible",
            treatment_col="tx_init",
            outcome_col="outcome",
            time_varying_cols=["N", "L", "P"],
            fixed_cols=["sex"],
            method="ITT",
            parameters=SEQopts(
                weighted=True,
                cense_colname="LTFU",
                numerator="sex",
                denominator="sex",
            ),
        )


def test_no_warning_when_not_weighted():
    with warnings.catch_warnings():
        warnings.simplefilter("error", UserWarning)
        _build(weighted=False, numerator="sex", denominator="sex")
