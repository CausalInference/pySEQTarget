"""Transformed terms such as np.log(x), and terms undefined for some rows."""

import polars as pl
import pytest

from pySEQTarget import SEQopts, SEQuential
from pySEQTarget.data import load_data
from pySEQTarget.helpers._col_string import _col_string

UNDEFINED = r"Model term\(s\) np.log\(x\) produced missing or infinite values"


def _data():
    # x is positive for N above its median and negative otherwise
    return (
        load_data("SEQdata")
        .with_columns(
            (pl.col("N") > pl.col("N").median()).cast(pl.Int8).alias("welig"),
            (pl.col("N") - pl.col("N").min() + 1).alias("pos"),
        )
        .with_columns(
            pl.when(pl.col("welig") == 1).then(pl.col("pos")).otherwise(-1.0).alias("x")
        )
    )


def _model(**opts):
    s = SEQuential(
        _data(),
        id_col="ID",
        time_col="time",
        eligible_col="eligible",
        treatment_col="tx_init",
        outcome_col="outcome",
        time_varying_cols=["x", "pos", "N", "L", "P"],
        fixed_cols=["sex"],
        method="censoring",
        parameters=SEQopts(**opts),
    )
    s.expand()
    return s


def _weights(term, **opts):
    return dict(
        weighted=True,
        weight_preexpansion=False,
        numerator=f"sex+followup+trial+{term}",
        denominator=f"sex+N+L+P+followup+trial+{term}",
        **opts,
    )


def test_column_reader_skips_the_module_in_a_call():
    assert _col_string(["followup+np.log(pos)+np.sqrt(N)"]) == {"followup", "pos", "N"}


@pytest.mark.filterwarnings("ignore")
def test_log_terms_fit_end_to_end():
    s = _model(covariates="tx_init_bas+followup+np.log(pos)", km_curves=True)
    s.fit()
    s.survival()
    assert "np.log(pos)" in s.outcome_model[0]["outcome"].params.index
    s = _model(**_weights("np.log(pos)"))
    s.fit()
    assert "np.log(pos)" in s.denominator_model[0].params.index


def test_undefined_outcome_term_is_named():
    s = _model(covariates="tx_init_bas+followup+np.log(x)")
    with pytest.raises(ValueError, match=UNDEFINED):
        s.fit()


def test_undefined_weight_term_is_named():
    s = _model(**_weights("np.log(x)"))
    with pytest.raises(ValueError, match=UNDEFINED):
        s.fit()


@pytest.mark.filterwarnings("ignore")
def test_term_undefined_only_where_predicted_is_named():
    # Fitted on eligible rows, where x > 0, but predicted on every row
    s = _model(**_weights("np.log(x)", weight_eligible_colnames=["welig", "welig"]))
    with pytest.raises(ValueError, match=UNDEFINED):
        s.fit()
