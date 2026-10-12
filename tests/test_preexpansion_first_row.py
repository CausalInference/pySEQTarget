"""Each subject's first row is left out of the pre-expansion treatment models."""

from pySEQTarget import SEQopts, SEQuential
from pySEQTarget.data import load_data


def _model():
    data = load_data("SEQdata")
    s = SEQuential(
        data,
        id_col="ID",
        time_col="time",
        eligible_col="eligible",
        treatment_col="tx_init",
        outcome_col="outcome",
        time_varying_cols=["N", "L", "P"],
        fixed_cols=["sex"],
        method="censoring",
        parameters=SEQopts(weighted=True),
    )
    s.expand()
    s.fit()
    return s, data


def test_first_rows_are_not_fitted():
    s, data = _model()
    n_ids = data["ID"].n_unique()
    fitted = sum(int(m.nobs) for m in s.denominator_model)
    assert fitted == data.height - n_ids
