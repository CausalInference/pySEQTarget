"""Regression test: time_varying_cols and fixed_cols are documented Optional.

Constructing without them used to crash in _param_checker (`set(None)`), and
several downstream sites iterate self.fixed_cols directly. Omitting both must
work through the whole pipeline.
"""

from pySEQTarget import SEQopts, SEQuential
from pySEQTarget.data import load_data


def test_pipeline_runs_without_covariate_args():
    s = SEQuential(
        load_data("SEQdata"),
        id_col="ID",
        time_col="time",
        eligible_col="eligible",
        treatment_col="tx_init",
        outcome_col="outcome",
        method="ITT",
        parameters=SEQopts(km_curves=True, seed=42),
    )
    s.expand()
    s.fit()
    s.survival()

    # The auto-built outcome formula contains no covariate terms beyond the
    # treatment/followup/trial defaults.
    assert "sex" not in s.covariates
    assert s.km_data.height > 0
