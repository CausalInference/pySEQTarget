from pySEQTarget import SEQopts, SEQuential
from pySEQTarget.data import load_data


def _coef_names(**opts):
    s = SEQuential(
        load_data("SEQdata"),
        id_col="ID",
        time_col="time",
        eligible_col="eligible",
        treatment_col="tx_init",
        outcome_col="outcome",
        time_varying_cols=["N", "L", "P"],
        fixed_cols=["sex"],
        method="ITT",
        parameters=SEQopts(**opts),
    )
    s.expand()
    s.fit()
    return set(s.outcome_model[0]["outcome"].params.index)


def test_followup_include_and_trial_include_add_or_drop_outcome_terms():
    # These flags control whether the follow-up and trial terms (and their
    # squares) enter the outcome-model formula, so the effect is visible in the
    # fitted coefficient names.
    both = _coef_names()
    no_fup = _coef_names(followup_include=False)
    no_trial = _coef_names(trial_include=False)

    fup_terms = {"followup", "followup_sq"}
    trial_terms = {"trial", "trial_sq"}

    # Baseline: all four terms present
    assert fup_terms <= both
    assert trial_terms <= both

    # followup_include=False drops the follow-up terms but keeps the trial terms
    assert fup_terms.isdisjoint(no_fup)
    assert trial_terms <= no_fup

    # trial_include=False drops the trial terms but keeps the follow-up terms
    assert trial_terms.isdisjoint(no_trial)
    assert fup_terms <= no_trial
