from pySEQTarget import SEQopts, SEQuential
from pySEQTarget.data import load_data


def _expand(**opts):
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
        parameters=SEQopts(expand_only=True, **opts),
    )
    s.expand()
    return s.DT


def test_followup_min_max_restrict_the_expanded_followup_range():
    # Expansion filters rows to followup in [followup_min, followup_max]. With
    # expand_only=True the DT is returned without any fit step, so the clamp is
    # directly visible.
    full = _expand()
    lim = _expand(followup_min=3, followup_max=10)

    # Unrestricted expansion genuinely extends past the requested window
    assert full["followup"].min() < 3
    assert full["followup"].max() > 10
    # Restricted expansion is clamped to exactly [3, 10] and has fewer rows
    assert lim["followup"].min() == 3
    assert lim["followup"].max() == 10
    assert lim.height < full.height
