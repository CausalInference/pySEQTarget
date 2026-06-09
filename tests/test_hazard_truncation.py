"""Regression tests for the survival-time reduction in the hazard g-formula.

`_truncate_to_first_event` collapses the simulated counterfactual grid (outcomes
drawn independently at every follow-up row) to one survival row per (id, trial):
the first-event row, or the max-follow-up row when there is no event.

The earlier implementation used the inclusive `cum_sum(outcome) <= 1` then
`.last()`, which kept post-event rows and returned the final follow-up row,
silently recording single events as censored. That dropped ~99% of simulated
events and inflated the marginal-HR variance ~8x relative to SEQTaRget (R).
"""

import polars as pl

from pySEQTarget.analysis._hazard import _truncate_to_first_event


def _grid(rows):
    # rows: list of (id, trial, [outcome per follow-up 0..T])
    recs = []
    for uid, trial, outs in rows:
        for f, o in enumerate(outs):
            recs.append((uid, trial, f, o))
    return pl.DataFrame(
        recs, schema=["id", "trial", "followup", "outcome"], orient="row"
    )


def test_first_event_row_is_kept_for_each_pattern():
    grid = _grid(
        [
            (1, 0, [0, 0, 1, 0, 0]),  # single interior event -> (followup=2, event=1)
            (2, 0, [0, 0, 0, 0, 0]),  # no event             -> (followup=4, event=0)
            (3, 0, [0, 1, 0, 1, 0]),  # two events; first    -> (followup=1, event=1)
            (4, 0, [1, 0, 0, 0, 0]),  # event at time 0       -> (followup=0, event=1)
            (5, 0, [0, 0, 0, 0, 1]),  # event at last row     -> (followup=4, event=1)
        ]
    )

    out = (
        _truncate_to_first_event(grid, "id", "outcome")
        .sort("id")
        .select(["id", "followup", "outcome"])
    )

    assert out.to_dicts() == [
        {"id": 1, "followup": 2, "outcome": 1},
        {"id": 2, "followup": 4, "outcome": 0},
        {"id": 3, "followup": 1, "outcome": 1},
        {"id": 4, "followup": 0, "outcome": 1},
        {"id": 5, "followup": 4, "outcome": 1},
    ]


def test_no_events_are_dropped():
    # Every unit that has >=1 simulated outcome must end up with event=1; only the
    # all-zero unit (id=2) is censored. This is the property the old idiom broke.
    grid = _grid(
        [
            (1, 0, [0, 0, 1, 0, 0]),
            (2, 0, [0, 0, 0, 0, 0]),
            (3, 0, [0, 1, 0, 1, 0]),
            (4, 0, [1, 0, 0, 0, 0]),
            (5, 0, [0, 0, 0, 0, 1]),
        ]
    )
    out = _truncate_to_first_event(grid, "id", "outcome")
    true_units_with_event = (
        grid.group_by("id").agg(pl.col("outcome").max().alias("ever"))["ever"].sum()
    )
    assert out["outcome"].sum() == true_units_with_event == 4


def test_grouping_is_per_id_and_trial():
    # Same id, two trials with different first-event times must be reduced
    # independently.
    grid = _grid(
        [
            (1, 0, [0, 0, 1, 0]),  # trial 0: event at 2
            (1, 1, [1, 0, 0, 0]),  # trial 1: event at 0
        ]
    )
    out = (
        _truncate_to_first_event(grid, "id", "outcome")
        .sort(["id", "trial"])
        .select(["id", "trial", "followup", "outcome"])
    )
    assert out.to_dicts() == [
        {"id": 1, "trial": 0, "followup": 2, "outcome": 1},
        {"id": 1, "trial": 1, "followup": 0, "outcome": 1},
    ]


def test_beats_the_old_buggy_idiom():
    # Lock the regression: the previous `cum_sum <= 1` then `.last()` loses the
    # single interior events that the fixed helper retains.
    grid = _grid([(uid, 0, [0, 0, 1, 0, 0]) for uid in range(1, 11)])

    fixed = _truncate_to_first_event(grid, "id", "outcome")["outcome"].sum()

    old = (
        grid.with_columns(pl.col("outcome").cum_sum().over(["id", "trial"]).alias("cs"))
        .filter(pl.col("cs") <= 1)
        .group_by(["id", "trial"])
        .last()["outcome"]
        .sum()
    )

    assert fixed == 10  # every unit's single event retained
    assert old == 0  # old idiom dropped all of them
    assert fixed > old
