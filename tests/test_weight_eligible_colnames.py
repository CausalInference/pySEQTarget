import polars as pl

from pySEQTarget import SEQopts, SEQuential
from pySEQTarget.data import load_data


def _nobs_per_arm(weight_eligible_colnames):
    data = load_data("SEQdata")
    # Balanced 0/1 eligibility indicator carried through expansion.
    median_n = data["N"].median()
    data = data.with_columns((pl.col("N") > median_n).cast(pl.Int32).alias("welig"))
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
        parameters=SEQopts(
            weighted=True,
            weight_eligible_colnames=weight_eligible_colnames,
            seed=1,
        ),
    )
    s.expand()
    s.fit()
    return [int(m.nobs) for m in s.denominator_model]


def test_weight_eligible_colnames_restricts_weight_models_to_eligible_rows():
    # Each arm's weight model is fit only on rows where its
    # weight_eligible_colnames indicator == 1 (_get_subset_for_level). With a
    # roughly half-on indicator the per-arm denominator nobs drops below the
    # unfiltered baseline.
    base = _nobs_per_arm(weight_eligible_colnames=[])
    elig = _nobs_per_arm(weight_eligible_colnames=["welig", "welig"])

    assert all(e < b for e, b in zip(elig, base))
