"""Regression test: bootstrap weights with weight_preexpansion=True.

_weight_bind joins the pre-expansion weight frame (un-resampled original IDs)
onto the bootstrap-resampled DT. It must do so WITHOUT collapsing the resampled
IDs back to originals: the weight cum_prod groups on (id, trial), and merging
the replicate copies of a multiply-sampled subject into one group interleaves
their rows — turning weights a, ab into a, a², a²b, a²b²… (each copy compounds
the other's). Every replicate copy duplicates the same source rows, so the
correct cumulative weights are identical across copies.
"""

import sys

import polars as pl

from pySEQTarget import SEQopts, SEQuential
from pySEQTarget.data import load_data


def test_boot_weights_identical_across_replicate_copies(monkeypatch):
    # The package __init__ re-exports shadow the submodule names, so patch the
    # name inside the SEQuential module via sys.modules.
    seq_mod = sys.modules["pySEQTarget.SEQuential"]
    wb_mod = sys.modules["pySEQTarget.weighting._weight_bind"]

    captured = []
    orig = wb_mod._weight_bind

    def spy(self, WDT):
        result = orig(self, WDT)
        if getattr(self, "_current_boot_idx", None) is not None:
            captured.append(self.DT)
        return result

    monkeypatch.setattr(seq_mod, "_weight_bind", spy)

    s = SEQuential(
        load_data("SEQdata"),
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
            weight_preexpansion=True,
            bootstrap_nboot=1,
            bootstrap_sample=1.0,
            seed=42,
        ),
    )
    s.expand()
    s.bootstrap()
    s.fit()

    assert len(captured) == 1
    DT = captured[0]

    # The resampled encoded IDs must survive the bind (replicate copies stay
    # distinct groups for the cum_prod) ...
    id_mult = s._boot_id_mult
    orig_ids = set(s.data["ID"].unique().to_list())
    assert not set(DT["ID"].unique().to_list()) <= orig_ids

    # ... and with replicate sampling (sample=1.0 guarantees duplicated
    # subjects), every copy of the same original (id, trial, followup) row must
    # carry the SAME cumulative weight.
    decoded = DT.with_columns((pl.col("ID") // id_mult).alias("_orig"))
    dup = decoded.group_by(["_orig", "trial", "followup"]).agg(
        [pl.len().alias("n"), pl.col("weight").n_unique().alias("n_weights")]
    )
    assert dup.filter(pl.col("n") > 1).height > 0  # duplicates actually present
    assert dup.filter(pl.col("n_weights") > 1).height == 0
