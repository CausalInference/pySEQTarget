"""Behavioural tests for the integer-ID bootstrap path."""
import polars as pl

from pySEQTarget import SEQopts, SEQuential
from pySEQTarget.data import load_data
from pySEQTarget.helpers._bootstrap import _prepare_boot_data


def _build(**opts):
    opts.setdefault("seed", 42)
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
    return s


def test_expand_preserves_int_id_dtype():
    # SEQdata's ID is Int64. expand() must not coerce it to Utf8 - string IDs
    # make every downstream join/groupby ~3-5x slower than the int path.
    s = _build()
    assert s.DT.schema["ID"] == pl.Int64
    assert s.data.schema["ID"] == pl.Int64


def test_bootstrap_id_uses_integer_arithmetic_for_int_ids():
    # Resampled IDs are built as orig_id * id_mult + replicate. Each
    # (orig_id, replicate) pair has to map to a unique new ID and the original
    # ID must be recoverable via integer division.
    s = _build(bootstrap_nboot=3)
    s.bootstrap()

    boot = _prepare_boot_data(s, s.DT, boot_id=0)
    assert boot.schema["ID"] == pl.Int64
    id_mult = s._boot_id_mult
    assert id_mult >= 2

    orig_ids = set(s.DT["ID"].to_list())
    # Recovered IDs are all from the original set
    recovered = boot["ID"] // id_mult
    assert set(recovered.to_list()) <= orig_ids
    # The replicate component is bounded by id_mult, so the (orig, rep) pair is
    # uniquely encoded
    rep = boot["ID"] - recovered * id_mult
    assert rep.min() >= 0
    assert rep.max() < id_mult


def test_bootstrap_id_falls_back_to_string_concat_for_non_int_ids():
    # User-supplied non-integer IDs still work via the original "{id}_{rep}"
    # string-concat path. Build a String-keyed DT manually since pySEQTarget no
    # longer casts to Utf8.
    s = _build(bootstrap_nboot=3)
    s.bootstrap()
    # Coerce id_col to Utf8 in both the DT and the boot_samples Counter keys so
    # the join lines up
    s.DT = s.DT.with_columns(pl.col("ID").cast(pl.Utf8))
    from collections import Counter
    s._boot_samples = [
        Counter({str(k): v for k, v in c.items()}) for c in s._boot_samples
    ]

    boot = _prepare_boot_data(s, s.DT, boot_id=0)
    assert boot.schema["ID"] == pl.Utf8
    # IDs follow the "{orig}_{rep}" pattern
    assert all("_" in v for v in boot["ID"].unique().to_list())
