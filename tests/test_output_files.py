"""Markdown report generation (SEQoutput.to_md / _build_md).

_build_md used to index numerator_models[0]/outcome_models[0] directly, which
crashed for weighted ITT analyses (no treatment-weight models exist — the
attribute is None) and for offloaded models (path refs, not fitted objects).
It now routes through SEQoutput.summary, which handles both.
"""

import pytest

# pandas.DataFrame.to_markdown needs tabulate (the "output" extra).
pytest.importorskip("tabulate")

from pySEQTarget import SEQopts, SEQuential  # noqa: E402
from pySEQTarget.data import load_data  # noqa: E402


def test_to_md_weighted_ITT_without_numerator_models(tmp_path):
    s = SEQuential(
        load_data("SEQdata_LTFU"),
        id_col="ID",
        time_col="time",
        eligible_col="eligible",
        treatment_col="tx_init",
        outcome_col="outcome",
        time_varying_cols=["N", "L", "P"],
        fixed_cols=["sex"],
        method="ITT",
        parameters=SEQopts(weighted=True, cense_colname="LTFU", seed=42),
    )
    s.expand()
    s.fit()
    out = s.collect()

    md_file = tmp_path / "report.md"
    out.to_md(str(md_file))
    content = md_file.read_text()
    assert "Outcome Model" in content
    # No treatment-weight models under ITT: the section is skipped, not a crash.
    assert "Numerator Model" not in content


def test_to_md_with_offloaded_models(tmp_path):
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
            offload=True,
            offload_dir=str(tmp_path / "models"),
            seed=42,
        ),
    )
    s.expand()
    s.fit()
    out = s.collect()

    md_file = tmp_path / "report.md"
    out.to_md(str(md_file))
    content = md_file.read_text()
    assert "Numerator Model" in content
    assert "Denominator Model" in content
    assert "Outcome Model" in content
