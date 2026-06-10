import re

import polars as pl

from pySEQTarget import SEQopts, SEQuential
from pySEQTarget.data import load_data


def _expand(method, capsys):
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
        method=method,
        parameters=SEQopts(verbose=True),
    )
    s.expand()
    return s, capsys.readouterr().out


def test_verbose_reports_uncensored_censored_split(capsys):
    # Under censoring the verbose output reports how many expanded rows enter the
    # outcome model (un-censored) vs are artificially censored, so the count
    # lines up with implementations that print only the modelled rows.
    s, out = _expand("censoring", capsys)

    total = int(re.search(r"Final analysis dataset: ([\d,]+)", out).group(1).replace(",", ""))
    unc = int(re.search(r"uncensored\): ([\d,]+)", out).group(1).replace(",", ""))
    cen = int(re.search(r"treatment switch\): ([\d,]+)", out).group(1).replace(",", ""))

    assert unc + cen == total
    assert cen > 0  # SEQdata has treatment switches, so some rows are censored
    # The reported un-censored count must equal the rows _outcome_fit fits on.
    assert unc == s.DT.filter(pl.col("switch") != 1).height


def test_verbose_no_censored_split_for_itt(capsys):
    # ITT applies no artificial censoring, so the split is not reported.
    _, out = _expand("ITT", capsys)
    assert "uncensored" not in out
    assert "artificially censored" not in out
