import polars as pl

from pySEQTarget import SEQopts, SEQuential
from pySEQTarget.data import load_data


def test_weight_predict_receives_polars_frame(monkeypatch):
    # Guard against re-introducing the pl.from_pandas() round-trip in the
    # weighted fit block: _weight_predict must receive the original polars
    # frame (not one that was just rebuilt from pandas), since the weight-fit
    # helpers store models on `self` and don't mutate WDT.
    import importlib

    seq_mod = importlib.import_module("pySEQTarget.SEQuential")

    original = seq_mod._weight_predict
    seen_types = []

    def spy(self, WDT):
        seen_types.append(type(WDT))
        return original(self, WDT)

    monkeypatch.setattr(seq_mod, "_weight_predict", spy)

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
            bootstrap_nboot=2,
            seed=42,
        ),
    )
    s.expand()
    s.bootstrap()
    s.fit()

    assert len(seen_types) >= 1
    assert all(t is pl.DataFrame for t in seen_types)
