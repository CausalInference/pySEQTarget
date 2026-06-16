import warnings

from pySEQTarget import SEQopts, SEQuential
from pySEQTarget.data import load_data


def test_compevent_offload():
    data = load_data("SEQdata_LTFU")
    options = SEQopts(
        bootstrap_nboot=2,
        cense_colname="LTFU",
        excused=True,
        excused_colnames=["excusedZero", "excusedOne"],
        km_curves=True,
        selection_random=True,
        selection_sample=0.30,
        weighted=True,
        weight_lag_condition=False,
        weight_p99=True,
        weight_preexpansion=True,
        offload=True,
        seed=42,
    )

    model = SEQuential(
        data,
        id_col="ID",
        time_col="time",
        eligible_col="eligible",
        treatment_col="tx_init",
        outcome_col="outcome",
        time_varying_cols=["N", "L", "P"],
        fixed_cols=["sex"],
        method="censoring",
        parameters=options,
    )
    model.expand()
    model.bootstrap()
    # Warnings from statsmodels about overflow in some bootstraps
    warnings.filterwarnings("ignore")
    model.fit()
    model.survival()


def test_weight_models_fully_offloaded(tmp_path):
    # _offload_weights used to check nonexistent attributes (LTFU_model,
    # visit_model) and only offload the LAST treatment level's model. All
    # fitted weight models must end up as path refs, and summaries must load
    # them back transparently.
    data = load_data("SEQdata_LTFU")
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
            weight_preexpansion=False,
            cense_colname="LTFU",
            offload=True,
            offload_dir=str(tmp_path),
            seed=42,
        ),
    )
    s.expand()
    s.fit()

    for m in s.numerator_model + s.denominator_model:
        assert m is None or isinstance(m, str)
    assert isinstance(s.cense_numerator_model, str)
    assert isinstance(s.cense_denominator_model, str)

    out = s.collect()
    for kind in ("numerator", "denominator", "outcome"):
        summaries = out.summary(kind)
        assert len(summaries) >= 1
        assert all(str(smry) for smry in summaries)


def test_serial_bootstrap_offload_writes_DT_once(monkeypatch, tmp_path):
    # The serial bootstrap path used to save the _DT parquet twice per fit.
    from pySEQTarget.helpers._offloader import Offloader

    writes = []
    real_save = Offloader.save_dataframe

    def spy(self, df, name):
        writes.append(name)
        return real_save(self, df, name)

    monkeypatch.setattr(Offloader, "save_dataframe", spy)

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
        parameters=SEQopts(
            bootstrap_nboot=2,
            seed=42,
            offload=True,
            offload_dir=str(tmp_path),
        ),
    )
    s.expand()
    s.bootstrap()
    s.fit()

    assert writes.count("_DT") == 1
