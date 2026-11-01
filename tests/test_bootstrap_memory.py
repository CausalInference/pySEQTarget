"""Bootstrap replicate outcome models keep no per-row data, so memory does not
grow with the expanded data times bootstrap_nboot."""

import pickle

import numpy as np
import pandas as pd
import pytest

from pySEQTarget import SEQopts, SEQuential
from pySEQTarget.data import load_data


def _fit(**kwargs):
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
        parameters=SEQopts(weighted=True, bootstrap_nboot=2, seed=7, **kwargs),
    )
    s.expand()
    s.bootstrap()
    s.fit()
    return s


@pytest.mark.filterwarnings("ignore")
@pytest.mark.parametrize(
    "kwargs",
    [
        dict(glm_package="statsmodels"),
        dict(glm_package="glum"),
        dict(glm_package="statsmodels", parallel=True, ncores=2),
        dict(glm_package="statsmodels", offload=True),
    ],
)
def test_replicate_models_keep_no_per_row_data(kwargs, tmp_path):
    if kwargs.get("offload"):
        kwargs["offload_dir"] = str(tmp_path)
    s = _fit(**kwargs)
    load = s._offloader.load_model
    main = load(s.outcome_model[0]["outcome"])
    reps = [load(m["outcome"]) for m in s.outcome_model[1:]]
    assert len(reps) == 2

    # ~100k expanded rows: a model holding per-row data pickles to megabytes
    for m in reps:
        assert len(pickle.dumps(m)) < 100_000

    # The replicates still predict, give standard errors and summarise
    df = s.DT.head(20).to_pandas()
    df["tx_init_bas"] = pd.Categorical(df["tx_init_bas"], categories=[0, 1])
    for m in reps:
        assert len(m.predict(df)) == 20
        assert m.bse.notna().all()
        m.summary()

    # Unpickling rebuilds the formula design from the reference rows kept in
    # place of the data; predictions must not change
    for m in reps:
        restored = pickle.loads(pickle.dumps(m))
        np.testing.assert_allclose(restored.predict(df), m.predict(df))

    # The main fit is untouched
    if kwargs["glm_package"] == "statsmodels" and not kwargs.get("offload"):
        assert main.model.data.frame is not None


@pytest.mark.filterwarnings("ignore")
def test_stateful_transform_keeps_the_full_frame():
    # center() memorises the mean of all the data, which a few reference rows
    # cannot reproduce, so the frame is kept for unpickling
    s = _fit(covariates="tx_init_bas+followup+trial+sex+center(N)")
    m = s.outcome_model[1]["outcome"]
    assert len(m.model.data.frame) > 10_000
    df = s.DT.head(20).to_pandas()
    df["tx_init_bas"] = pd.Categorical(df["tx_init_bas"], categories=[0, 1])
    restored = pickle.loads(pickle.dumps(m))
    np.testing.assert_allclose(restored.predict(df), m.predict(df))
