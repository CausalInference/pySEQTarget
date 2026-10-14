"""Terms that remember their fitting data, e.g. centred splines or center(x)."""

import numpy as np
import pytest

import pySEQTarget.analysis._survival_pred as survival_pred
from pySEQTarget import SEQopts, SEQuential
from pySEQTarget.data import load_data


def _model(**opts):
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
        parameters=SEQopts(seed=42, **opts),
    )
    s.expand()
    if s.bootstrap_nboot:
        s.bootstrap()
    s.fit()
    return s


def _weights(s):
    return s.DT.sort(["ID", "trial", "followup"])["weight"].to_numpy()


@pytest.mark.filterwarnings("ignore")
def test_glum_weight_spline_survives_offloading(tmp_path):
    # Unpickled glum models rebuild their design from two rows, so the
    # centring constraint must be baked into the formula, not memorised
    opts = dict(
        weighted=True,
        weight_preexpansion=False,
        weight_spline=True,
        glm_package="glum",
        offload_dir=str(tmp_path),
    )
    in_memory = _weights(_model(**opts))
    offloaded = _weights(_model(offload=True, **opts))
    assert np.allclose(in_memory, offloaded)


@pytest.mark.filterwarnings("ignore")
@pytest.mark.parametrize("term", ["center(N_bas)", "standardize(L_bas)"])
def test_bootstrap_survival_uses_each_replicates_own_transform(term, monkeypatch):
    opts = dict(
        covariates=f"tx_init_bas+followup+trial+{term}",
        km_curves=True,
        bootstrap_nboot=2,
    )

    def km(s):
        s.survival()
        return s.km_data.sort(["estimate", "tx_init", "followup"])[
            ["pred", "LCI", "UCI"]
        ].to_numpy()

    cached = km(_model(**opts))
    # Without the cached design matrix every model builds its own
    monkeypatch.setattr(
        survival_pred,
        "_cached_predict",
        lambda model, X, ref, data: survival_pred._safe_predict(model, data),
    )
    uncached = km(_model(**opts))
    assert np.allclose(cached, uncached, equal_nan=True)
