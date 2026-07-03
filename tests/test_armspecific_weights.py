"""Per-treatment-level weight models: 'numerator'/'denominator' given as a list
fit a separate model (with its own covariates) in each treatment arm. Post-
expansion weights only (weight_preexpansion=False).
"""

import numpy as np
import pytest

from pySEQTarget import SEQopts, SEQuential
from pySEQTarget.data import load_data


def _model(numerator, denominator, **opts):
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
            weight_preexpansion=False,
            numerator=numerator,
            denominator=denominator,
            seed=42,
            **opts,
        ),
    )
    s.expand()
    s.fit()
    return s


def _coef_names(model):
    return list(model.params.index)


def test_per_arm_denominator_fits_different_covariates_in_each_arm():
    s = _model(numerator=["sex", "sex"], denominator=["N+sex", "N+L+P+sex"])

    den0 = _coef_names(s.denominator_model[0])
    den1 = _coef_names(s.denominator_model[1])

    # Arm 0's model excludes L and P; arm 1's includes them.
    assert not any(n.startswith("L") for n in den0)
    assert not any(n.startswith("P") for n in den0)
    assert any(n.startswith("N") for n in den0)
    assert any(n.startswith("L") for n in den1)
    assert any(n.startswith("P") for n in den1)


def test_per_arm_formulas_with_identical_elements_match_shared_fit():
    shared = _model(numerator="sex", denominator="N+L+P+sex")
    perarm = _model(
        numerator=["sex", "sex"], denominator=["N+L+P+sex", "N+L+P+sex"]
    )

    for i in range(2):
        np.testing.assert_allclose(
            np.asarray(shared.numerator_model[i].params),
            np.asarray(perarm.numerator_model[i].params),
        )
        np.testing.assert_allclose(
            np.asarray(shared.denominator_model[i].params),
            np.asarray(perarm.denominator_model[i].params),
        )

    # Same weights feed the outcome model, so its coefficients must match too.
    shared_out = np.concatenate(
        [np.asarray(m["outcome"].params) for m in shared.outcome_model]
    )
    perarm_out = np.concatenate(
        [np.asarray(m["outcome"].params) for m in perarm.outcome_model]
    )
    np.testing.assert_allclose(shared_out, perarm_out)


def test_per_arm_weight_formulas_rejected_for_preexpansion_weights():
    with pytest.raises(ValueError, match="post-expansion weights"):
        SEQuential(
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
                numerator=["sex", "sex"],
                denominator=["N+sex", "N+L+P+sex"],
            ),
        )


def test_per_arm_weight_formulas_rejected_for_wrong_length():
    with pytest.raises(ValueError, match="one per treatment level"):
        SEQuential(
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
                weight_preexpansion=False,
                denominator=["N+sex", "N+L+P+sex", "N+sex"],
            ),
        )


def test_per_arm_weight_formulas_rejected_for_itt():
    with pytest.raises(ValueError, match="weighted, non-ITT"):
        SEQuential(
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
                weighted=True,
                cense_colname="eligible",
                numerator=["sex", "sex"],
                denominator=["N+sex", "N+L+P+sex"],
            ),
        )
