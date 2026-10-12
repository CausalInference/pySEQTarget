import numpy as np
import patsy


def _undefined_terms(formula, data):
    """Design-matrix columns with missing or infinite values; ``formula`` may be a DesignInfo."""
    if isinstance(formula, str):
        formula = formula.split("~", 1)[-1]
        if formula.strip() in ("", "1"):
            return []
    X = patsy.dmatrix(
        formula, data, NA_action=patsy.NAAction(NA_types=[]), return_type="dataframe"
    )
    return [col for col in X.columns if not np.isfinite(X[col]).all()]


def _check_design(formula, data):
    """Raise, naming the terms, if a model's design matrix has missing or infinite values."""
    bad = _undefined_terms(formula, data)
    if bad:
        raise ValueError(
            f"Model term(s) {', '.join(bad)} produced missing or infinite values, "
            "e.g. log() or sqrt() of a non-positive value; transform the variable "
            "so the term is defined for every row."
        )
