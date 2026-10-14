import numpy as np

from ._check_design import _check_design
from ._fix_categories import (_cast_fixed_categories,
                              _fix_categories_for_predict, _model_design_info)


def _check_predictions(model, data, probs):
    """If predictions are NaN because a model term is undefined, say which."""
    if np.any(np.isnan(probs)):
        design_info = _model_design_info(model)
        if design_info is not None:
            _check_design(design_info, data)


def _safe_predict(model, data, clip_probs=True):
    """
    Predict with category fix fallback if needed.

    Parameters
    ----------
    model : statsmodels model
        Fitted model object
    data : pandas DataFrame
        Data to predict on
    clip_probs : bool
        If True, clip probabilities to [0, 1]. Raises ValueError if any
        predicted probability is NaN (this signals a train/predict dtype
        mismatch or coefficient overflow, not a value to silently impute).
    """
    try:
        probs = model.predict(data)
    except Exception as e:
        if "mismatching levels" in str(e):
            data = _fix_categories_for_predict(model, data.copy())
            probs = model.predict(data)
        else:
            raise

    if clip_probs:
        probs = np.array(probs)
        _check_predictions(model, data, probs)
        if np.any(np.isnan(probs)):
            raise ValueError(
                "NaN values in predicted probabilities. This typically indicates "
                "a mismatch between the model's training data types and the "
                "prediction data (e.g. missing categorical casting), or numerical "
                "overflow in the model coefficients."
            )
        probs = np.clip(probs, 0, 1)

    return probs


def _prep_predict_frame(self, newdata):
    """Convert a polars frame to pandas with non-numeric fixed_cols cast to category.

    Split out from _predict_model so callers predicting with several models on
    the same rows (e.g. numerator + denominator in _weight_predict) can pay
    the conversion once and share the frame.
    """
    return _cast_fixed_categories(self.fixed_cols, newdata.to_pandas())


def _predict_model_pd(model, newdata):
    """Predict on an already-prepared pandas frame, with category fix retry."""
    try:
        probs = np.array(model.predict(newdata))
    except Exception as e:
        if "mismatching levels" not in str(e):
            raise
        newdata = _fix_categories_for_predict(model, newdata)
        probs = np.array(model.predict(newdata))
    _check_predictions(model, newdata, probs)
    return probs


def _predict_model(self, model, newdata):
    return _predict_model_pd(model, _prep_predict_frame(self, newdata))
