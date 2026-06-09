import numpy as np

from ._fix_categories import _fix_categories_for_predict


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
        if np.any(np.isnan(probs)):
            raise ValueError(
                "NaN values in predicted probabilities. This typically indicates "
                "a mismatch between the model's training data types and the "
                "prediction data (e.g. missing categorical casting), or numerical "
                "overflow in the model coefficients."
            )
        probs = np.clip(probs, 0, 1)

    return probs


def _predict_model(self, model, newdata):
    newdata = newdata.to_pandas()

    # Original behavior - convert fixed_cols to category
    for col in self.fixed_cols:
        if col in newdata.columns:
            newdata[col] = newdata[col].astype("category")

    try:
        return np.array(model.predict(newdata))
    except Exception as e:
        if "mismatching levels" in str(e):
            newdata = _fix_categories_for_predict(model, newdata)
            return np.array(model.predict(newdata))
        else:
            raise
