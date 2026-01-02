import warnings

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
        If True, clip probabilities to [0, 1] and replace NaN with 0.5
    """
    data = data.copy()
    
    try:
        probs = model.predict(data)
    except Exception as e:
        if "mismatching levels" in str(e):
            data = _fix_categories_for_predict(model, data)
            probs = model.predict(data)
        else:
            raise
    
    if clip_probs:
        probs = np.array(probs)
        if np.any(np.isnan(probs)):
            warnings.warn("NaN values in predicted probabilities, replacing with 0.5")
            probs = np.where(np.isnan(probs), 0.5, probs)
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
