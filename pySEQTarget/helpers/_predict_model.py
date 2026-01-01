import numpy as np

from ._fix_categories import _fix_categories_for_predict


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
