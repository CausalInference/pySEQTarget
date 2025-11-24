import numpy as np


def _predict_model(self, model, newdata):
    newdata = newdata.to_pandas()
    for col in self.fixed_cols:
        if col in newdata.columns:
            newdata[col] = newdata[col].astype("category")
    return np.array(model.predict(newdata))
