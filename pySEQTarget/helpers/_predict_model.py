import numpy as np


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
            # Fix category levels from model's design_info
            if hasattr(model, 'model') and hasattr(model.model, 'data') and hasattr(model.model.data, 'design_info'):
                design_info = model.model.data.design_info
                for factor, factor_info in design_info.factor_infos.items():
                    if factor_info.type == 'categorical':
                        col_name = factor.name()
                        if col_name in newdata.columns:
                            expected_categories = list(factor_info.categories)
                            newdata[col_name] = newdata[col_name].astype(str)
                            newdata[col_name] = newdata[col_name].astype('category')
                            newdata[col_name] = newdata[col_name].cat.set_categories(expected_categories)
            return np.array(model.predict(newdata))
        else:
            raise
