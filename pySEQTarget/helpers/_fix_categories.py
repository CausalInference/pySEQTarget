def _fix_categories_for_predict(model, newdata):
    """
    Fix categorical column ordering in newdata to match what the model expects.
    """
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
    return newdata
