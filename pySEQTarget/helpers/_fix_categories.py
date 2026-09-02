import pandas as pd


def _model_design_info(model):
    """The patsy DesignInfo a fitted model was built with, or None.

    statsmodels 0.15 renamed ``model.data.design_info`` to the engine-neutral
    ``model.data.model_spec`` (still a patsy DesignInfo under the default patsy
    engine); older statsmodels and the glum/jax wrappers use ``design_info``.
    Check both so every backend and version resolves through one place.
    """
    inner = getattr(model, "model", None)
    data = getattr(inner, "data", None)
    if data is None:
        return None
    design_info = getattr(data, "design_info", None)
    if design_info is None:
        design_info = getattr(data, "model_spec", None)
    return design_info


def _fix_categories_for_predict(model, newdata):
    """
    Fix categorical column ordering in newdata to match what the model expects.
    """
    design_info = _model_design_info(model)
    if design_info is not None:
        for factor, factor_info in design_info.factor_infos.items():
            if factor_info.type == "categorical":
                col_name = factor.name()
                if col_name in newdata.columns:
                    expected_categories = list(factor_info.categories)
                    cat_type = pd.CategoricalDtype(categories=expected_categories)
                    newdata[col_name] = (
                        newdata[col_name]
                        .astype(type(expected_categories[0]))
                        .astype(cat_type)
                    )
    return newdata
