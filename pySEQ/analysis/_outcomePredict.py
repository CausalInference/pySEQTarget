import statsmodels.api as sm
import statsmodels.formula.api as smf
import polars as pl

def _outcome_fit(
    df: pl.DataFrame,
    outcome: str,
    formula: str,
    weighted: bool = False,
    weight_col: str = "weight",
):
    df_pd = df.to_pandas()
    formula = f"{outcome}~{formula}"

    if weighted:
        model = smf.glm(
            formula=formula,
            data=df_pd,
            family=sm.families.Binomial(),
            freq_weights=df_pd[weight_col])
    else:
        model = smf.glm(
            formula=formula,
            data=df_pd,
            family=sm.families.Binomial())

    model_fit = model.fit()
    return model_fit
