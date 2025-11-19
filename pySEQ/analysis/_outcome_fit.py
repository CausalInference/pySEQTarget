import statsmodels.api as sm
import statsmodels.formula.api as smf
import polars as pl

def _outcome_fit(self,
    df: pl.DataFrame,
    outcome: str,
    formula: str,
    weighted: bool = False,
    weight_col: str = "weight",
):
    if weighted:
        df = self.DT.with_columns(
            pl.col(weight_col).clip(
                lower_bound=self.weight_min, 
                upper_bound=self.weight_max
                )
            )
    df_pd = df.to_pandas()
    for col in self.fixed_cols:
        if col in df_pd.columns:
            df_pd[col] = df_pd[col].astype("category")
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
