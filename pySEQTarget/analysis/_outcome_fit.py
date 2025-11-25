import re

import polars as pl
import statsmodels.api as sm
import statsmodels.formula.api as smf


def _outcome_fit(
    self,
    df: pl.DataFrame,
    outcome: str,
    formula: str,
    weighted: bool = False,
    weight_col: str = "weight",
):
    if weighted:
        df = df.with_columns(
            pl.col(weight_col).clip(
                lower_bound=self.weight_min, upper_bound=self.weight_max
            )
        )

    if self.method == "censoring":
        df = df.filter(pl.col("switch") != 1)

    df_pd = df.to_pandas()

    df_pd[self.treatment_col] = df_pd[self.treatment_col].astype("category")
    tx_bas = f"{self.treatment_col}{self.indicator_baseline}"
    df_pd[tx_bas] = df_pd[tx_bas].astype("category")

    if self.followup_class and not self.followup_spline:
        df_pd["followup"] = df_pd["followup"].astype("category")
        squared_col = f"followup{self.indicator_squared}"
        if squared_col in df_pd.columns:
            df_pd[squared_col] = df_pd[squared_col].astype("category")

    if self.followup_spline:
        spline = "cr(followup, df=3)"

        formula = re.sub(r"(\w+)\s*\*\s*followup\b", rf"\1*{spline}", formula)
        formula = re.sub(r"\bfollowup\s*\*\s*(\w+)", rf"{spline}*\1", formula)
        formula = re.sub(
            rf"\bfollowup{re.escape(self.indicator_squared)}\b", "", formula
        )
        formula = re.sub(r"\bfollowup\b", "", formula)

        formula = re.sub(r"\s+", " ", formula)
        formula = re.sub(r"\+\s*\+", "+", formula)
        formula = re.sub(r"^\s*\+\s*|\s*\+\s*$", "", formula).strip()

        if formula:
            formula = f"{formula} + I({spline}**2)"
        else:
            formula = f"I({spline}**2)"

    if self.fixed_cols:
        for col in self.fixed_cols:
            if col in df_pd.columns:
                df_pd[col] = df_pd[col].astype("category")

    full_formula = f"{outcome} ~ {formula}"

    glm_kwargs = {
        "formula": full_formula,
        "data": df_pd,
        "family": sm.families.Binomial(),
    }

    if weighted:
        glm_kwargs["var_weights"] = df_pd[weight_col]

    model = smf.glm(**glm_kwargs)
    model_fit = model.fit()
    return model_fit
