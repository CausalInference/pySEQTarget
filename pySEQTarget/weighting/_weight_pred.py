import numpy as np
import polars as pl

from ..helpers import _predict_model


def _weight_predict(self, WDT):
    grouping = [self.id_col]
    grouping += ["trial"] if not self.weight_preexpansion else []
    time = self.time_col if self.weight_preexpansion else "followup"

    # Check if binary 0/1 treatment with censoring (set during fitting)
    # Must match the logic in _weight_fit.py
    is_binary = getattr(self, '_is_binary_treatment', 
                        sorted(self.treatment_level) == [0, 1] and self.method == "censoring")

    if self.method == "ITT":
        WDT = WDT.with_columns(
            [pl.lit(1.0).alias("numerator"), pl.lit(1.0).alias("denominator")]
        )
    else:
        WDT = WDT.with_columns(
            [pl.lit(1.0).alias("numerator"), pl.lit(1.0).alias("denominator")]
        )

        if not self.excused:
            for i, level in enumerate(self.treatment_level):
                mask = pl.col("tx_lag") == level
                lag_mask = (WDT["tx_lag"] == level).to_numpy()

                # Load models via offloader (handles both offloaded and in-memory models)
                denom_model = self._offloader.load_model(self.denominator_model[i])
                num_model = self._offloader.load_model(self.numerator_model[i])

                if denom_model is not None:
                    pred_denom = np.ones(WDT.height)
                    if lag_mask.sum() > 0:
                        subset = WDT.filter(pl.Series(lag_mask))
                        p = _predict_model(self, denom_model, subset)
                        
                        # Handle binary vs multinomial prediction output
                        if is_binary:
                            # logit returns P(Y=1) directly as 1D array
                            # For i=0 (level 0): want P(stay at 0) = 1 - P(Y=1)
                            # For i=1 (level 1): want P(switch to 1) = P(Y=1)
                            p_class = p if i == 1 else (1 - p)
                        else:
                            # mnlogit returns [P(Y=0), P(Y=1), ...] as 2D array
                            if p.ndim == 1:
                                p = p.reshape(-1, 1)
                            p_class = p[:, i]
                        
                        switched_treatment = (
                            subset[self.treatment_col] != subset["tx_lag"]
                        ).to_numpy()
                        pred_denom[lag_mask] = np.where(switched_treatment, 1.0 - p_class, p_class)
                else:
                    pred_denom = np.ones(WDT.height)

                if num_model is not None:
                    pred_num = np.ones(WDT.height)
                    if lag_mask.sum() > 0:
                        subset = WDT.filter(pl.Series(lag_mask))
                        p = _predict_model(self, num_model, subset)
                        
                        # Handle binary vs multinomial prediction output
                        if is_binary:
                            # logit returns P(Y=1) directly as 1D array
                            p_class = p if i == 1 else (1 - p)
                        else:
                            # mnlogit returns [P(Y=0), P(Y=1), ...] as 2D array
                            if p.ndim == 1:
                                p = p.reshape(-1, 1)
                            p_class = p[:, i]
                        
                        switched_treatment = (
                            subset[self.treatment_col] != subset["tx_lag"]
                        ).to_numpy()
                        pred_num[lag_mask] = np.where(switched_treatment, 1.0 - p_class, p_class)
                else:
                    pred_num = np.ones(WDT.height)

                WDT = WDT.with_columns(
                    [
                        pl.when(mask)
                        .then(pl.Series(pred_num))
                        .otherwise(pl.col("numerator"))
                        .alias("numerator"),
                        pl.when(mask)
                        .then(pl.Series(pred_denom))
                        .otherwise(pl.col("denominator"))
                        .alias("denominator"),
                    ]
                )

        else:
            for i, level in enumerate(self.treatment_level):
                col = self.excused_colnames[i]

                if col is not None:
                    denom_model = self._offloader.load_model(self.denominator_model[i])
                    denom_mask = ((WDT["tx_lag"] == level) & (WDT[col] != 1)).to_numpy()

                    if denom_model is not None and denom_mask.sum() > 0:
                        pred_denom = np.ones(WDT.height)
                        subset = WDT.filter(pl.Series(denom_mask))
                        p = _predict_model(self, denom_model, subset)

                        if p.ndim == 1:
                            prob_switch = p
                        else:
                            prob_switch = p[:, 1] if p.shape[1] > 1 else p.flatten()

                        pred_denom[denom_mask] = prob_switch

                        WDT = WDT.with_columns(
                            pl.when(pl.Series(denom_mask))
                            .then(pl.Series(pred_denom))
                            .otherwise(pl.col("denominator"))
                            .alias("denominator")
                        )

                    if i == 0:
                        flip_mask = (
                            (WDT["tx_lag"] == level)
                            & (WDT[col] == 0)
                            & (WDT[self.treatment_col] == level)
                        ).to_numpy()
                    else:
                        flip_mask = (
                            (WDT["tx_lag"] == level)
                            & (WDT[col] == 0)
                            & (WDT[self.treatment_col] != level)
                        ).to_numpy()

                    WDT = WDT.with_columns(
                        pl.when(pl.Series(flip_mask))
                        .then(1.0 - pl.col("denominator"))
                        .otherwise(pl.col("denominator"))
                        .alias("denominator")
                    )

            if self.weight_preexpansion:
                WDT = WDT.with_columns(pl.lit(1.0).alias("numerator"))
            else:
                for i, level in enumerate(self.treatment_level):
                    col = self.excused_colnames[i]

                    if col is not None:
                        num_model = self._offloader.load_model(self.numerator_model[i])
                        num_mask = (
                            (WDT[self.treatment_col] == level) & (WDT[col] == 0)
                        ).to_numpy()

                        if num_model is not None and num_mask.sum() > 0:
                            pred_num = np.ones(WDT.height)
                            subset = WDT.filter(pl.Series(num_mask))
                            p = _predict_model(self, num_model, subset)

                            if p.ndim == 1:
                                prob_switch = p
                            else:
                                prob_switch = p[:, 1] if p.shape[1] > 1 else p.flatten()

                            pred_num[num_mask] = prob_switch

                            WDT = WDT.with_columns(
                                pl.when(pl.Series(num_mask))
                                .then(pl.Series(pred_num))
                                .otherwise(pl.col("numerator"))
                                .alias("numerator")
                            )

                first_level = self.treatment_level[0]
                WDT = WDT.with_columns(
                    pl.when(pl.col(self.treatment_col) == first_level)
                    .then(1.0 - pl.col("numerator"))
                    .otherwise(pl.col("numerator"))
                    .alias("numerator")
                )
    if self.cense_colname is not None:
        cense_num_model = self._offloader.load_model(self.cense_numerator_model)
        cense_denom_model = self._offloader.load_model(self.cense_denominator_model)
        p_num = _predict_model(self, cense_num_model, WDT).flatten()
        p_denom = _predict_model(self, cense_denom_model, WDT).flatten()
        WDT = WDT.with_columns(
            [
                pl.Series("cense_numerator", p_num),
                pl.Series("cense_denominator", p_denom),
            ]
        ).with_columns(
            (pl.col("cense_numerator") / pl.col("cense_denominator")).alias("_cense")
        )
    else:
        WDT = WDT.with_columns(pl.lit(1.0).alias("_cense"))

    if self.visit_colname is not None:
        visit_num_model = self._offloader.load_model(self.visit_numerator_model)
        visit_denom_model = self._offloader.load_model(self.visit_denominator_model)
        p_num = _predict_model(self, visit_num_model, WDT).flatten()
        p_denom = _predict_model(self, visit_denom_model, WDT).flatten()

        WDT = WDT.with_columns(
            [
                pl.Series("visit_numerator", p_num),
                pl.Series("visit_denominator", p_denom),
            ]
        ).with_columns(
            (pl.col("visit_numerator") / pl.col("visit_denominator")).alias("_visit")
        )
    else:
        WDT = WDT.with_columns(pl.lit(1.0).alias("_visit"))

    kept = [
        "numerator",
        "denominator",
        "_cense",
        "_visit",
        self.id_col,
        "trial",
        time,
        "tx_lag",
    ]
    exists = [col for col in kept if col in WDT.columns]
    return WDT.select(exists).sort(grouping + [time])
