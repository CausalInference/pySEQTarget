import numpy as np
import polars as pl

from ..helpers import _predict_model
from ..helpers._predict_model import _predict_model_pd, _prep_predict_frame


def _extract_class_probability(p, level_idx, is_binary):
    """
    Extract the probability for a specific class from model predictions.
    Handles both binary (logit) and multinomial (mnlogit) output formats.
    """
    if is_binary:
        # logit returns P(Y=1) directly as 1D array
        # For level 0: want P(stay at 0) = 1 - P(Y=1)
        # For level 1: want P(switch to 1) = P(Y=1)
        return p if level_idx == 1 else (1 - p)
    else:
        # mnlogit returns [P(Y=0), P(Y=1), ...] as 2D array
        if p.ndim == 1:
            p = p.reshape(-1, 1)
        return p[:, level_idx]


def _weight_predict(self, WDT):
    grouping = [self.id_col]
    grouping += ["trial"] if not self.weight_preexpansion else []
    time = self.time_col if self.weight_preexpansion else "followup"

    # Check if binary 0/1 treatment with censoring (set during fitting)
    # Must match the logic in _weight_fit.py
    is_binary = getattr(
        self,
        "_is_binary_treatment",
        sorted(self.treatment_level) == [0, 1] and self.method == "censoring",
    )

    if self.method == "ITT":
        WDT = WDT.with_columns(
            [pl.lit(1.0).alias("numerator"), pl.lit(1.0).alias("denominator")]
        )
    else:
        WDT = WDT.with_columns(
            [pl.lit(1.0).alias("numerator"), pl.lit(1.0).alias("denominator")]
        )

        if not self.excused:
            # Pre-allocate arrays once for all treatment levels
            pred_num = np.ones(WDT.height)
            pred_denom = np.ones(WDT.height)

            # Pre-compute treatment switch mask once
            switched_treatment = (WDT[self.treatment_col] != WDT["tx_lag"]).to_numpy()
            tx_lag_array = WDT["tx_lag"].to_numpy()

            for i, level in enumerate(self.treatment_level):
                lag_mask = tx_lag_array == level

                # Load models via offloader (handles both offloaded and in-memory models)
                denom_model = self._offloader.load_model(self.denominator_model[i])
                num_model = self._offloader.load_model(self.numerator_model[i])

                if (denom_model is None and num_model is None) or lag_mask.sum() == 0:
                    continue

                # Numerator and denominator predict on the same rows — pay the
                # filter + pandas conversion once and share the frame.
                subset_pd = _prep_predict_frame(self, WDT.filter(pl.Series(lag_mask)))

                if denom_model is not None:
                    p = _predict_model_pd(denom_model, subset_pd)
                    p_class = _extract_class_probability(p, i, is_binary)
                    pred_denom[lag_mask] = np.where(
                        switched_treatment[lag_mask], 1.0 - p_class, p_class
                    )

                if num_model is not None:
                    p = _predict_model_pd(num_model, subset_pd)
                    p_class = _extract_class_probability(p, i, is_binary)
                    pred_num[lag_mask] = np.where(
                        switched_treatment[lag_mask], 1.0 - p_class, p_class
                    )

            WDT = WDT.with_columns(
                [
                    pl.Series("numerator", pred_num),
                    pl.Series("denominator", pred_denom),
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
    # Full-frame pandas conversion shared by the cense and visit predictions
    # (built lazily — only when at least one of those model pairs exists).
    WDT_pd = None

    if self.cense_colname is not None:
        cense_num_model = self._offloader.load_model(self.cense_numerator_model)
        cense_denom_model = self._offloader.load_model(self.cense_denominator_model)
        if cense_num_model is not None and cense_denom_model is not None:
            WDT_pd = _prep_predict_frame(self, WDT)
            p_num = _predict_model_pd(cense_num_model, WDT_pd).flatten()
            p_denom = _predict_model_pd(cense_denom_model, WDT_pd).flatten()
            WDT = (
                WDT.with_columns(
                    [
                        pl.Series("cense_numerator", p_num),
                        pl.Series("cense_denominator", p_denom),
                    ]
                )
                .with_columns(
                    (pl.col("cense_numerator") / pl.col("cense_denominator")).alias(
                        "_cense"
                    )
                )
                .drop(["cense_numerator", "cense_denominator"])
            )
        else:
            WDT = WDT.with_columns(pl.lit(1.0).alias("_cense"))
    else:
        WDT = WDT.with_columns(pl.lit(1.0).alias("_cense"))

    if self.visit_colname is not None:
        visit_num_model = self._offloader.load_model(self.visit_numerator_model)
        visit_denom_model = self._offloader.load_model(self.visit_denominator_model)
        if visit_num_model is not None and visit_denom_model is not None:
            # The visit formulas don't reference the _cense column added above,
            # so the frame converted before the cense block is still valid.
            if WDT_pd is None:
                WDT_pd = _prep_predict_frame(self, WDT)
            p_num = _predict_model_pd(visit_num_model, WDT_pd).flatten()
            p_denom = _predict_model_pd(visit_denom_model, WDT_pd).flatten()
            WDT = (
                WDT.with_columns(
                    [
                        pl.Series("visit_numerator", p_num),
                        pl.Series("visit_denominator", p_denom),
                    ]
                )
                .with_columns(
                    (pl.col("visit_numerator") / pl.col("visit_denominator")).alias(
                        "_visit"
                    )
                )
                .drop(["visit_numerator", "visit_denominator"])
            )
        else:
            WDT = WDT.with_columns(pl.lit(1.0).alias("_visit"))
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
