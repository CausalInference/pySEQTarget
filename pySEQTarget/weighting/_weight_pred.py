import numpy as np
import polars as pl

from ..helpers import _predict_model
from ..helpers._predict_model import _predict_model_pd, _prep_predict_frame


def _extract_class_probability(p, model, level, is_binary):
    """P(treatment == level): logit predicts P(A = 1), mnlogit one column per class."""
    if is_binary:
        return p if level == 1 else 1 - p
    if level not in model.seq_classes:
        return np.zeros(len(p))
    return p.reshape(len(p), -1)[:, model.seq_classes.index(level)]


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
                    p_class = _extract_class_probability(
                        p, denom_model, level, is_binary
                    )
                    pred_denom[lag_mask] = np.where(
                        switched_treatment[lag_mask], 1.0 - p_class, p_class
                    )

                if num_model is not None:
                    p = _predict_model_pd(num_model, subset_pd)
                    p_class = _extract_class_probability(p, num_model, level, is_binary)
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
                if col is None:
                    continue
                denom_model = self._offloader.load_model(self.denominator_model[i])
                denom_mask = ((WDT["tx_lag"] == level) & (WDT[col] != 1)).to_numpy()
                if denom_model is None or denom_mask.sum() == 0:
                    continue
                subset = WDT.filter(pl.Series(denom_mask))
                p = _predict_model(self, denom_model, subset)
                if self.weight_preexpansion:
                    # Treatment model: probability of the treatment received
                    p_level = _extract_class_probability(
                        p, denom_model, level, is_binary
                    )
                    stay = (subset[self.treatment_col] == level).to_numpy()
                    prob = np.where(stay, p_level, 1.0 - p_level)
                else:
                    # Censoring model: probability of remaining uncensored
                    p_cens = _extract_class_probability(p, denom_model, 1, is_binary)
                    prob = np.where(
                        subset["switch"].to_numpy() == 0, 1.0 - p_cens, p_cens
                    )
                pred_denom = np.ones(WDT.height)
                pred_denom[denom_mask] = prob
                WDT = WDT.with_columns(
                    pl.when(pl.Series(denom_mask))
                    .then(pl.Series(pred_denom))
                    .otherwise(pl.col("denominator"))
                    .alias("denominator")
                )

            if self.weight_preexpansion:
                WDT = WDT.with_columns(pl.lit(1.0).alias("numerator"))
            else:
                for i, level in enumerate(self.treatment_level):
                    col = self.excused_colnames[i]
                    if col is None:
                        continue
                    num_model = self._offloader.load_model(self.numerator_model[i])
                    num_mask = (
                        (WDT[self.treatment_col] == level) & (WDT[col] == 0)
                    ).to_numpy()
                    if num_model is None or num_mask.sum() == 0:
                        continue
                    subset = WDT.filter(pl.Series(num_mask))
                    p = _predict_model(self, num_model, subset)
                    pred_num = np.ones(WDT.height)
                    pred_num[num_mask] = _extract_class_probability(
                        p, num_model, 1, is_binary
                    )
                    WDT = WDT.with_columns(
                        pl.when(pl.Series(num_mask))
                        .then(pl.Series(pred_num))
                        .otherwise(pl.col("numerator"))
                        .alias("numerator")
                    )

                # As for the denominator, the model predicts P(censored)
                WDT = WDT.with_columns(
                    pl.when(pl.col("switch") == 0)
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
