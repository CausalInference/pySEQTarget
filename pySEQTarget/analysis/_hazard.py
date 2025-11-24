import warnings

import numpy as np
import polars as pl
from lifelines import CoxPHFitter


def _calculate_hazard(self):
    if self.subgroup_colname is None:
        return _calculate_hazard_single(self, self.DT, idx=None, val=None)

    all_hazards = []
    original_DT = self.DT

    for i, val in enumerate(self._unique_subgroups):
        subgroup_DT = original_DT.filter(pl.col(self.subgroup_colname) == val)
        hazard = _calculate_hazard_single(self, subgroup_DT, i, val)
        all_hazards.append(hazard)

    self.DT = original_DT
    return pl.concat(all_hazards)


def _calculate_hazard_single(self, data, idx=None, val=None):
    full_hr = _hazard_handler(self, data, idx, 0, self._rng)

    if full_hr is None or np.isnan(full_hr):
        return _create_hazard_output(None, None, None, val, self)

    if self.bootstrap_nboot > 0:
        boot_hrs = []

        for boot_idx in range(len(self._boot_samples)):
            id_counts = self._boot_samples[boot_idx]

            boot_data_list = []
            for id_val, count in id_counts.items():
                id_data = data.filter(pl.col(self.id_col) == id_val)
                for _ in range(count):
                    boot_data_list.append(id_data)

            boot_data = pl.concat(boot_data_list)

            boot_hr = _hazard_handler(self, boot_data, idx, boot_idx + 1, self._rng)
            if boot_hr is not None and not np.isnan(boot_hr):
                boot_hrs.append(boot_hr)

        if len(boot_hrs) == 0:
            return _create_hazard_output(full_hr, None, None, val, self)

        if self.bootstrap_CI_method == "se":
            from scipy.stats import norm

            z = norm.ppf(1 - (1 - self.bootstrap_CI) / 2)
            se = np.std(boot_hrs)
            lci = full_hr - z * se
            uci = full_hr + z * se
        else:
            lci = np.quantile(boot_hrs, (1 - self.bootstrap_CI) / 2)
            uci = np.quantile(boot_hrs, 1 - (1 - self.bootstrap_CI) / 2)
    else:
        lci, uci = None, None

    return _create_hazard_output(full_hr, lci, uci, val, self)


def _hazard_handler(self, data, idx, boot_idx, rng):
    exclude_cols = [
        "followup",
        f"followup{self.indicator_squared}",
        self.treatment_col,
        f"{self.treatment_col}{self.indicator_baseline}",
        "period",
        self.outcome_col,
    ]
    if self.compevent_colname:
        exclude_cols.append(self.compevent_colname)
    keep_cols = [col for col in data.columns if col not in exclude_cols]

    trials = (
        data.select(keep_cols)
        .group_by([self.id_col, "trial"])
        .first()
        .with_columns([pl.lit(list(range(self.followup_max + 1))).alias("followup")])
        .explode("followup")
        .with_columns(
            [(pl.col("followup") ** 2).alias(f"followup{self.indicator_squared}")]
        )
    )

    if idx is not None:
        model_dict = self.outcome_model[boot_idx][idx]
    else:
        model_dict = self.outcome_model[boot_idx]

    outcome_model = model_dict["outcome"]
    ce_model = model_dict.get("compevent", None) if self.compevent_colname else None

    all_treatments = []
    for val in self.treatment_level:
        tmp = trials.with_columns(
            [pl.lit(val).alias(f"{self.treatment_col}{self.indicator_baseline}")]
        )

        tmp_pd = tmp.to_pandas()
        outcome_prob = outcome_model.predict(tmp_pd)
        outcome_sim = rng.binomial(1, outcome_prob)

        tmp = tmp.with_columns([pl.Series("outcome", outcome_sim)])

        if ce_model is not None:
            ce_prob = ce_model.predict(tmp_pd)
            ce_sim = rng.binomial(1, ce_prob)
            tmp = tmp.with_columns([pl.Series("ce", ce_sim)])

            tmp = (
                tmp.with_columns(
                    [
                        pl.when((pl.col("outcome") == 1) | (pl.col("ce") == 1))
                        .then(1)
                        .otherwise(0)
                        .alias("any_event")
                    ]
                )
                .with_columns(
                    [
                        pl.col("any_event")
                        .cum_sum()
                        .over([self.id_col, "trial"])
                        .alias("event_cumsum")
                    ]
                )
                .filter(pl.col("event_cumsum") <= 1)
            )
        else:
            tmp = tmp.with_columns(
                [
                    pl.col("outcome")
                    .cum_sum()
                    .over([self.id_col, "trial"])
                    .alias("event_cumsum")
                ]
            ).filter(pl.col("event_cumsum") <= 1)

        tmp = tmp.group_by([self.id_col, "trial"]).last()
        all_treatments.append(tmp)

    sim_data = pl.concat(all_treatments)

    if ce_model is not None:
        sim_data = sim_data.with_columns(
            [
                pl.when(pl.col("outcome") == 1)
                .then(pl.lit(1))
                .when(pl.col("ce") == 1)
                .then(pl.lit(2))
                .otherwise(pl.lit(0))
                .alias("event")
            ]
        )
    else:
        sim_data = sim_data.with_columns([pl.col("outcome").alias("event")])

    sim_data_pd = sim_data.to_pandas()

    try:
        # COXPHFITER CURRENTLY HAS DEPRECATED datetime.datetime.utcnow()
        warnings.filterwarnings("ignore", message=".*datetime.datetime.utcnow.*")
        if ce_model is not None:
            cox_data = sim_data_pd[sim_data_pd["event"].isin([0, 1])].copy()
            cox_data["event_binary"] = (cox_data["event"] == 1).astype(int)

            cph = CoxPHFitter()
            cph.fit(
                cox_data,
                duration_col="followup",
                event_col="event_binary",
                formula=f"`{self.treatment_col}{self.indicator_baseline}`",
            )
        else:
            cph = CoxPHFitter()
            cph.fit(
                sim_data_pd,
                duration_col="followup",
                event_col="event",
                formula=f"`{self.treatment_col}{self.indicator_baseline}`",
            )

        hr = np.exp(cph.params_.values[0])
        return hr
    except Exception as e:
        print(f"Cox model fitting failed: {e}")
        return None


def _create_hazard_output(hr, lci, uci, val, self):
    if lci is not None and uci is not None:
        output = pl.DataFrame(
            {
                "Hazard": [hr if hr is not None else float("nan")],
                "LCI": [lci],
                "UCI": [uci],
            }
        )
    else:
        output = pl.DataFrame({"Hazard": [hr if hr is not None else float("nan")]})

    if val is not None:
        output = output.with_columns(pl.lit(val).alias(self.subgroup_colname))

    return output
