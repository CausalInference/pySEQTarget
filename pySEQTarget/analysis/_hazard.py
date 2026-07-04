import copy
import warnings
from concurrent.futures import ProcessPoolExecutor

import numpy as np
import polars as pl
from lifelines import CoxPHFitter

from ..helpers._predict_model import _safe_predict
from ._outcome_fit import _cast_categories


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
    if self.seed is not None:
        self._rng = np.random.RandomState(self.seed)
    full_log_hr = _hazard_handler(self, data, idx, 0, self._rng)

    if full_log_hr is None or np.isnan(full_log_hr):
        return _create_hazard_output(None, None, None, val, self)

    if self.bootstrap_nboot > 0:
        # outcome_model[model_pos + 1] was fit on _boot_samples[sample_idx];
        # skipped replicates make this mapping non-identity, so iterate it
        # explicitly rather than assuming model index == sample index.
        boot_sample_idx = getattr(self, "_boot_sample_idx", None)
        if boot_sample_idx is None:
            boot_sample_idx = list(range(len(self._boot_samples)))

        # The per-replicate hazard simulation is GIL-bound (patsy design build),
        # so spread it over a process pool when parallel=True. Needs a concrete
        # seed (always set since SEQuential pins a default) so each replicate's
        # RNG — and therefore the result — is identical to the serial path.
        if getattr(self, "parallel", False) and self.seed is not None:
            boot_log_hrs = _parallel_boot_log_hrs(self, data, idx, boot_sample_idx)
        else:
            boot_log_hrs = []
            for model_pos, sample_idx in enumerate(boot_sample_idx):
                boot_log_hr = _one_boot_log_hr(self, data, idx, model_pos, sample_idx)
                if boot_log_hr is not None and not np.isnan(boot_log_hr):
                    boot_log_hrs.append(boot_log_hr)

        if len(boot_log_hrs) == 0:
            return _create_hazard_output(np.exp(full_log_hr), None, None, val, self)

        if self.bootstrap_CI_method == "se":
            from scipy.stats import norm

            z = norm.ppf(1 - (1 - self.bootstrap_CI) / 2)
            se = np.std(boot_log_hrs)
            lci = np.exp(full_log_hr - z * se)
            uci = np.exp(full_log_hr + z * se)
        else:
            lci = np.exp(np.quantile(boot_log_hrs, (1 - self.bootstrap_CI) / 2))
            uci = np.exp(np.quantile(boot_log_hrs, 1 - (1 - self.bootstrap_CI) / 2))
    else:
        lci, uci = None, None

    return _create_hazard_output(np.exp(full_log_hr), lci, uci, val, self)


def _truncate_to_first_event(tmp, id_col, event_col):
    """Reduce a simulated counterfactual grid to one survival row per (id, trial).

    Keeps the FIRST row in which ``event_col`` fires (status 1 at the first event
    time); if the unit never has an event it keeps the final follow-up row
    (status 0, censored at max follow-up).

    Outcomes are simulated independently at every follow-up row, so a unit may
    have ``event_col == 1`` at several rows. We therefore keep only rows whose
    cumulative event count *strictly before* the current row is 0 — i.e. every
    row up to and including the first event — and then take the last of those,
    which is the first-event row (or the max-follow-up row when there is no
    event).

    NOTE: the inclusive form ``cum_sum(event_col) <= 1`` is WRONG here: it
    retains post-event rows (the cumulative count stays at 1 until a second
    event), so ``.last()`` returns the final follow-up row and a single event is
    silently recorded as censored. That dropped ~99% of simulated events and
    inflated the marginal-HR variance ~8x relative to SEQTaRget (R). See
    tests/test_hazard_truncation.py.
    """
    return (
        tmp.with_columns(
            (
                pl.col(event_col).cum_sum().over([id_col, "trial"]) - pl.col(event_col)
            ).alias("_event_prior")
        )
        .filter(pl.col("_event_prior") == 0)
        .group_by([id_col, "trial"])
        .last()
        .drop("_event_prior")
    )


def _one_boot_log_hr(self, data, idx, model_pos, sample_idx):
    """Build one bootstrap resample of ``data`` and return its log hazard ratio.

    The RNG is rebuilt from ``seed + sample_idx + 1`` (matching the serial loop
    exactly), so this is bit-identical whether called serially or in a worker.
    """
    seed = getattr(self, "seed", None)
    rng = (
        np.random.RandomState(seed + sample_idx + 1) if seed is not None else self._rng
    )

    id_counts = self._boot_samples[sample_idx]
    counts = pl.DataFrame(
        {
            self.id_col: list(id_counts.keys()),
            "_count": list(id_counts.values()),
        }
    )
    boot_data = (
        data.lazy()
        .join(counts.lazy(), on=self.id_col, how="inner")
        .with_columns(pl.int_ranges(0, pl.col("_count")).alias("_rep"))
        .explode("_rep", empty_as_null=True)
        .drop("_count", "_rep")
        .collect()
    )
    return _hazard_handler(self, boot_data, idx, model_pos + 1, rng)


# Process-pool worker state. Set once per worker process by the initializer so
# each task ships only small integers, not the (slimmed) SEQuential object or
# the analysis frame.
_HZ_WORKER_OBJ = None
_HZ_WORKER_DATA = None


def _hazard_pool_init(obj, data_ref):
    global _HZ_WORKER_OBJ, _HZ_WORKER_DATA
    _HZ_WORKER_OBJ = obj
    _HZ_WORKER_DATA = obj._offloader.load_dataframe(data_ref)


def _hazard_pool_task(idx, model_pos, sample_idx):
    return _one_boot_log_hr(_HZ_WORKER_OBJ, _HZ_WORKER_DATA, idx, model_pos, sample_idx)


def _parallel_boot_log_hrs(self, data, idx, boot_sample_idx):
    """Run the bootstrap hazard simulations over a process pool.

    The analysis frame is handed to each worker process once (via the offloader
    ref + pool initializer), and a slimmed copy of ``self`` carries the fitted
    models. Results are gathered in submission order, matching the serial loop;
    NaN/None replicates are dropped the same way.
    """
    data_ref = self._offloader.save_dataframe(data, f"_haz_DT_{idx}")

    # Slim copy for the pool: drop the large frames workers reload from data_ref;
    # keep the fitted models, bootstrap samples, and config. _GlumFit and
    # SEQuential each drop their unpicklable / heavy state on pickle.
    slim = copy.copy(self)
    slim.DT = None
    slim.data = None
    slim._rng = None

    boot_log_hrs = []
    with ProcessPoolExecutor(
        max_workers=self.ncores,
        initializer=_hazard_pool_init,
        initargs=(slim, data_ref),
    ) as executor:
        futures = [
            executor.submit(_hazard_pool_task, idx, model_pos, sample_idx)
            for model_pos, sample_idx in enumerate(boot_sample_idx)
        ]
        for future in futures:
            result = future.result()
            if result is not None and not np.isnan(result):
                boot_log_hrs.append(result)

    return boot_log_hrs


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
        .sort([self.id_col, "trial"])
        .with_columns([pl.lit(list(range(self.followup_max + 1))).alias("followup")])
        .explode("followup", empty_as_null=True)
        .with_columns(
            [(pl.col("followup") ** 2).alias(f"followup{self.indicator_squared}")]
        )
    )

    if idx is not None:
        model_dict = self.outcome_model[boot_idx][idx]
    else:
        model_dict = self.outcome_model[boot_idx]

    outcome_model = self._offloader.load_model(model_dict["outcome"])
    ce_model = None
    if self.compevent_colname and "compevent" in model_dict:
        ce_model = self._offloader.load_model(model_dict["compevent"])

    all_treatments = []
    for val in self.treatment_level:
        tmp = trials.with_columns(
            [pl.lit(val).alias(f"{self.treatment_col}{self.indicator_baseline}")]
        )

        tmp_pd = _cast_categories(self, tmp.to_pandas())
        outcome_prob = _safe_predict(outcome_model, tmp_pd)
        outcome_sim = rng.binomial(1, outcome_prob)

        tmp = tmp.with_columns([pl.Series("outcome", outcome_sim)])

        if ce_model is not None:
            ce_tmp_pd = _cast_categories(self, tmp.to_pandas())
            ce_prob = _safe_predict(ce_model, ce_tmp_pd)
            ce_sim = rng.binomial(1, ce_prob)
            tmp = tmp.with_columns([pl.Series("ce", ce_sim)])

            tmp = tmp.with_columns(
                [
                    pl.when((pl.col("outcome") == 1) | (pl.col("ce") == 1))
                    .then(1)
                    .otherwise(0)
                    .alias("any_event")
                ]
            )
            tmp = _truncate_to_first_event(tmp, self.id_col, "any_event")
        else:
            tmp = _truncate_to_first_event(tmp, self.id_col, "outcome")

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
    tx_bas = f"{self.treatment_col}{self.indicator_baseline}"

    try:
        # COXPHFITTER CURRENTLY HAS DEPRECATED datetime.datetime.utcnow()
        warnings.filterwarnings("ignore", message=".*datetime.datetime.utcnow.*")
        if ce_model is not None:
            cox_data = sim_data_pd[sim_data_pd["event"].isin([0, 1])].copy()
            cox_data["event_binary"] = (cox_data["event"] == 1).astype(int)
            return _cox_log_hr(self, cox_data, "followup", "event_binary", tx_bas)
        return _cox_log_hr(self, sim_data_pd, "followup", "event", tx_bas)
    except Exception as e:
        print(f"Cox model fitting failed: {e}")
        return None


def _cox_log_hr(self, data_pd, duration_col, event_col, covariate_col):
    """
    Fit a univariate Cox model (single covariate = baseline treatment) and
    return the log hazard ratio, dispatching on self.cox_package. scikit-survival
    uses Efron tie handling to match lifelines, which matters here because
    integer follow-up produces many tied event times.
    """
    if getattr(self, "cox_package", "lifelines") == "scikit-survival":
        from sksurv.linear_model import CoxPHSurvivalAnalysis

        y = np.empty(len(data_pd), dtype=[("event", bool), ("time", "float64")])
        y["event"] = data_pd[event_col].to_numpy().astype(bool)
        y["time"] = data_pd[duration_col].to_numpy().astype(float)
        X = data_pd[[covariate_col]].to_numpy().astype(float)
        cox = CoxPHSurvivalAnalysis(ties="efron")
        cox.fit(X, y)
        return float(cox.coef_[0])

    cph = CoxPHFitter()
    cph.fit(
        data_pd,
        duration_col=duration_col,
        event_col=event_col,
        formula=f"`{covariate_col}`",
    )
    return cph.params_.values[0]


def _create_hazard_output(hr, lci, uci, val, self):
    if lci is not None and uci is not None:
        output = pl.DataFrame(
            {
                "Hazard ratio": [hr if hr is not None else float("nan")],
                "LCI": [lci],
                "UCI": [uci],
            }
        )
    else:
        output = pl.DataFrame(
            {"Hazard ratio": [hr if hr is not None else float("nan")]}
        )

    if val is not None:
        output = output.with_columns(pl.lit(val).alias(self.subgroup_colname))

    return output
