import copy
import re

import numpy as np
import pandas as pd
import polars as pl
from scipy.special import expit, softmax

from ..helpers._fix_categories import _cast_fixed_categories
from ..weighting import _fit_denominator, _fit_LTFU, _fit_numerator, _fit_visit
from ..weighting._weight_stats import _weight_bounds
from ._fit import _class_names
from ._frame import _WEIGHT_MODELS
from ._models import _Model

# max log weight gap to the package
_WEIGHT_TOLERANCE = 1e-8

_NUMBER = re.compile(r"-?\d+\.\d*(?:e-?\d+)?")


def _column_key(name):
    return _NUMBER.sub(lambda m: f"{float(m.group()):.10g}", name)


def _aligned(params, columns):
    keys = {_column_key(c): c for c in params.index}
    wanted = [_column_key(c) for c in columns]
    if len(params) != len(columns) or set(keys) != set(wanted):
        return None
    return params[[keys[k] for k in wanted]]


def _rhs(spec, i=None):
    rhs = spec[i] if isinstance(spec, (list, tuple)) else spec
    return "1" if rhs in (None, "", "1") else rhs


def _weight_models(self, formulas):
    id_col = self.id_col
    switch = self.excused and not self.weight_preexpansion
    tx_bas = f"{self.treatment_col}{self.indicator_baseline}"

    def eligible(col):
        return None if col is None else (lambda f: f[col].to_numpy() == 1)

    def treatment_rows(i, level, numerator=False):
        col = (self.weight_eligible_colnames or [None] * (i + 1))[i]
        excused = self.excused_colnames[i] if self.excused else None

        def rows(f):
            keep = np.ones(len(f), bool)
            if self.weight_preexpansion:
                keep = f.groupby(id_col, sort=False).cumcount().to_numpy() > 0
            if excused is not None:
                keep &= f[excused].to_numpy() == 0
            if self.weight_lag_condition:
                lag = tx_bas if switch and numerator else "tx_lag"
                keep &= f[lag].to_numpy() == level
            if switch and not numerator:
                keep &= f["followup"].to_numpy() != 0
            if col is not None:
                keep &= f[col].to_numpy() == 1
            return keep

        return rows

    models = []
    if self.cense_colname is not None:
        rows = eligible(self.cense_eligible_colname)
        models += [
            _Model(
                "cense_numerator",
                self.cense_colname,
                _rhs(formulas["cense_numerator"]),
                rows,
            ),
            _Model(
                "cense_denominator",
                self.cense_colname,
                _rhs(formulas["cense_denominator"]),
                rows,
            ),
        ]
    if self.visit_colname is not None:
        models += [
            _Model(
                "visit_numerator", self.visit_colname, _rhs(formulas["cense_numerator"])
            ),
            _Model(
                "visit_denominator",
                self.visit_colname,
                _rhs(formulas["cense_denominator"]),
            ),
        ]
    if self.method != "ITT":
        outcome = "switch" if switch else self.treatment_col
        for i, level in enumerate(self.treatment_level):
            if self.weight_stabilized and not (
                self.excused and self.weight_preexpansion
            ):
                models.append(
                    _Model(
                        f"numerator_{i}",
                        outcome,
                        _rhs(formulas["numerator"], i),
                        treatment_rows(i, level, numerator=True),
                    )
                )
            models.append(
                _Model(
                    f"denominator_{i}",
                    outcome,
                    _rhs(formulas["denominator"], i),
                    treatment_rows(i, level),
                )
            )
    return models


def _fit_gold_weights(self, formulas, frame):
    f = copy.copy(self)
    for name, formula in formulas.items():
        setattr(f, name, formula)
    f._current_boot_idx = None
    WDT_pd = _cast_fixed_categories(f.fixed_cols, frame)
    _fit_LTFU(f, WDT_pd)
    _fit_visit(f, WDT_pd)
    tx_WDT_pd = WDT_pd
    if f.weight_preexpansion:
        tx_WDT_pd = WDT_pd[WDT_pd.groupby(f.id_col).cumcount() > 0]
    _fit_numerator(f, tx_WDT_pd)
    _fit_denominator(f, tx_WDT_pd)
    for name in _WEIGHT_MODELS:
        if hasattr(f, name):
            setattr(self, name, getattr(f, name))


def _package_model(self, name):
    if name.startswith(("numerator_", "denominator_")):
        kind, i = name.rsplit("_", 1)
        models = getattr(self, f"{kind}_model", None)
        model = models[int(i)] if models else None
    else:
        model = getattr(self, f"{name}_model", None)
    return None if model is None else self._offloader.load_model(model)


def _classes(self, name):
    # outcome values of a multinomial weight model or None for a 0 1 logit
    classes = getattr(_package_model(self, name), "seq_classes", None)
    if classes is None or [int(c) for c in classes] == [0, 1]:
        return None
    return [c.item() if hasattr(c, "item") else c for c in classes]


def _package_params(self, name):
    model = _package_model(self, name)
    if model is None:
        return None
    params = model.params
    if isinstance(params, pd.DataFrame):
        classes = _classes(self, name)
        if classes is None:
            # binary mnlogit has one column
            return params.iloc[:, 0]
        # stacked class by class as the condensed fit names them
        return pd.Series(
            params.to_numpy().T.ravel(), index=_class_names(params.index, classes)
        )
    return params


class _Theta:
    def __init__(self, block):
        self.names = [m.name for m in block.models]
        self.columns = [
            _class_names([block.columns[c] for c in cols], m.classes)
            for cols, m in zip(block.model_cols, block.models)
        ]
        sizes = [len(c) for c in self.columns]
        self.offsets = np.r_[0, np.cumsum(sizes)]

    @property
    def size(self):
        return int(self.offsets[-1])

    def slice(self, j):
        return slice(self.offsets[j], self.offsets[j + 1])

    def stack(self, params):
        theta = np.zeros(self.size)
        for j, name in enumerate(self.names):
            p = params[name]
            if isinstance(p, pd.Series):
                q = _aligned(p, self.columns[j])
                if q is None:
                    raise ValueError(
                        f"The {name} model's columns {list(p.index)} are not the "
                        f"condensed ones {self.columns[j]}"
                    )
                p = q.to_numpy(float)
            theta[self.slice(j)] = p
        return theta

    def apply(self, fitted, theta):
        for name in ("numerator_model", "denominator_model"):
            if getattr(fitted, name, None) is not None:
                setattr(fitted, name, list(getattr(fitted, name)))
        for j, name in enumerate(self.names):
            model = copy.deepcopy(_package_model(fitted, name))
            p = theta[self.slice(j)]
            keys = [_column_key(c) for c in self.columns[j]]
            order = [
                keys.index(_column_key(c)) for c in _package_params(fitted, name).index
            ]
            shape = np.shape(model._results.params)
            if len(shape) == 2:
                # mnlogit holds columns by classes
                model._results.params = p[order].reshape(shape[1], shape[0]).T
            else:
                model._results.params = p[order]
            if name.startswith(("numerator_", "denominator_")):
                kind, i = name.rsplit("_", 1)
                getattr(fitted, f"{kind}_model")[int(i)] = model
            else:
                setattr(fitted, f"{name}_model", model)

    def of(self, fitted):
        return self.stack({name: _package_params(fitted, name) for name in self.names})


def _record_contributions(self, block, theta, frame):
    layout = _Theta(block)
    frame = block.prepare(frame)
    n = len(frame)
    step, ratio = np.zeros(n), np.zeros(n)
    d_step, d_ratio = np.zeros((n, layout.size)), np.zeros((n, layout.size))
    names = layout.names
    tx_lag = frame["tx_lag"].to_numpy()
    for j, name in enumerate(names):
        pos, Z = block.design(frame, j)
        p = expit(Z @ theta[layout.slice(j)][: Z.shape[1]])
        kind, _, part = name.partition("_")
        sign = 1.0 if kind == "numerator" or part == "numerator" else -1.0
        if kind in ("numerator", "denominator"):
            # rows whose previous treatment is the level
            level = self.treatment_level[int(part)]
            on = tx_lag[pos] == level
            modelled = self.treatment_col
            if self.excused:
                # excused rows and levels without a column carry no weight
                col = self.excused_colnames[int(part)]
                if col is None:
                    continue
                excused = frame[col].to_numpy()[pos]
                on &= excused != 1
                if not self.weight_preexpansion:
                    # post expansion excused models are of the switch
                    modelled = "switch"
                    if kind == "numerator":
                        current = frame[self.treatment_col].to_numpy()[pos]
                        on = (current == level) & (excused == 0)
            pos, Z, p = pos[on], Z[on], p[on]
            y = frame[modelled].to_numpy(float)[pos]
            classes = block.models[j].classes
            if classes is None:
                step[pos] += sign * np.where(y == 1, np.log(p), np.log1p(-p))
                d_step[pos, layout.slice(j)] = sign * (y - p)[:, None] * Z
                continue
            # multinomial weight is P of the level if stayed else 1 minus it
            m = len(classes) - 1
            gamma = theta[layout.slice(j)].reshape(m, Z.shape[1])
            P = softmax(np.hstack([np.zeros((len(Z), 1)), Z @ gamma.T]), axis=1)
            at = classes.index(level)
            stay, pl_ = y == level, P[:, at]
            step[pos] += sign * np.where(stay, np.log(pl_), np.log1p(-pl_))
            lead = np.where(stay, 1.0, -pl_ / (1 - pl_))
            a = (np.eye(m + 1)[at][1:] - P[:, 1:]) * lead[:, None]
            d_step[pos, layout.slice(j)] = sign * (
                a[:, :, None] * Z[:, None, :]
            ).reshape(len(Z), -1)
        elif kind == "cense":
            # weight uses P remaining and needs both models
            if f"cense_{'denominator' if sign > 0 else 'numerator'}" not in names:
                continue
            ratio[pos] += sign * np.log1p(-p)
            d_ratio[pos, layout.slice(j)] = -sign * p[:, None] * Z
        else:
            if f"visit_{'denominator' if sign > 0 else 'numerator'}" not in names:
                continue
            ratio[pos] += sign * np.log(p)
            d_ratio[pos, layout.slice(j)] = sign * (1 - p)[:, None] * Z
    return step, ratio, d_step, d_ratio


def _attach_sensitivities(self, block, theta, frame, b):
    id_col = self.id_col
    step, ratio, d_step, d_ratio = _record_contributions(self, block, theta, frame)
    D = d_step.shape[1]
    # record rows join by period and expanded rows by trial and followup
    if self.weight_preexpansion:
        keys = {id_col: frame[id_col], "period": frame[self.time_col]}
    else:
        keys = {col: frame[col] for col in (id_col, "trial", "followup")}
    parts = pl.DataFrame(
        {
            **{col: values.to_numpy() for col, values in keys.items()},
            "_step": step,
            "_ratio": ratio,
        }
    ).hstack(
        pl.DataFrame(
            np.hstack([d_step, d_ratio]),
            schema=[f"_ds{j}" for j in range(D)] + [f"_dr{j}" for j in range(D)],
        )
    )
    filled = ["_step", "_ratio"] + [f"_d{kind}{j}" for kind in "sr" for j in range(D)]
    DT = b.DT.join(
        parts.with_columns([pl.col(col).cast(b.DT.schema[col]) for col in keys]),
        on=list(keys),
        how="left",
        maintain_order="left",
        # rows without a frame row carry no step
    ).with_columns([pl.col(col).fill_null(0.0) for col in filled])

    # _weight_bind override where the step is 1
    trial_start = (pl.col("trial") == pl.col("trial").min().over(id_col)) & (
        pl.col("followup") == 0
    )
    visit = (
        pl.col(self.visit_colname) == 0
        if self.visit_colname is not None
        else pl.lit(False)
    )
    override = (
        trial_start
        | visit
        | pl.col(self.outcome_col).is_null()
        | (pl.col("denominator") < 1e-15)
        | pl.col("numerator").is_null()
    )
    if self.excused:
        # excused override from the switch to the trial end
        DT = DT.with_columns(
            (
                pl.col("isExcused").fill_null(False).cum_sum().over([id_col, "trial"])
                > 0
            ).alias("_excused")
        )
        if self.weight_preexpansion:
            first = (pl.col("trial") == 0) & (pl.col("period") == 0)
            degenerate = pl.col("denominator") < 1e-7
        else:
            # post expansion override also covers zero numerators
            first = pl.col("followup") == 0
            degenerate = (pl.col("denominator") < 1e-7) | (pl.col("numerator") < 1e-7)
        override = (
            first
            | pl.col("_excused")
            | visit
            | pl.col(self.outcome_col).is_null()
            | degenerate
        )

    def along(col):
        return (
            pl.when(pl.col("_override"))
            .then(0.0)
            .otherwise(pl.col(col))
            .cum_sum()
            .over([id_col, "trial"])
        )

    # override computed first as nested windows misbehave
    DT = DT.with_columns(override.alias("_override")).with_columns(
        (along("_step") + pl.col("_ratio")).alias("_logw"),
        *[(along(f"_ds{j}") + pl.col(f"_dr{j}")).alias(f"_s{j}") for j in range(D)],
    )
    check = (
        DT.filter(pl.col("weight") > 0)
        .select((pl.col("_logw") - pl.col("weight").log()).abs().max())
        .item()
    )
    if check is None or not check <= _WEIGHT_TOLERANCE:
        raise RuntimeError(
            "The weights rebuilt from the weight models' coefficients are not "
            f"the package's (largest log difference {check})"
        )

    lower, upper = _weight_bounds(self)
    clipped = pl.col("weight") < lower
    if upper is not None:
        clipped |= pl.col("weight") > upper
    s_cols = [f"_s{j}" for j in range(D)]
    helpers = ["_step", "_ratio", "_override", "_logw"]
    if self.excused:
        helpers.append("_excused")
    helpers += [f"_d{kind}{j}" for kind in "sr" for j in range(D)]
    b.DT = DT.with_columns(
        [pl.when(clipped).then(0.0).otherwise(pl.col(c)).alias(c) for c in s_cols]
    ).drop(helpers)
    return s_cols


class _Tally:
    # streaming weight statistics with quantiles from a fine log histogram
    def __init__(self):
        self.edges = np.linspace(-12, 12, 100_001)
        self.hist = np.zeros(100_000)
        self.n = self.s = self.ss = 0.0
        self.lo, self.hi = np.inf, -np.inf

    def add(self, w):
        w = np.asarray(w, float)
        self.n += len(w)
        self.s += w.sum()
        self.ss += (w**2).sum()
        self.lo, self.hi = min(self.lo, w.min()), max(self.hi, w.max())
        self.hist += np.histogram(np.log10(np.clip(w, 1e-12, 1e12)), self.edges)[0]

    def quantile(self, q):
        i = min(np.searchsorted(np.cumsum(self.hist), q * self.n), len(self.hist) - 1)
        mid = 10 ** ((self.edges[i] + self.edges[i + 1]) / 2)
        return float(np.clip(mid, self.lo, self.hi))

    def frame(self):
        # same columns as _weight_stats
        var = (self.ss - self.s**2 / self.n) / max(self.n - 1, 1)
        return pl.DataFrame(
            {
                "weight_min": [float(self.lo)],
                "weight_max": [float(self.hi)],
                "weight_mean": [self.s / self.n],
                "weight_std": [float(np.sqrt(max(var, 0.0)))],
                "weight_p01": [self.quantile(0.01)],
                "weight_p25": [self.quantile(0.25)],
                "weight_p50": [self.quantile(0.50)],
                "weight_p75": [self.quantile(0.75)],
                "weight_p99": [self.quantile(0.99)],
            }
        )
