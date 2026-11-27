import copy
from types import SimpleNamespace

import numpy as np
import polars as pl
from scipy.special import expit
from scipy.stats import norm

from ..analysis._outcome_fit import _cast_categories
from ..analysis._risk_estimates import _risk_estimates
from ..expansion._selection import _selection_weight
from ..helpers._bootstrap import _rescale_boot

# max hazard cells held per chunk
_CHUNK_CELLS = 2e7


def _baseline_rows(self, b):
    return b.DT.group_by([self.id_col, "trial"], maintain_order=True).first()


def _replicates(c, nboot, sample, seed):
    rng = np.random.default_rng(seed)
    size = int(sample * len(c.cohort))
    out = []
    for _ in range(nboot):
        ids, n = np.unique(rng.choice(c.cohort, size=size), return_counts=True)
        counts = dict(zip(ids.tolist(), n.tolist()))
        fits = {m: c.fit(m, counts=counts).params.to_numpy() for m in c.outcome.models}
        out.append((counts, fits))
    return out


def _grid(a, chunk, level, dose):
    sq = f"followup{a.indicator_squared}"
    grid = (
        chunk.drop([col for col in ("followup", sq) if col in chunk.columns])
        .with_columns(pl.lit(list(range(a.followup_max + 1))).alias("followup"))
        .explode("followup")
        .with_columns(
            (pl.col("followup") ** 2).alias(sq),
            pl.lit(level).alias(f"{a.treatment_col}{a.indicator_baseline}"),
        )
    )
    if dose:
        if level == a.treatment_level[0]:
            grid = grid.with_columns(
                pl.lit(0.0).alias("dose"), pl.lit(0.0).alias("dose_sq")
            )
        else:
            grid = grid.with_columns(
                pl.col("followup").alias("dose"), pl.col(sq).alias("dose_sq")
            )
    return grid


def _groups(c):
    a = c.analysis
    roles = ["outcome"] + (["compevent"] if a.compevent_colname is not None else [])
    if a.subgroup_colname is None:
        yield None, c.baseline, {role: role for role in roles}
        return
    for value in c.census.subgroups:
        yield (
            value,
            c.baseline.filter(pl.col(a.subgroup_colname) == value),
            {role: f"{role}[{value}]" for role in roles},
        )


def _stacked(names, main, replicates):
    return {
        role: np.vstack([main[name]] + [fits[name] for _, fits in replicates])
        for role, name in names.items()
    }


def _selection_weights(a, chunk):
    # each kept trial stands for the trials selection_random dropped, as in the
    # package; taken from the observed baseline treatment before the grid sets it
    sel = _selection_weight(a)
    if sel is None:
        return np.ones(chunk.height)
    return chunk.select(sel).to_series().to_numpy()


def _hazards(c, baseline, names, coefficients, dose):
    a = copy.copy(c.analysis)
    # stop _cast_categories recording chunk levels
    a._current_boot_idx = 0
    block = c.outcome.block
    F = a.followup_max + 1
    R = next(iter(coefficients.values())).shape[0]
    step = max(1, int(_CHUNK_CELLS / (F * R)))
    for start in range(0, baseline.height, step):
        chunk = baseline.slice(start, step)
        ids = chunk[a.id_col].to_numpy()
        w = _selection_weights(a, chunk)
        for level in a.treatment_level:
            frame = block.prepare(
                _cast_categories(a, _grid(a, chunk, level, dose).to_pandas())
            )
            out, designs = {}, {}
            for model, B in coefficients.items():
                j = c.outcome.models.index(names[model])
                rhs = block.models[j].rhs
                if rhs not in designs:
                    pos, X = block.design(frame, j)
                    if len(pos) != len(frame):
                        raise ValueError(
                            "Some trials have missing covariates at baseline; "
                            "their hazards cannot be predicted"
                        )
                    designs[rhs] = X
                out[model] = expit(designs[rhs] @ B.T).reshape(len(chunk), F, R)
            yield ids, w, level, out


def _curves(c, baseline, names, coefficients):
    dose = c.analysis.method == "dose-response"
    sums, n = {}, 0
    for ids, w, level, h in _hazards(c, baseline, names, coefficients, dose):
        ho = h["outcome"]
        acc = sums.setdefault(level, {})
        surv = np.einsum("i,ifr->fr", w, np.cumprod(1 - ho, axis=1))
        acc["surv"] = acc.get("surv", 0) + surv
        if "compevent" in h:
            hc = h["compevent"]
            cce = np.cumprod((1 - ho) * (1 - hc), axis=1)
            inc = np.cumsum(ho * (1 - hc) * cce, axis=1)
            acc["inc"] = acc.get("inc", 0) + np.einsum("i,ifr->fr", w, inc)
        if level == c.analysis.treatment_level[0]:
            n += w.sum()
    return {
        level: {key: total / n for key, total in acc.items()}
        for level, acc in sums.items()
    }


def _interval(point, boot, level, method, sample):
    alpha = 1 - level
    boot = _rescale_boot(boot, point[:, None], sample)
    se = boot.std(axis=1, ddof=1)
    if method == "se":
        z = norm.ppf(1 - alpha / 2)
        return se, point - z * se, point + z * se
    lo = [pl.Series(row).quantile(alpha / 2) for row in boot]
    hi = [pl.Series(row).quantile(1 - alpha / 2) for row in boot]
    return se, np.array(lo), np.array(hi)


def _survival(c, main, replicates, CI, CI_method, risk_times, sample):
    a = c.analysis
    nboot = len(replicates)
    followup = np.arange(a.followup_max + 2)

    def table(level, values, estimate, start, subgroup):
        # followup k means k elapsed intervals
        values = np.vstack([np.full((1, values.shape[1]), start), values])
        cols = {"followup": followup, "pred": values[:, 0]}
        if nboot:
            se, lo, hi = _interval(values[:, 0], values[:, 1:], CI, CI_method, sample)
            cols |= {"SE": se, "LCI": lo, "UCI": hi}
        out = pl.DataFrame(cols).with_columns(
            pl.lit(level).cast(pl.Int32).alias(a.treatment_col),
            pl.lit(estimate).alias("estimate"),
        )
        if subgroup is not None:
            out = out.with_columns(pl.lit(subgroup).alias(a.subgroup_colname))
        return out

    parts, boot_risks = [], {}
    for subgroup, baseline, names in _groups(c):
        curves = _curves(c, baseline, names, _stacked(names, main, replicates))
        for level in a.treatment_level:
            surv = curves[level]["surv"]
            parts.append(table(level, 1 - surv, "risk", 0.0, subgroup))
            parts.append(table(level, surv, "survival", 1.0, subgroup))
            if "inc" in curves[level]:
                parts.append(
                    table(level, curves[level]["inc"], "incidence", 0.0, subgroup)
                )
            if nboot and subgroup is None:
                risk = 1 - surv[:, 1:]
                boot_risks[level] = pl.DataFrame(
                    {
                        "followup": np.repeat(followup[1:], nboot),
                        "risk": risk.ravel(),
                        "boot_idx": np.tile(np.arange(1, nboot + 1), len(risk)),
                    }
                ).with_columns(pl.col("boot_idx").cast(pl.Int32))
    km = pl.concat(parts).with_columns(
        [
            pl.col(col).clip(0.0, 1.0)
            for col in ("pred", "LCI", "UCI")
            if col in parts[0].columns
        ]
    )

    stub = SimpleNamespace(
        km_data=km,
        risk_times=risk_times,
        subgroup_colname=a.subgroup_colname,
        bootstrap_nboot=nboot,
        bootstrap_CI=CI,
        bootstrap_CI_method=CI_method,
        bootstrap_sample=sample,
        treatment_level=a.treatment_level,
        treatment_col=a.treatment_col,
    )
    # paired contrasts for the whole cohort only
    if boot_risks:
        stub._boot_risks = boot_risks
    return {"km_data": km, **_risk_estimates(stub)}


def _cox(events, at_risk, values, counts=None, tol=1e-12, max_iter=50):
    # cox fit on the arm value as the package codes it; events and at_risk may
    # be weighted, with counts the number of events, for lifelines' weighted
    # Efron ties
    counts = events if counts is None else counts
    times = np.flatnonzero(events.sum(0) > 0)
    if len(times) == 0 or (events.sum(1) > 0).sum() < 2:
        return float("nan")
    # one term per event with its rank in the tie
    d = np.rint(counts[:, times].sum(0)).astype(int)
    t = np.repeat(np.arange(len(times)), d)
    # each term carries the tie's mean event weight
    mean_w = np.repeat(events[:, times].sum(0) / d, d)
    frac = (np.arange(d.sum()) - np.repeat(np.cumsum(d) - d, d)) / np.repeat(d, d)
    # arms by terms still at risk
    A = at_risk[:, times][:, t] - frac * events[:, times][:, t]
    x = np.asarray(values, float)[:, None]
    total = (x[:, 0] * events.sum(1)).sum()
    beta = 0.0
    for _ in range(max_iter):
        R = np.exp(beta * x) * A
        s0, s1, s2 = R.sum(0), (x * R).sum(0), (x**2 * R).sum(0)
        g = total - np.sum(mean_w * s1 / s0)
        H = np.sum(mean_w * (s2 / s0 - (s1 / s0) ** 2))
        step = g / H
        beta += step
        if abs(step) < tol:
            break
    return float(beta)


def _simulate(h, hc, rng):
    any_h = h if hc is None else 1 - (1 - h) * (1 - hc)
    alive = np.cumprod(1 - any_h, axis=1)
    time = (alive >= rng.random((len(h), 1))).sum(1)
    F = h.shape[1]
    none = time == F
    time = np.minimum(time, F - 1)
    rows = np.arange(len(h))
    status = np.ones(len(h), int)
    if hc is not None:
        p_outcome = h[rows, time] / np.maximum(any_h[rows, time], 1e-300)
        status[rng.random(len(h)) >= p_outcome] = 2
    status[none] = 0
    return time, status


def _hazard(c, main, replicates, CI, CI_method, seed, sample):
    a = c.analysis
    if a.method == "dose-response":
        raise NotImplementedError(
            "Hazard ratio estimation is not supported for method='dose-response'"
        )
    out = []
    for subgroup, baseline, names in _groups(c):
        row = _hazard_within(
            c, baseline, names, main, replicates, CI, CI_method, seed, sample
        )
        if subgroup is not None:
            row = row.with_columns(pl.lit(subgroup).alias(a.subgroup_colname))
        out.append(row)
    return pl.concat(out, how="diagonal")


def _hazard_within(c, baseline, names, main, replicates, CI, CI_method, seed, sample):
    a = c.analysis
    levels = list(a.treatment_level)
    coefficients = _stacked(names, main, replicates)
    R = len(replicates) + 1
    F = a.followup_max + 1
    seed = a.seed if seed is None else seed
    rngs = [np.random.default_rng([seed, r]) for r in range(R)]
    # replicate arm status time, weighted and counted
    hist = np.zeros((R, len(levels), 3, F))
    count = np.zeros((R, len(levels), 3, F))
    for ids, w, level, h in _hazards(c, baseline, names, coefficients, dose=False):
        arm = levels.index(level)
        for r in range(R):
            if r == 0:
                copies = np.arange(len(ids))
            else:
                counts = replicates[r - 1][0]
                copies = np.repeat(
                    np.arange(len(ids)), [counts.get(i, 0) for i in ids.tolist()]
                )
            if len(copies) == 0:
                continue
            ho = h["outcome"][copies, :, r]
            hc = h["compevent"][copies, :, r] if "compevent" in h else None
            time, status = _simulate(ho, hc, rngs[r])
            np.add.at(hist[r, arm], (status, time), w[copies])
            np.add.at(count[r, arm], (status, time), 1)

    log_hr = []
    for r in range(R):
        kept = hist[r, :, :2].sum(1)
        at_risk = kept[:, ::-1].cumsum(1)[:, ::-1]
        log_hr.append(_cox(hist[r, :, 1], at_risk, levels, count[r, :, 1]))
    full, boot = log_hr[0], np.array([x for x in log_hr[1:] if np.isfinite(x)])
    out = {"Hazard ratio": [float(np.exp(full))]}
    if len(boot) and np.isfinite(full):
        boot = _rescale_boot(boot, full, sample)
        if CI_method == "se":
            z = norm.ppf(1 - (1 - CI) / 2)
            se = np.std(boot)
            lo, hi = np.exp(full - z * se), np.exp(full + z * se)
        else:
            lo = np.exp(np.quantile(boot, (1 - CI) / 2))
            hi = np.exp(np.quantile(boot, 1 - (1 - CI) / 2))
        out |= {"LCI": [float(lo)], "UCI": [float(hi)]}
    return pl.DataFrame(out)
