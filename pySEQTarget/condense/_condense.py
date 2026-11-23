import copy
import time
from dataclasses import dataclass

import numpy as np
import polars as pl
import scipy.sparse as sp

from ._census import _Census
from ._correct import (Exact, _block_diag, _condensed_side, _correct,
                       _exact_side, _ExactRows, _fit_of)
from ._eof import Eof, _eof_batch
from ._estimands import _baseline_rows, _hazard, _replicates, _survival
from ._fit import _class_names, _newton
from ._frame import (_bind_weights, _expanded, _outcome_frame, _record_frame,
                     _recorded, _segments)
from ._models import _Block, _Model
from ._probes import _pilot_fits, _probes, _whitener
from ._select import select_atom
from ._weights import (_attach_sensitivities, _classes, _fit_gold_weights,
                       _package_params, _Tally, _Theta, _weight_models)


@dataclass
class _Atom:
    cols: np.ndarray
    W: np.ndarray
    # probes per model each probes by columns by classes
    train: list
    test: list


class _BlockSetup:
    def __init__(
        self, block, gold, n_probes, n_submodels, spread, ridge, rng, device, sensitive
    ):
        self.block, self.gold = block, gold
        self.n_probes, self.spread = n_probes, spread
        self.ridge, self.device, self.sensitive = ridge, device, sensitive
        # own probe stream per atom so batch order does not matter
        self.atom_seed = int(rng.integers(2**62))
        self.fits = []
        for j, m in enumerate(block.models):
            on = gold.member[:, j]
            cols = block.model_cols[j]
            names = [block.columns[c] for c in cols]
            intercept = names.index("Intercept") if "Intercept" in names else None
            self.fits.append(
                _pilot_fits(
                    gold.X[np.ix_(on, cols)],
                    m.index(gold.Y[on, j]),
                    gold.w[on],
                    intercept,
                    n_submodels,
                    rng,
                    m.m,
                    name=m.name,
                )
            )
        self.atoms = {}

    def atom(self, code):
        if code not in self.atoms:
            self.atoms[code] = self._atom(code)
        return self.atoms[code]

    def _atom(self, code):
        block, gold = self.block, self.gold
        rng = np.random.default_rng([self.atom_seed, int(code)])
        models = [j for j in range(len(block.models)) if code >> j & 1]
        cols = np.array(sorted({c for j in models for c in block.model_cols[j]}))

        def embed(j, B):
            out = np.zeros((len(B), len(cols), B.shape[2]))
            out[:, np.searchsorted(cols, block.model_cols[j])] = B
            return out

        def probes():
            split = np.array_split(np.arange(self.n_probes), len(models))
            return [
                embed(j, _probes(self.fits[j], len(idx), self.spread, rng))
                for j, idx in zip(models, split)
                if len(idx)
            ]

        train, test = probes(), probes()
        on = gold.member[:, models].any(1) & ~np.isnan(gold.X[:, cols]).any(1)
        Xg, wg = gold.X[np.ix_(on, cols)], gold.w[on]
        betas = [embed(j, self.fits[j][0][0][None])[0] for j in models]
        W = _whitener(Xg, wg, betas, len(np.unique(gold.ids[on])))
        return _Atom(cols, W, train, test)


def _model_name(model, subgroup=None):
    return model if subgroup is None else f"{model}[{subgroup}]"


def _condense_batch(setup, rows, k, timings, S=None):
    block = setup.block
    people, person = np.unique(rows.ids, return_inverse=True)
    codes = rows.member @ (1 << np.arange(len(block.models)))
    out = {"people": people, "person": [], "code": [], "X": [], "w": [], "S": []}
    diagnostics = []
    for code in np.unique(codes):
        sel = np.flatnonzero(codes == code)
        atom = setup.atom(int(code))
        X, w = rows.X[np.ix_(sel, atom.cols)], rows.w[sel]
        Sa = None if S is None else S[sel]
        seg, starts = _segments(rows.ids[sel])
        Xr, wr, mask, counts, Sk, errors = select_atom(
            atom, X, w, seg, starts, k, Sa, setup.ridge, timings, setup.device
        )
        keep = mask.ravel()
        Xfull = np.full((keep.sum(), len(block.columns)), np.nan)
        Xfull[:, atom.cols] = Xr.reshape(-1, len(atom.cols))[keep]
        unit_person = person[sel][starts]
        out["person"].append(np.repeat(unit_person, k)[keep])
        out["code"].append(np.full(keep.sum(), code))
        out["X"].append(Xfull)
        out["w"].append(wr.ravel()[keep])
        if Sa is not None:
            out["S"].append(Sk.reshape(-1, Sa.shape[1])[keep])

        diag = {
            "person": unit_person,
            "atom": np.full(len(starts), code),
            "rows": counts,
            "exact": counts <= k,
        }
        for name, e in zip(("grad", "hess", "cross"), errors):
            if e is not None:
                diag[f"rel_{name}"] = e
        diagnostics.append(pl.DataFrame(diag))

    out["t"], out["events"] = [], []
    for j, cols in enumerate(block.model_cols):
        model = block.models[j]
        y = model.index(np.nan_to_num(rows.Y[:, j]))
        on = np.flatnonzero(rows.member[:, j] & (y != 0))
        by_person = sp.csr_matrix(
            (np.ones(len(on)), (person[on], np.arange(len(on)))),
            shape=(len(people), len(on)),
        )
        x = rows.X[np.ix_(on, cols)]
        if model.classes is None:
            wy = rows.w[on] * y[on]
            out["t"].append(np.asarray(by_person @ (wy[:, None] * x)))
        else:
            # XWy per class for a multinomial model
            wy = rows.w[on]
            hot = wy[:, None] * (y[on, None] == np.arange(1, model.m + 1))
            out["t"].append(
                np.stack([by_person @ (h[:, None] * x) for h in hot.T], axis=2)
            )
        # event rows kept exactly for moving weights and non linear models
        out["events"].append((person[on], x, wy, None if S is None else S[on]))
    out["diagnostics"] = pl.concat(diagnostics)
    return out


class CondensedBlock:
    """
    Condensed rows of the models fit on one frame

    :ivar columns: Union of the model columns
    :ivar models: Model names
    :ivar ids: IDs in the order of the per-person statistics
    :ivar diagnostics: Per person row counts and held-out errors
    """

    def __init__(self, block, parts, id_col):
        self.block = block
        self.columns = block.columns
        self.models = [m.name for m in block.models]
        self.model_cols = block.model_cols
        offsets = np.cumsum([0] + [len(p["people"]) for p in parts])
        self.ids = np.concatenate([p["people"] for p in parts])
        self.person = np.concatenate(
            [q + o for p, o in zip(parts, offsets) for q in p["person"]]
        )
        self.code = np.concatenate([q for p in parts for q in p["code"]])
        self.X = np.vstack([q for p in parts for q in p["X"]])
        self.w = np.concatenate([q for p in parts for q in p["w"]])
        sensitive = bool(parts[0]["S"])
        self.S = np.vstack([q for p in parts for q in p["S"]]) if sensitive else None
        self.t = [
            np.vstack([p["t"][j] for p in parts]) for j in range(len(self.models))
        ]
        self.events = [
            tuple(
                (
                    np.concatenate(
                        [
                            p["events"][j][f] + (o if f == 0 else 0)
                            for p, o in zip(parts, offsets)
                        ]
                    )
                    if f < 3 or sensitive
                    else None
                )
                for f in range(4)
            )
            for j in range(len(self.models))
        ]
        self.diagnostics = pl.concat(
            [
                p["diagnostics"].with_columns(
                    pl.Series(
                        id_col, p["people"][p["diagnostics"]["person"].to_numpy()]
                    )
                )
                for p in parts
            ]
        ).drop("person")

    @property
    def table(self):
        """
        The condensed rows as a dataframe
        """
        table = pl.DataFrame(
            {"id": self.ids[self.person], "atom": self.code, "w": self.w}
        ).hstack(pl.DataFrame(self.X, schema=self.columns))
        if self.S is not None:
            table = table.hstack(
                pl.DataFrame(self.S, schema=[f"s{j}" for j in range(self.S.shape[1])])
            )
        return table

    def rows(self, model):
        """
        Rows any model of the columns can be trained on

        :param model: Model name
        :returns: Condensed rows with weights and exact event rows
        :rtype: dict
        """
        j = self.models.index(model)
        cols = self.model_cols[j]
        sel = (self.code >> j) & 1 == 1
        person, x, wy, _ = self.events[j]
        return {
            "columns": [self.columns[c] for c in cols],
            "X": self.X[np.ix_(sel, cols)],
            "w": self.w[sel],
            "person": self.person[sel],
            "X_events": x,
            "wy_events": wy,
            "person_events": person,
        }

    def fit(self, model, columns=None, delta=None, counts=None):
        """
        Fits a logistic model on the condensed rows

        :param model: Model name
        :param columns: Subset of the model columns
        :param delta: Shift of the weight model coefficients
        :param counts: Times each person is counted
        :rtype: Fit
        """
        j = self.models.index(model)
        names = [self.columns[c] for c in self.model_cols[j]]
        use = (
            np.arange(len(names))
            if columns is None
            else [names.index(c) for c in columns]
        )
        sel = (self.code >> j) & 1 == 1
        X = self.X[np.ix_(sel, self.model_cols[j][use])]
        w = self.w[sel]
        if delta is not None and self.S is not None:
            w = w * np.exp(self.S[sel] @ delta)
            person, x, wy, S = self.events[j]
            wy = wy * np.exp(S @ delta)
            if counts is not None:
                wy = wy * counts[person]
            t = wy @ x[:, use]
        else:
            t = self.t[j][:, use]
            t = t.sum(axis=0) if counts is None else np.tensordot(counts, t, 1)
        if counts is not None:
            w = w * counts[self.person[sel]]
        columns = _class_names([names[i] for i in use], self.block.models[j].classes)
        return _newton(X, w, t, columns)


class Condensed:
    """
    Condensed analysis standing in for the expanded data

    :ivar outcome: Condensed rows for the outcome models, None with ``end_of_fup``
    :ivar record: Condensed rows for the weight models, None when unweighted
    :ivar theta0: Weight model coefficients the weights were computed at
    :ivar analysis: SEQuential the batches were expanded with
    :ivar cohort: IDs of every person
    :ivar gold: Exact rows of the gold subsample
    :ivar baseline: First row of every trial
    :ivar weight_stats: Weight statistics over every expanded row
    """

    def __init__(
        self, outcome, record, theta, theta0, analysis, census, cohort, timings
    ):
        self.outcome = outcome
        self.record = record
        self._theta = theta
        self.theta0 = theta0
        self.analysis = analysis
        self.census = census
        self.cohort = np.sort(np.asarray(cohort))
        self.timings = timings
        self.gold = None
        self.baseline = None
        self.weight_stats = None

    def exact(self, store, ids=None, people_per_batch=1000):
        """
        Exactly expanded rows of some people

        :param store: Record on disk
        :param ids: People to expand, default everyone
        :param people_per_batch: People expanded at a time
        :rtype: Exact
        """
        a = self.analysis
        if ids is None:
            ids = self.cohort
            batches = store.batches(people_per_batch)
        else:
            ids = np.sort(np.asarray(ids))
            batches = (
                store.read(ids[i : i + people_per_batch])
                for i in range(0, len(ids), people_per_batch)
            )
        record, outcome = [], []
        for batch in batches:
            b = _expanded(a, batch)
            S = None
            if self.record is not None:
                frame = _record_frame(a, b)
                rows = self.record.block.matrices(frame)
                record.append(_ExactRows(rows.ids, rows.X, rows.Y, rows.member, rows.w))
                _bind_weights(a, b)
                s_cols = _attach_sensitivities(
                    a, self.record.block, self.theta0, frame, b
                )
            frame = _outcome_frame(a, b)
            rows = self.outcome.block.matrices(frame)
            if self.record is not None:
                S = frame[s_cols].to_numpy(float)[rows.keep]
            outcome.append(_ExactRows(rows.ids, rows.X, rows.Y, rows.member, rows.w, S))
        return Exact(
            ids,
            _ExactRows.concat(outcome),
            _ExactRows.concat(record) if record else None,
        )

    def fit_corrected(
        self, model="outcome", columns=None, sample=None, lam="optimal", subgroup=None
    ):
        """
        Outcome model fit corrected with an exact sample of people

        :param model: Model to fit ['outcome' or 'compevent']
        :param columns: Subset of the model columns
        :param sample: Exact rows of the sample, default the gold subsample
        :param lam: Weight of the condensed rows, or 'optimal'
        :param subgroup: Subgroup whose model to fit
        :rtype: CorrectedFit
        """
        model = _model_name(model, subgroup)
        sample = self.gold if sample is None else sample
        N = len(self.cohort)
        n = len(sample.people)

        def people_of(ids):
            return np.searchsorted(self.cohort, ids)

        S_idx = people_of(sample.people)
        weights, solutions, delta = {}, [], None
        if self.record is not None:
            for j, name in enumerate(self.record.models):
                cols = self.record.model_cols[j]
                kind = self.record.block.models[j]
                names = _class_names(
                    [self.record.columns[c] for c in cols], kind.classes
                )
                sol = _correct(
                    _condensed_side(self.record, j, np.arange(len(cols)), people_of, N),
                    _exact_side(sample.record, cols, kind, j, people_of, N),
                    S_idx,
                    N,
                    lam,
                    self.record.fit(name).params.to_numpy(),
                )
                solutions.append(sol)
                weights[name] = _fit_of(names, sol, S_idx, N)
            delta = np.concatenate([s.params for s in solutions]) - self.theta0

        j = self.outcome.models.index(model)
        names = [self.outcome.columns[c] for c in self.outcome.model_cols[j]]
        use = (
            np.arange(len(names))
            if columns is None
            else np.array([names.index(c) for c in columns])
        )
        names = [names[i] for i in use]
        cols = self.outcome.model_cols[j][use]
        cond = _condensed_side(self.outcome, j, use, people_of, N, delta)
        kind = self.outcome.block.models[j]
        exact = _exact_side(sample.outcome, cols, kind, j, people_of, N, delta)
        start = self.outcome.fit(model, [n_ for n_ in names], delta)
        sol = _correct(cond, exact, S_idx, N, lam, start.params.to_numpy())
        if not solutions:
            return _fit_of(names, sol, S_idx, N)

        # propagate weight model estimation into the outcome variance
        in_sample = np.zeros(N, bool)
        in_sample[S_idx] = True
        A = sol.lam * cond.cross(sol.params) + N / n * (
            exact.cross(sol.params, in_sample)
            - sol.lam * cond.cross(sol.params, in_sample)
        )
        M = A @ _block_diag([np.linalg.pinv(s.H, rcond=1e-12) for s in solutions])
        Gw_e = np.hstack([s.exact for s in solutions])
        Gw_c = np.hstack([s.condensed for s in solutions])
        lam_w = np.concatenate([np.full(s.params.shape, s.lam) for s in solutions])
        e = (sol.exact[S_idx] - sol.lam * sol.condensed[S_idx]) - (
            Gw_e[S_idx] - lam_w * Gw_c[S_idx]
        ) @ M.T
        fit = _fit_of(
            names, sol, S_idx, N, e, sol.exact - Gw_e @ M.T, sol.condensed - Gw_c @ M.T
        )
        fit.weights = weights
        return fit

    def end_of_followup(
        self, nboot=0, sample=1.0, seed=None, CI=None, CI_method=None, resamples=None
    ):
        """
        Assembles the end-of-follow-up estimates (``end_of_fup=True``)

        :param nboot: Number of bootstrap resamples
        :param sample: Share of the cohort drawn per resample
        :param seed: RNG seed
        :param CI: Confidence interval level
        :param CI_method: Confidence interval method ['se' or 'percentile']
        :param resamples: Resamples to use as a list of {ID: times drawn}
        :returns: ``eof_data`` and ``eof_comparison``
        :rtype: dict
        """
        a = self.analysis
        if getattr(self, "eof", None) is None:
            raise ValueError("Not an end-of-follow-up analysis (end_of_fup=False)")
        if resamples is None:
            rng = np.random.default_rng(a.seed if seed is None else seed)
            size = int(sample * len(self.cohort))
            resamples = []
            for _ in range(nboot):
                draw = rng.choice(self.cohort, size=size)
                ids, n = np.unique(draw, return_counts=True)
                resamples.append(dict(zip(ids.tolist(), n.tolist())))

        def estimate(counts=None):
            delta = None
            if self.record is not None:
                fits = self.fit_weights(counts)
                delta = self._theta.stack({n: f.params for n, f in fits.items()})
                delta = delta - self.theta0
            return self.eof.estimate(delta, counts)

        return self.eof.tables(
            [estimate()] + [estimate(counts) for counts in resamples],
            a.bootstrap_CI if CI is None else CI,
            a.bootstrap_CI_method if CI_method is None else CI_method,
        )

    def bootstrap(self, nboot, sample=1.0, seed=None):
        """
        Draws bootstrap resamples and fits every outcome model on each

        :param nboot: Number of bootstrap resamples
        :param sample: Share of the cohort drawn per resample
        :param seed: RNG seed
        """
        seed = self.analysis.seed if seed is None else seed
        self.replicates = _replicates(self, nboot, sample, seed)
        return self.replicates

    def _main(self, fits):
        fits = fits or {}
        return {
            m: (fits[m] if m in fits else self.fit(m)).params.to_numpy()
            for m in self.outcome.models
        }

    def survival(self, fits=None, CI=None, CI_method=None, risk_times=None):
        """
        Estimates risk, survival, and incidence curves from the condensed fit

        :param fits: Fits to use in place of the condensed ones, by model name
        :param CI: Confidence interval level
        :param CI_method: Confidence interval method ['se' or 'percentile']
        :param risk_times: Followup times to report risk difference and ratio at
        :returns: ``km_data``, ``risk_difference`` and ``risk_ratio``
        :rtype: dict
        """
        a = self.analysis
        return _survival(
            self,
            self._main(fits),
            getattr(self, "replicates", []),
            a.bootstrap_CI if CI is None else CI,
            a.bootstrap_CI_method if CI_method is None else CI_method,
            a.risk_times if risk_times is None else risk_times,
        )

    def hazard(self, fits=None, CI=None, CI_method=None, seed=None):
        """
        Estimates hazard ratios from the condensed fit

        :param fits: Fits to use in place of the condensed ones, by model name
        :param CI: Confidence interval level
        :param CI_method: Confidence interval method ['se' or 'percentile']
        :param seed: RNG seed of the simulation
        :rtype: pl.DataFrame
        """
        a = self.analysis
        return _hazard(
            self,
            self._main(fits),
            getattr(self, "replicates", []),
            a.bootstrap_CI if CI is None else CI,
            a.bootstrap_CI_method if CI_method is None else CI_method,
            seed,
        )

    @staticmethod
    def _counts(block, counts):
        if counts is None:
            return None
        return np.array([counts.get(i, 0) for i in block.ids.tolist()], float)

    def fit_weights(self, counts=None):
        """
        Fits the weight models on the condensed rows

        :param counts: Times each ID is counted, e.g. a bootstrap resample
        :returns: Fit per weight model
        :rtype: dict
        """
        if self.record is None:
            return {}
        c = self._counts(self.record, counts)
        return {name: self.record.fit(name, counts=c) for name in self.record.models}

    def theta_of(self, fitted):
        """
        Weight model coefficients of a fitted SEQuential as one vector
        """
        return self._theta.of(fitted)

    def fit(
        self, model="outcome", columns=None, theta="fitted", counts=None, subgroup=None
    ):
        """
        Fits an outcome model on the condensed rows

        :param model: Model to fit ['outcome' or 'compevent']
        :param columns: Subset of the model columns
        :param theta: Weight model coefficients ['fitted', 'pilot', dict or vector]
        :param counts: Times each ID is counted, e.g. a bootstrap resample
        :param subgroup: Subgroup whose model to fit
        :rtype: Fit
        """
        if self.outcome is None:
            raise ValueError(
                "An end-of-follow-up analysis has no outcome model; use "
                "end_of_followup()"
            )
        delta = None
        if self.record is not None:
            if isinstance(theta, str) and theta == "pilot":
                delta = np.zeros_like(self.theta0)
            elif isinstance(theta, str) and theta == "fitted":
                fits = self.fit_weights(counts)
                delta = self._theta.stack({n: f.params for n, f in fits.items()})
                delta = delta - self.theta0
            elif isinstance(theta, dict):
                delta = self._theta.stack(theta) - self.theta0
            else:
                delta = np.asarray(theta, float) - self.theta0
        return self.outcome.fit(
            _model_name(model, subgroup),
            columns,
            delta,
            self._counts(self.outcome, counts),
        )


def condense(
    self,
    store,
    k=32,
    k_record=None,
    gold_fraction=0.1,
    gold_max=5000,
    center="record",
    n_probes=64,
    n_submodels=8,
    spread=3.0,
    people_per_batch=1000,
    sensitivity_ridge=1e-3,
    seed=None,
    device="cpu",
):
    """
    Condenses the expansion to a few reweighted rows per person

    :param self: Analysis whose options are used
    :type self: SEQuential
    :param store: Record on disk
    :type store: RecordStore
    :param k: Rows kept per person for the outcome models
    :param k_record: Rows kept per person for the weight models, default ``k``
    :param gold_fraction: Share of people expanded exactly for the pilot fits
    :param gold_max: Most people expanded exactly for the pilot fits, None for no limit
    :param center: Weight models the weights are computed at ['record' or 'gold']
    :param n_probes: Coefficient vectors the rows are matched at
    :param n_submodels: Random sub-models among the pilot fits
    :param spread: Probe spread in pilot standard errors
    :param people_per_batch: People expanded at a time
    :param sensitivity_ridge: Shrinkage of the weight sensitivities
    :param seed: RNG seed
    :param device: Torch device, default ``"cpu"``
    :returns: The condensed analysis
    :rtype: Condensed
    """
    if center not in ("record", "gold"):
        raise ValueError("center must be 'record' or 'gold'")
    rng = np.random.default_rng(self.seed if seed is None else seed)
    k_record = k if k_record is None else k_record
    timings = dict.fromkeys(
        ("census", "record", "expand", "weights", "targets", "condense"), 0.0
    )

    t = time.perf_counter()
    census = _Census(self, store, people_per_batch)
    timings["census"] = time.perf_counter() - t

    a = copy.copy(self)
    ids = np.asarray(store.ids())
    n_gold = max(1, round(gold_fraction * len(ids)))
    # gold is held in memory at once
    n_gold = n_gold if gold_max is None else min(n_gold, gold_max)
    gold_ids = np.sort(rng.choice(ids, size=n_gold, replace=False))
    gold = _expanded(a, store.read(gold_ids))

    settings = (n_probes, n_submodels, spread, sensitivity_ridge, rng, device)
    record = theta = theta0 = None
    if self.weighted:
        gold_record = _record_frame(a, gold)
        _fit_gold_weights(a, census.formulas, gold_record.copy())
        models = [
            m
            for m in _weight_models(a, census.formulas)
            if _package_params(a, m.name) is not None
        ]
        if not models:
            raise ValueError("No weight model could be fit on the gold subsample")
        for m in models:
            m.classes = _classes(a, m.name)
        record_block = _Block(models, gold_record, census.levels["record"], self.id_col)
        theta = _Theta(record_block)
        setup = _BlockSetup(
            record_block, record_block.matrices(gold_record), *settings, False
        )
        parts = []
        for batch in store.batches(people_per_batch):
            t = time.perf_counter()
            # weight model rows are the record or the expansion
            b = _recorded(a, batch) if self.weight_preexpansion else _expanded(a, batch)
            rows = record_block.matrices(_record_frame(a, b))
            timings["record"] += time.perf_counter() - t
            parts.append(_condense_batch(setup, rows, k_record, timings))
        record = CondensedBlock(record_block, parts, self.id_col)
        if center == "record":
            fits = {name: record.fit(name) for name in record.models}
            theta.apply(a, theta.stack({n: f.params for n, f in fits.items()}))
        theta0 = theta.of(a)
        if self.weight_p99:
            # extra pass for the 1st and 99th weight percentiles
            t = time.perf_counter()
            bounds = _Tally()
            for batch in store.batches(people_per_batch):
                b = _expanded(a, batch)
                _bind_weights(a, b)
                bounds.add(b.DT["weight"].to_numpy())
            a.weight_stats = bounds.frame()
            timings["weights"] += time.perf_counter() - t
        _bind_weights(a, gold)
    tally = _Tally()

    if self.end_of_fup:
        # no outcome model so measurements are kept exactly
        parts, s_cols = [], []
        for batch in store.batches(people_per_batch):
            t = time.perf_counter()
            b = _expanded(a, batch)
            timings["expand"] += time.perf_counter() - t
            if self.weighted:
                t = time.perf_counter()
                frame = _record_frame(a, b)
                _bind_weights(a, b)
                tally.add(b.DT["weight"].to_numpy())
                s_cols = _attach_sensitivities(a, record_block, theta0, frame, b)
                timings["weights"] += time.perf_counter() - t
            parts.append(_eof_batch(a, b, s_cols, census.last_period))
        c = Condensed(None, record, theta, theta0, a, census, ids, timings)
        c.eof = Eof(a, parts, s_cols)
        c.weight_stats = tally.frame() if self.weighted else None
        return c

    rhs = census.covariates
    roles = [("outcome", self.outcome_col)]
    if self.compevent_colname is not None:
        roles.append(("compevent", self.compevent_colname))
    if self.subgroup_colname is None:
        models = [_Model(role, col, rhs) for role, col in roles]
    else:
        # one model per subgroup sharing weights and design
        def within(value):
            return lambda f: f[self.subgroup_colname].to_numpy() == value

        models = [
            _Model(_model_name(role, value), col, rhs, within(value))
            for value in census.subgroups
            for role, col in roles
        ]
    gold_frame = _outcome_frame(a, gold)
    outcome_block = _Block(
        models,
        gold_frame,
        census.levels["outcome"],
        self.id_col,
        "weight" if self.weighted else None,
    )
    setup = _BlockSetup(
        outcome_block, outcome_block.matrices(gold_frame), *settings, self.weighted
    )
    parts, baseline = [], []
    for batch in store.batches(people_per_batch):
        t = time.perf_counter()
        b = _expanded(a, batch)
        baseline.append(_baseline_rows(a, b))
        timings["expand"] += time.perf_counter() - t

        S = None
        if self.weighted:
            t = time.perf_counter()
            frame = _record_frame(a, b)
            _bind_weights(a, b)
            tally.add(b.DT["weight"].to_numpy())
            s_cols = _attach_sensitivities(a, record_block, theta0, frame, b)
            timings["weights"] += time.perf_counter() - t

        t = time.perf_counter()
        frame = _outcome_frame(a, b)
        rows = outcome_block.matrices(frame)
        if self.weighted:
            S = frame[s_cols].to_numpy(float)[rows.keep]
        timings["expand"] += time.perf_counter() - t
        parts.append(_condense_batch(setup, rows, k, timings, S))

    c = Condensed(
        CondensedBlock(outcome_block, parts, self.id_col),
        record,
        theta,
        theta0,
        a,
        census,
        ids,
        timings,
    )
    c.baseline = pl.concat(baseline)
    c.weight_stats = tally.frame() if self.weighted else None
    c.gold = c.exact(store, gold_ids, people_per_batch)
    return c
