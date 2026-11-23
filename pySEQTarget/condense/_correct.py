import warnings
from dataclasses import dataclass

import numpy as np
import pandas as pd
import scipy.sparse as sp
from scipy.special import softmax


def _block_diag(blocks):
    out = np.zeros((sum(len(b) for b in blocks),) * 2)
    i = 0
    for b in blocks:
        out[i : i + len(b), i : i + len(b)] = b
        i += len(b)
    return out


def _by_person(person, n):
    return sp.csr_matrix(
        (np.ones(len(person)), (person, np.arange(len(person)))),
        shape=(n, len(person)),
    )


class _Side:
    def __init__(self, X, w, person, t, n, m=1, events=None, S=None):
        # t is people by coefficients stacked class by class
        self.X, self.w, self.person, self.t, self.m = X, w, person, t, m
        # event rows and sensitivities for weights that move
        self.events, self.S = events, S
        self.by = _by_person(person, n)

    def _p(self, X, beta):
        eta = X @ beta.reshape(self.m, -1).T
        return softmax(np.hstack([np.zeros((len(X), 1)), eta]), axis=1)[:, 1:]

    def scores(self, beta):
        p = self._p(self.X, beta)
        return (
            np.hstack(
                [
                    self.by @ ((self.w * p[:, c])[:, None] * self.X)
                    for c in range(self.m)
                ]
            )
            - self.t
        )

    def hessian(self, beta, people=None):
        rows = slice(None) if people is None else people[self.person]
        X, w = self.X[rows], self.w[rows]
        p = self._p(X, beta)
        m, D = self.m, X.shape[1]
        H = np.zeros((m, D, m, D))
        for c in range(m):
            for d in range(c, m):
                v = w * (p[:, c] * (c == d) - p[:, c] * p[:, d])
                H[c, :, d] = H[d, :, c] = (X * v[:, None]).T @ X
        return H.reshape(m * D, m * D)

    def cross(self, beta, people=None):
        # outcome models only and those are binary
        xt, wy, person_t, St = self.events
        rows = slice(None) if people is None else people[self.person]
        rows_t = slice(None) if people is None else people[person_t]
        X = self.X[rows]
        mu = self._p(X, beta)[:, 0]
        return (X * (self.w[rows] * mu)[:, None]).T @ self.S[rows] - (
            xt[rows_t] * wy[rows_t][:, None]
        ).T @ St[rows_t]


def _condensed_side(block, j, use, people_of, n, delta=None):
    sel = (block.code >> j) & 1 == 1
    X = block.X[np.ix_(sel, block.model_cols[j][use])]
    w = block.w[sel]
    person = people_of(block.ids[block.person[sel]])
    if block.S is None:
        t = block.t[j][:, use]
        t = t.reshape(len(t), len(use), -1).transpose(0, 2, 1).reshape(len(t), -1)
        full = np.zeros((n, t.shape[1]))
        full[people_of(block.ids)] = t
        return _Side(X, w, person, full, n, block.block.models[j].m)
    S = block.S[sel]
    p, x, wy, St = block.events[j]
    if delta is not None:
        w = w * np.exp(S @ delta)
        wy = wy * np.exp(St @ delta)
    person_t = people_of(block.ids[p])
    t = _by_person(person_t, n) @ (wy[:, None] * x[:, use])
    return _Side(X, w, person, t, n, 1, (x[:, use], wy, person_t, St), S)


def _exact_side(rows, cols, model, j, people_of, n, delta=None):
    on = rows.member[:, j]
    X = rows.X[np.ix_(on, cols)]
    w = rows.w[on]
    S = None if rows.S is None else rows.S[on]
    if S is not None and delta is not None:
        w = w * np.exp(S @ delta)
    y = model.index(np.nan_to_num(rows.Y[on, j]))
    person = people_of(rows.ids[on])
    hot = w[:, None] * (y[:, None] == np.arange(1, model.m + 1))
    by = _by_person(person, n)
    t = np.hstack([by @ (hot[:, [c]] * X) for c in range(model.m)])
    on = y != 0
    events = (X[on], hot[on, 0], person[on], None if S is None else S[on])
    return _Side(X, w, person, t, n, model.m, events, S)


@dataclass
class _Solution:
    params: np.ndarray
    H: np.ndarray
    lam: float
    exact: np.ndarray
    condensed: np.ndarray
    converged: bool


def _solve(cond, exact, sample, n_people, lam, start, tol=1e-16, max_iter=50):
    in_sample = np.zeros(n_people, bool)
    in_sample[sample] = True
    r = n_people / len(sample)
    beta = np.asarray(start, float)
    converged = False
    for _ in range(max_iter):
        Gc, Ge = cond.scores(beta), exact.scores(beta)
        psi = lam * Gc.sum(0) + r * (Ge[sample] - lam * Gc[sample]).sum(0)
        H = lam * cond.hessian(beta) + r * (
            exact.hessian(beta, in_sample) - lam * cond.hessian(beta, in_sample)
        )
        step = np.linalg.lstsq(H, psi, rcond=1e-12)[0]
        if psi @ step <= tol:
            converged = True
            break
        beta = beta - step
    if not converged:
        warnings.warn(
            f"Corrected fit did not converge in {max_iter} Newton steps",
            RuntimeWarning,
            stacklevel=3,
        )
    Gc, Ge = cond.scores(beta), exact.scores(beta)
    return _Solution(beta, H, lam, Ge, Gc, converged)


def _optimal_lam(H, Ge, Gc):
    Hi = np.linalg.pinv(H, rcond=1e-12)
    Ge, Gc = Ge - Ge.mean(0), Gc - Gc.mean(0)
    C = Ge.T @ Gc
    den = np.trace(Hi @ (Gc.T @ Gc) @ Hi)
    if den <= 0:
        return 1.0
    return float(np.clip(np.trace(Hi @ (C + C.T) / 2 @ Hi) / den, 0.0, 1.0))


def _correct(cond, exact, sample, n_people, lam, start):
    if lam == "optimal":
        first = _solve(cond, exact, sample, n_people, 1.0, start)
        lam = _optimal_lam(first.H, first.exact[sample], first.condensed[sample])
        start = first.params
    return _solve(cond, exact, sample, n_people, float(lam), start)


def _design(e, n_people):
    n = len(e)
    if n < 2 or n == n_people:
        return np.zeros((e.shape[1], e.shape[1]))
    return n_people**2 * (1 - n / n_people) / n * np.cov(e, rowvar=False)


def _total(lam, Ge, Gc, sample, n_people):
    r = n_people / len(sample)
    return lam * Gc.T @ Gc + r * (
        Ge[sample].T @ Ge[sample] - lam * Gc[sample].T @ Gc[sample]
    )


class CorrectedFit:
    """
    Fit corrected with an exact sample of people

    :ivar params: Corrected coefficients
    :ivar bse: Standard errors
    :ivar bse_design: Standard errors from sampling the exact rows alone
    :ivar lam: Weight of the condensed rows
    :ivar weights: Corrected weight models
    """

    def __init__(self, columns, params, cov, cov_design, lam, converged, weights=None):
        self.params = pd.Series(params, index=columns)
        self._cov = pd.DataFrame(cov, index=columns, columns=columns)
        self._cov_design = pd.DataFrame(cov_design, index=columns, columns=columns)
        self.bse = pd.Series(np.sqrt(np.diag(cov)), index=columns)
        self.bse_design = pd.Series(np.sqrt(np.diag(cov_design)), index=columns)
        self.lam = lam
        self.converged = converged
        self.weights = weights or {}

    def cov_params(self):
        return self._cov

    def cov_design(self):
        return self._cov_design


def _fit_of(columns, sol, sample, n_people, e=None, U_exact=None, U_cond=None):
    Ue = sol.exact if U_exact is None else U_exact
    Uc = sol.condensed if U_cond is None else U_cond
    if e is None:
        e = sol.exact[sample] - sol.lam * sol.condensed[sample]
    Hi = np.linalg.pinv(sol.H, rcond=1e-12)
    D = _design(e, n_people)
    J = _total(sol.lam, Ue, Uc, sample, n_people)
    return CorrectedFit(
        columns, sol.params, Hi @ (J + D) @ Hi, Hi @ D @ Hi, sol.lam, sol.converged
    )


@dataclass
class _ExactRows:
    ids: np.ndarray
    X: np.ndarray
    Y: np.ndarray
    member: np.ndarray
    w: np.ndarray
    S: np.ndarray = None

    @classmethod
    def concat(cls, parts):
        fields = ("ids", "X", "Y", "member", "w")
        S = [p.S for p in parts]
        return cls(
            *(np.concatenate([getattr(p, f) for p in parts]) for f in fields),
            None if S[0] is None else np.concatenate(S),
        )

    def subset(self, people):
        on = np.isin(self.ids, people)
        return _ExactRows(
            self.ids[on],
            self.X[on],
            self.Y[on],
            self.member[on],
            self.w[on],
            None if self.S is None else self.S[on],
        )


class Exact:
    """
    Exactly expanded rows of a sample of people

    :ivar people: IDs of the sampled people
    """

    def __init__(self, people, outcome, record=None):
        self.people = np.sort(np.asarray(people))
        self.outcome = outcome
        self.record = record

    def subset(self, people):
        """
        Rows of some of the sampled people
        """
        people = np.sort(np.asarray(people))
        return Exact(
            people,
            self.outcome.subset(people),
            None if self.record is None else self.record.subset(people),
        )
