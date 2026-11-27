import warnings

import numpy as np
from scipy.special import softmax

from ._fit import _newton


def _pilot_fits(
    X, y, w, intercept, n_submodels, rng, m=1, max_eta_se=3.0, name="outcome"
):
    # y is the class index and each fit is coefficients D by m with covariance
    D = X.shape[1]
    fits = []
    masks = [np.ones(D, bool)] + [rng.random(D) < 0.5 for _ in range(n_submodels)]
    hot = w[:, None] * (y[:, None] == np.arange(1, m + 1))
    for keep in masks:
        if intercept is not None:
            keep[intercept] = True
        try:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                # own newton as statsmodels holds several copies of the rows
                Xk = X[:, keep]
                f = _newton(Xk, w, Xk.T @ hot, list(range(m * keep.sum())))
            where = np.flatnonzero(np.tile(keep, m))
            beta = np.zeros(m * D)
            beta[where] = f.params
            cov = np.zeros((m * D, m * D))
            cov[np.ix_(where, where)] = f.cov_params()
            eta_se = max(
                np.sqrt(np.mean(((X @ block) * X).sum(1)))
                for block in (
                    cov[c * D : (c + 1) * D, c * D : (c + 1) * D] for c in range(m)
                )
            )
            usable = np.all(np.isfinite(beta)) and eta_se <= max_eta_se
        except Exception:
            usable = False
        if usable:
            fits.append((beta.reshape(m, D).T, cov))
        elif not fits:
            raise ValueError(
                f"The {name} model is separated or does not fit on the gold "
                "subsample; increase gold_fraction"
            )
    return fits


def _probes(fits, n, spread, rng):
    # n coefficient arrays D by m spread around the pilot fits
    per_fit = np.array_split(np.arange(n), len(fits))
    out = []
    for (beta, cov), idx in zip(fits, per_fit):
        if len(idx) == 0:
            continue
        D, m = beta.shape
        vals, vecs = np.linalg.eigh(cov)
        root = vecs * np.sqrt(np.clip(vals, 0, None))
        z = rng.standard_normal((len(idx), m * D)) @ root.T
        c = rng.uniform(0, spread, size=(len(idx), 1))
        c[0] = 0
        out.append((beta.T.ravel() + c * z).reshape(len(idx), m, D).transpose(0, 2, 1))
    return np.concatenate(out)


def _whitener(X, w, betas, n_people):
    H = np.zeros((X.shape[1], X.shape[1]))
    for beta in betas:
        eta = np.hstack([np.zeros((len(X), 1)), X @ beta])
        p = softmax(eta, axis=1)[:, 1:]
        H += (X * (w[:, None] * p * (1 - p)).sum(1)[:, None]).T @ X
    vals, vecs = np.linalg.eigh(H / (len(betas) * n_people))
    return vecs / np.sqrt(np.clip(vals, vals.max() * 1e-12, None))
