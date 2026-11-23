import warnings

import numpy as np
import pandas as pd
from scipy.special import logsumexp, softmax


class Fit:
    """
    Weighted logistic fit on condensed rows

    :ivar params: Coefficients
    :ivar bse: Model-based standard errors
    """

    def __init__(self, columns, params, cov, iterations, converged):
        self.params = pd.Series(params, index=columns)
        self.bse = pd.Series(np.sqrt(np.diag(cov)), index=columns)
        self._cov = pd.DataFrame(cov, index=columns, columns=columns)
        self.iterations = iterations
        self.converged = converged

    def cov_params(self):
        return self._cov

    def summary(self):
        """
        Coefficients and standard errors as a dataframe
        """
        return pd.DataFrame({"coef": self.params, "std err": self.bse})


def _class_names(columns, classes):
    # multinomial coefficients are stacked class by class
    if classes is None:
        return list(columns)
    return [f"{col}[{cls}]" for cls in classes[1:] for col in columns]


def _newton(X, w, t, columns, tol=1e-16, max_iter=100):
    # t is D for a binary model or D by classes for a multinomial one
    D = X.shape[1]
    T = np.asarray(t, float).reshape(D, -1)
    m = T.shape[1]
    beta = np.zeros((D, m))

    def padded(B):
        return np.hstack([np.zeros((len(X), 1)), X @ B])

    def loss(B):
        return w @ logsumexp(padded(B), axis=1) - (T * B).sum()

    def derivatives(B):
        p = softmax(padded(B), axis=1)[:, 1:]
        g = (X.T @ (w[:, None] * p) - T).T.ravel()
        H = np.zeros((m, D, m, D))
        for c in range(m):
            for d in range(c, m):
                v = w * (p[:, c] * (c == d) - p[:, c] * p[:, d])
                H[c, :, d] = H[d, :, c] = (X * v[:, None]).T @ X
        return g, H.reshape(m * D, m * D)

    f = loss(beta)
    converged = False
    for iteration in range(1, max_iter + 1):
        g, H = derivatives(beta)
        # min norm step for rank deficient designs
        step = np.linalg.lstsq(H, g, rcond=1e-12)[0]
        decrement = g @ step
        if decrement <= tol:
            converged = True
            break
        s = 1.0
        move = step.reshape(m, D).T
        # backtrack only while the decrement is large
        while decrement > 1e-6 and s > 1e-10:
            if loss(beta - s * move) <= f - 1e-4 * s * decrement:
                break
            s /= 2
        beta = beta - s * move
        f = loss(beta)
    if not converged:
        warnings.warn(
            f"Condensed fit did not converge in {max_iter} Newton steps",
            RuntimeWarning,
            stacklevel=3,
        )
    _, H = derivatives(beta)
    cov = np.linalg.pinv(H, rcond=1e-12)
    return Fit(columns, beta.T.ravel(), cov, iteration, converged)
