from functools import partial

import jax
import jax.numpy as jnp
from jax.flatten_util import ravel_pytree


class MultinomialLogisticRegression:
    """
    Multinomial logistic regression in JAX.
    """

    def __init__(self, max_iter=25, ridge=1e-8, eps=1e-7, n_classes=None):
        self.max_iter = max_iter
        self.ridge = ridge  # tiny Hessian jitter for numerical stability
        self.eps = eps
        self.n_classes = n_classes
        self.params = None
        self.loss_history_ = None
        self.n_classes_ = None
        self.mean_ = None
        self.std_ = None

    def _prep(self, X):
        X = jnp.asarray(X)
        if self.mean_ is not None:
            X = (X - self.mean_) / self.std_
        return X

    @staticmethod
    def _logits(params, X):
        W, b = params
        z = jnp.dot(X, W) + b
        return jnp.concatenate([jnp.zeros((z.shape[0], 1)), z], axis=1)

    @staticmethod
    def _predict(params, X):
        return jax.nn.softmax(MultinomialLogisticRegression._logits(params, X), axis=-1)

    @staticmethod
    @jax.jit
    def _loss(params, X, Y, w):
        logp = jax.nn.log_softmax(
            MultinomialLogisticRegression._logits(params, X), axis=-1
        )
        ce = -jnp.sum(Y * logp, axis=-1)
        return jnp.sum(w * ce) / jnp.sum(w)

    @staticmethod
    @partial(jax.jit, static_argnums=(4,))
    def _run(params, X, Y, w, max_iter, ridge):
        flat0, unravel = ravel_pytree(params)
        eye = jnp.eye(flat0.shape[0])

        def loss_flat(f):
            return MultinomialLogisticRegression._loss(unravel(f), X, Y, w)

        grad_fn = jax.grad(loss_flat)
        hess_fn = jax.hessian(loss_flat)

        def step(f, _):
            g = grad_fn(f)
            H = hess_fn(f)
            f = f - jnp.linalg.solve(H + ridge * eye, g)
            return f, loss_flat(f)

        flat, loss_history = jax.lax.scan(step, flat0, xs=None, length=max_iter)
        return unravel(flat), loss_history

    def fit(self, X, y, sample_weight=None, init_params=None):
        X = self._prep(X)
        y = jnp.asarray(y)
        K = self.n_classes if self.n_classes is not None else int(y.max()) + 1
        self.n_classes_ = K
        Y = jax.nn.one_hot(y, K)
        if sample_weight is None:
            w = jnp.ones(X.shape[0])
        else:
            w = jnp.asarray(sample_weight, dtype=float)
        if init_params is None:
            params = (jnp.zeros((X.shape[1], K - 1)), jnp.zeros(K - 1))
        else:
            W, b = init_params
            W = jnp.asarray(W, dtype=float)
            b = jnp.asarray(b, dtype=float)
            if W.shape != (X.shape[1], K - 1):
                raise ValueError(
                    f"init_params W has shape {W.shape}, expected ({X.shape[1]}, {K - 1})."
                )
            params = (W, b)
        self.params, self.loss_history_ = self._run(
            params, X, Y, w, self.max_iter, self.ridge
        )
        return self

    def predict(self, X):
        if self.params is None:
            raise RuntimeError("Model is not fitted yet; call `fit` first.")
        return self._predict(self.params, self._prep(X))
