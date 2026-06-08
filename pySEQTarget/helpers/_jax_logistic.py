from functools import partial

import jax
import jax.numpy as jnp


class MultinomialLogisticRegression:
    """
    Multinomial logistic regression in JAX.
    """

    def __init__(self, learning_rate=0.1, num_epochs=2000, eps=1e-7, n_classes=None):
        self.learning_rate = learning_rate
        self.num_epochs = num_epochs
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
    def _predict(params, X):
        W, b = params
        return jax.nn.softmax(jnp.dot(X, W) + b, axis=-1)

    @staticmethod
    @jax.jit
    def _loss(params, X, Y, w):
        """Sample-weighted mean softmax cross-entropy; ``w`` is ``(n_samples,)``."""
        W, b = params
        logp = jax.nn.log_softmax(jnp.dot(X, W) + b, axis=-1)
        ce = -jnp.sum(Y * logp, axis=-1)
        return jnp.sum(w * ce) / jnp.sum(w)

    @staticmethod
    @partial(jax.jit, static_argnums=(4, 5))
    def _run(params, X, Y, w, num_epochs, learning_rate):
        b1, b2, eps = 0.9, 0.999, 1e-8
        W0, b0 = params
        zeros = (jnp.zeros_like(W0), jnp.zeros_like(b0))

        def step(carry, t):
            # ADAM
            params, (mW, mb), (vW, vb) = carry
            dW, db = jax.grad(MultinomialLogisticRegression._loss)(params, X, Y, w)
            mW, vW = b1 * mW + (1 - b1) * dW, b2 * vW + (1 - b2) * dW ** 2
            mb, vb = b1 * mb + (1 - b1) * db, b2 * vb + (1 - b2) * db ** 2
            mWh, vWh = mW / (1 - b1 ** t), vW / (1 - b2 ** t)
            mbh, vbh = mb / (1 - b1 ** t), vb / (1 - b2 ** t)
            W, b = params
            W = W - learning_rate * mWh / (jnp.sqrt(vWh) + eps)
            b = b - learning_rate * mbh / (jnp.sqrt(vbh) + eps)
            params = (W, b)
            loss = MultinomialLogisticRegression._loss(params, X, Y, w)
            return (params, (mW, mb), (vW, vb)), loss

        ts = jnp.arange(1, num_epochs + 1, dtype=float)
        (params, _, _), loss_history = jax.lax.scan(step, (params, zeros, zeros), ts)
        return params, loss_history

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
            params = (jnp.zeros((X.shape[1], K)), jnp.zeros(K))
        else:
            W, b = init_params
            W = jnp.asarray(W, dtype=float)
            b = jnp.asarray(b, dtype=float)
            if W.shape != (X.shape[1], K):
                raise ValueError(
                    f"init_params W has shape {W.shape}, expected ({X.shape[1]}, {K})."
                )
            params = (W, b)
        self.params, self.loss_history_ = self._run(
            params, X, Y, w, self.num_epochs, self.learning_rate
        )
        return self

    def predict(self, X):
        if self.params is None:
            raise RuntimeError("Model is not fitted yet; call `fit` first.")
        return self._predict(self.params, self._prep(X))
