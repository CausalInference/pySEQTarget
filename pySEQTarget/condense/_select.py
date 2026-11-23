import time

import numpy as np
import torch

# byte budgets for row and unit chunks
_ROW_BYTES = 2**29
_UNIT_BYTES = 2**32


def _rates(X, probes):
    # per row class probabilities and their variances at every probe
    first, second = [], []
    for B in probes:
        eta = torch.einsum("...d,pdm->...pm", X, B)
        zero = torch.zeros_like(eta[..., :1])
        p = torch.softmax(torch.cat([zero, eta], -1), -1)[..., 1:]
        V = torch.diag_embed(p) - p[..., :, None] * p[..., None, :]
        first.append(p.flatten(-2))
        second.append(V.flatten(-3))
    return torch.cat(first, -1), torch.cat(second, -1)


def _targets(X, w, seg, probes, n_units, S=None):
    D = X.shape[1]
    G = H = C = None
    widest = D if S is None else max(D, S.shape[1])
    step = max(1, int(_ROW_BYTES / (8 * D * widest * 4)))
    for start in range(0, len(X), step):
        rows = slice(start, start + step)
        x, s = X[rows], seg[rows]
        first, second = _rates(x, probes)
        if G is None:
            G = X.new_zeros(n_units, first.shape[1], D)
            H = X.new_zeros(n_units, second.shape[1], D, D)
            if S is not None:
                C = X.new_zeros(n_units, first.shape[1], D, S.shape[1])
        v = w[rows, None] * first
        G.index_add_(0, s, v[:, :, None] * x[:, None, :])
        xx = x[:, :, None] * x[:, None, :]
        vv = w[rows, None] * second
        for b in range(second.shape[1]):
            H[:, b].index_add_(0, s, vv[:, b, None, None] * xx)
        if S is not None:
            xs = x[:, :, None] * S[rows][:, None, :]
            for b in range(first.shape[1]):
                C[:, b].index_add_(0, s, v[:, b, None, None] * xs)
    return G, H, C


def _nnls(G, c):
    m = G.shape[-1]
    eye = torch.eye(m, dtype=G.dtype, device=G.device)
    jitter = 1e-10 * G.diagonal(dim1=1, dim2=2).mean(1)[:, None, None] * eye
    active = torch.ones_like(c, dtype=torch.bool)
    for _ in range(m):
        both = active[:, :, None] & active[:, None, :]
        w = torch.linalg.solve(
            torch.where(both, G, eye) + jitter, torch.where(active, c, 0.0)[..., None]
        ).squeeze(-1)
        negative = (w < 0) & active
        if not negative.any():
            break
        active = active & ~negative
    return torch.where(active, w, 0.0).clamp_min(0.0)


def _select(W, X, w, valid, probes, k):
    Z = X @ W
    first, second = _rates(X, probes)
    Q = Z @ Z.transpose(1, 2)
    G = (first @ first.transpose(1, 2)) * Q
    G += (second @ second.transpose(1, 2)) * Q**2
    del Q, first, second, Z
    c = (G @ w[..., None]).squeeze(-1)
    norm = G.diagonal(dim1=1, dim2=2).clamp_min(1e-300).sqrt()
    U, n = w.shape
    units = torch.arange(U, device=w.device)
    chosen = ~valid
    current = torch.zeros_like(w)
    picks = torch.zeros((U, k), dtype=torch.long, device=w.device)
    for j in range(k):
        score = (c - (G @ current[..., None]).squeeze(-1)) / norm
        score[chosen] = -torch.inf
        p = score.argmax(1)
        picks[:, j] = p
        chosen[units, p] = True
        S = picks[:, : j + 1]
        Gs = G.gather(1, S[:, :, None].expand(-1, -1, n)).gather(
            2, S[:, None, :].expand(-1, j + 1, -1)
        )
        ws = _nnls(Gs, c.gather(1, S))
        current = torch.zeros_like(w).scatter(1, S, ws)
    return picks, ws


def _sensitivities(W, X, w, probes, C, S0, ridge):
    v = w[..., None] * _rates(X, probes)[0]
    Z = X @ W
    # M is unit probe d k
    M = (v.transpose(1, 2)[..., None] * Z[:, None]).transpose(-1, -2)
    M = M.reshape(len(X), -1, X.shape[1])
    R = torch.einsum("de,ubdj->ubej", W, C).reshape(len(X), M.shape[1], -1)
    A = M.transpose(1, 2) @ M
    K = A.shape[-1]
    lam = (ridge * A.diagonal(dim1=1, dim2=2).sum(1) / K + 1e-300)[:, None, None]
    eye = torch.eye(K, dtype=A.dtype, device=A.device)
    return torch.linalg.solve(A + lam * eye, M.transpose(1, 2) @ R + lam * S0)


def _relative(W, X, w, probes, GW, HW, Sk=None, C=None):
    first, second = _rates(X, probes)
    Z = X @ W
    wm = (w[..., None] * first).transpose(1, 2)
    eg = wm @ Z - GW
    A = (w[..., None] * second).transpose(1, 2)[..., None] * Z[:, None]
    eH = A.transpose(-1, -2) @ Z[:, None] - HW
    rel_g = torch.sqrt((eg**2).flatten(1).sum(1) / (GW**2).flatten(1).sum(1))
    rel_H = torch.sqrt((eH**2).flatten(1).sum(1) / (HW**2).flatten(1).sum(1))
    if Sk is None:
        return rel_g.cpu().numpy(), rel_H.cpu().numpy(), None
    Ct = (wm[..., None] * X[:, None]).transpose(-1, -2) @ Sk[:, None]
    eC = torch.einsum("de,ubdj->ubej", W, Ct - C)
    Cw = torch.einsum("de,ubdj->ubej", W, C)
    rel_C = torch.sqrt((eC**2).flatten(1).sum(1) / (Cw**2).flatten(1).sum(1))
    return rel_g.cpu().numpy(), rel_H.cpu().numpy(), rel_C.cpu().numpy()


@torch.no_grad()
def select_atom(atom, X, w, seg, starts, k, Sa, ridge, timings, device):
    device = torch.device(device)

    def t(a):
        return torch.tensor(np.array(a), dtype=torch.float64, device=device)

    W = t(atom.W)
    train, test = [t(B) for B in atom.train], [t(B) for B in atom.test]
    n_units = len(starts)
    Xt, wt = t(X), t(w)
    segt = torch.as_tensor(seg, dtype=torch.long, device=device)
    St = None if Sa is None else t(Sa)
    starts_t = torch.as_tensor(starts, dtype=torch.long, device=device)
    counts = np.diff(np.r_[starts, len(seg)])
    counts_t = torch.as_tensor(counts, dtype=torch.long, device=device)

    start = time.perf_counter()
    C = None
    if Sa is not None:
        C = _targets(Xt, wt, segt, train, n_units, St)[2]
    Gt, Ht, Ct = _targets(Xt, wt, segt, test, n_units, St)
    GWt, HWt = Gt @ W, W.T @ Ht @ W
    del Gt, Ht
    timings["targets"] += time.perf_counter() - start

    start = time.perf_counter()
    # start from each units first k rows
    ar = torch.arange(k, device=device)
    mask = ar[None, :] < counts_t[:, None]
    first = torch.where(mask, starts_t[:, None] + ar[None, :], 0)
    Xs, ws = Xt[first], wt[first] * mask
    Ss = None if Sa is None else St[first] * mask[..., None]

    free = np.flatnonzero(counts > k)
    free = free[np.argsort(-counts[free], kind="stable")]
    i = 0
    while i < len(free):
        # chunk units of similar length
        n = int(counts[free[i]])
        size = max(1, int(_UNIT_BYTES / (8 * n * n * 6)))
        idx = torch.as_tensor(free[i : i + size], device=device)
        i += size
        ar = torch.arange(n, device=device)
        valid = ar[None, :] < counts_t[idx][:, None]
        rows = torch.where(valid, starts_t[idx][:, None] + ar[None, :], 0)
        picks, weights = _select(W, Xt[rows], wt[rows] * valid, valid, train, k)
        chosen = rows.gather(1, picks)
        Xs[idx], ws[idx], mask[idx] = Xt[chosen], weights, weights > 0
        if Sa is not None:
            Ss[idx] = St[chosen]

    P = sum(B.shape[0] * B.shape[2] ** 2 for B in train)
    step = max(1, int(_UNIT_BYTES / (8 * P * k * W.shape[0] * 12)))
    if Sa is not None:
        for j in range(0, len(free), step):
            idx = torch.as_tensor(free[j : j + step], device=device)
            Ss[idx] = _sensitivities(W, Xs[idx], ws[idx], train, C[idx], Ss[idx], ridge)
    if device.type == "cuda":
        torch.cuda.synchronize()
    timings["condense"] += time.perf_counter() - start

    parts = []
    for j in range(0, n_units, step):
        s = slice(j, j + step)
        parts.append(
            _relative(
                W,
                Xs[s],
                ws[s],
                test,
                GWt[s],
                HWt[s],
                None if Sa is None else Ss[s],
                None if Sa is None else Ct[s],
            )
        )
    errors = tuple(None if p[0] is None else np.concatenate(p) for p in zip(*parts))
    return (
        Xs.cpu().numpy(),
        ws.cpu().numpy(),
        mask.cpu().numpy(),
        counts,
        None if Sa is None else Ss.cpu().numpy(),
        errors,
    )
