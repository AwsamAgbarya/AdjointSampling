"""Evaluation metrics: energy W2, symmetry-aware (geometric) W2 and pair-distance histograms.

Geometric W2 (Havens et al., 2025, App. E.4; Köhler et al., 2020):
    d(x, y)^2 = min_{R in O(d), P in S(n)} || x - (R ⊗ P) y ||^2  for centred x, y,
    W2(mu, nu)^2 = min_{pi} E_pi[d(x, y)^2]  (exact discrete OT between two equal-size sample sets).

Implementation notes (both make the N x M cost matrix cheap):
* For a fixed permutation the optimal *orthogonal* map has a closed form: with H = Y^T X,
  min_R ||X - Y R||^2 = ||X||^2 + ||Y||^2 - 2 ||H||_*  (nuclear norm; no det-correction is needed
  because reflections are allowed).  We batch this over all pairs with ``torch.linalg.svdvals``.
* DW-4 has only 4! = 24 permutations, so we minimise over all of them *exactly*.  For LJ-13
  (13! permutations) we use the sequential approximation of the paper: P~ = argmin_P ||x - P y||^2
  (Hungarian algorithm on particle distances), then the optimal rotation for P~.
"""
from __future__ import annotations

import itertools

import numpy as np
import torch
from scipy.optimize import linear_sum_assignment


def energy_w2(e1, e2, n_quantiles: int = 2000) -> float:
    """1D Wasserstein-2 distance between two empirical energy distributions (quantile coupling)."""
    q = (np.arange(n_quantiles) + 0.5) / n_quantiles
    a = np.quantile(np.asarray(e1, dtype=np.float64), q)
    b = np.quantile(np.asarray(e2, dtype=np.float64), q)
    return float(np.sqrt(np.mean((a - b) ** 2)))


def _center(p):
    return p - p.mean(-2, keepdim=True)


def _o_d_aligned_sqdist(X, Y):
    """min_{R in O(d)} ||X - Y R||^2 for batches X, Y of shape (..., n, d)."""
    H = Y.transpose(-1, -2) @ X
    if H.shape[-1] == 2:  # closed form for 2x2: s1 + s2 = sqrt(||H||_F^2 + 2 |det H|)
        det = H[..., 0, 0] * H[..., 1, 1] - H[..., 0, 1] * H[..., 1, 0]
        nuc = torch.sqrt((H**2).sum((-1, -2)) + 2 * det.abs())
    else:
        nuc = torch.linalg.svdvals(H).sum(-1)
    return (X**2).sum((-1, -2)) + (Y**2).sum((-1, -2)) - 2 * nuc


@torch.no_grad()
def symmetry_aware_cost_matrix(x, y, n_particles: int, spatial_dim: int, exact_perm: bool | None = None,
                               device: str | None = None, chunk: int = 64):
    """Matrix C[i, j] = d(x_i, y_j)^2 of symmetry-aware squared distances. x: (N, dim), y: (M, dim)."""
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    X = _center(torch.as_tensor(np.asarray(x), dtype=torch.float64, device=device).reshape(-1, n_particles, spatial_dim))
    Y = _center(torch.as_tensor(np.asarray(y), dtype=torch.float64, device=device).reshape(-1, n_particles, spatial_dim))
    if exact_perm is None:
        exact_perm = n_particles <= 6
    N, M = X.shape[0], Y.shape[0]
    C = torch.empty(N, M, dtype=torch.float64)
    if exact_perm:
        perms = torch.tensor(list(itertools.permutations(range(n_particles))), device=device)
        Yp = Y[:, perms]  # (M, P, n, d)
        for s in range(0, N, chunk):
            Xi = X[s:s + chunk, None, None]  # (c, 1, 1, n, d)
            C[s:s + chunk] = _o_d_aligned_sqdist(Xi, Yp[None]).min(-1).values.cpu()
    else:
        Xn, Yn = X.cpu().numpy(), Y.cpu().numpy()
        for i in range(N):
            D = ((Xn[i][:, None, :] - Yn[:, None, :, :]) ** 2).sum(-1)  # (M, n, n) particle distances
            perm = np.stack([linear_sum_assignment(D[j])[1] for j in range(M)])  # Hungarian
            Yp = torch.as_tensor(np.take_along_axis(Yn, perm[:, :, None], axis=1), device=device)
            C[i] = _o_d_aligned_sqdist(X[i][None].expand_as(Yp), Yp).cpu()
    return C.clamp_min(0).numpy()


def geometric_w2(x, y, n_particles: int, spatial_dim: int, max_samples: int | None = 1000,
                 seed: int = 0, **kw) -> float:
    """Symmetry-aware W2 between two sample sets (sub-sampled to equal size ``max_samples``)."""
    rng = np.random.default_rng(seed)
    n = min(len(x), len(y), max_samples or len(x))
    x = np.asarray(x)[rng.choice(len(x), n, replace=False)]
    y = np.asarray(y)[rng.choice(len(y), n, replace=False)]
    C = symmetry_aware_cost_matrix(x, y, n_particles, spatial_dim, **kw)
    r, c = linear_sum_assignment(C)
    return float(np.sqrt(C[r, c].mean()))


def pair_distances_np(x, n_particles: int, spatial_dim: int) -> np.ndarray:
    p = np.asarray(x).reshape(-1, n_particles, spatial_dim)
    i, j = np.triu_indices(n_particles, 1)
    return np.linalg.norm(p[:, i] - p[:, j], axis=-1)


def histogram_tv(a, b, bins) -> float:
    """Total-variation distance between two histograms on shared bins (simple sanity measure)."""
    ha, _ = np.histogram(a, bins=bins)
    hb, _ = np.histogram(b, bins=bins)
    ha = ha / max(ha.sum(), 1)
    hb = hb / max(hb.sum(), 1)
    return float(0.5 * np.abs(ha - hb).sum())
