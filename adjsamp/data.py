"""Reference data: storage, chain-wise train/val/test splits and lazy generators.

Storage format (``save_reference``): a directory with
    samples.npy   float32, (n_keep, n_chains, dim)  — thinned chain states, memory-mappable
    energies.npy  float32, (n_keep, n_chains)       — E(x) of the stored states
    meta.json     sampler settings and diagnostics

Splitting.  Consecutive states of one chain are strongly correlated, so a random split of
*samples* would put near-duplicates into train and test.  We instead split *chains*: every chain
is independent of every other one (independent initialisations and random streams), so assigning
whole chains to train/val/test makes the three sets statistically independent.  Within a chain
the stored states are additionally thinned (``stride``) to roughly one integrated
autocorrelation time, so that the samples inside a split are close to independent as well.
"""
from __future__ import annotations

import json
import os
from typing import Callable, Iterator, Sequence

import numpy as np
import torch


def save_reference(out_dir: str, result: dict, meta: dict | None = None):
    os.makedirs(out_dir, exist_ok=True)
    np.save(os.path.join(out_dir, "samples.npy"), result["samples"].astype(np.float32))
    np.save(os.path.join(out_dir, "energies.npy"), result["energies"].astype(np.float32))
    np.save(os.path.join(out_dir, "energy_trace.npy"), result["trace"].astype(np.float32))
    np.save(os.path.join(out_dir, "warmup_trace.npy"), result["warmup_energies"].astype(np.float32))
    info = {k: v for k, v in result.items() if np.isscalar(v) or isinstance(v, str)}
    info["accept_rate_mean"] = float(np.mean(result["accept_rate"]))
    info.update(meta or {})
    with open(os.path.join(out_dir, "meta.json"), "w") as f:
        json.dump(info, f, indent=2)


def chain_split(n_chains: int, fractions=(0.8, 0.1, 0.1), seed: int = 0):
    """Disjoint sets of chain indices for train / val / test."""
    perm = np.random.default_rng(seed).permutation(n_chains)
    n_tr = int(round(fractions[0] * n_chains))
    n_va = int(round(fractions[1] * n_chains))
    return {"train": np.sort(perm[:n_tr]), "val": np.sort(perm[n_tr:n_tr + n_va]),
            "test": np.sort(perm[n_tr + n_va:])}


# --- transforms --------------------------------------------------------------------------------
# A transform maps a single configuration (dim,) -> (dim,).  E(x) is invariant under all of them.

def center_transform(n_particles: int, spatial_dim: int) -> Callable:
    def f(x):
        p = x.reshape(n_particles, spatial_dim)
        return (p - p.mean(0, keepdims=True)).reshape(-1)
    return f


def random_rotation_transform(n_particles: int, spatial_dim: int, seed: int = 0) -> Callable:
    """Data augmentation: random orthogonal matrix (Haar measure on O(d))."""
    rng = np.random.default_rng(seed)

    def f(x):
        q, r = np.linalg.qr(rng.standard_normal((spatial_dim, spatial_dim)))
        q = q * np.sign(np.diag(r))
        return (x.reshape(n_particles, spatial_dim) @ q.T).reshape(-1).astype(x.dtype)
    return f


def random_permutation_transform(n_particles: int, spatial_dim: int, seed: int = 0) -> Callable:
    rng = np.random.default_rng(seed)

    def f(x):
        return x.reshape(n_particles, spatial_dim)[rng.permutation(n_particles)].reshape(-1)
    return f


def compose(transforms: Sequence[Callable]) -> Callable:
    def f(x):
        for t in transforms:
            x = t(x)
        return x
    return f


# --- lazy loading ------------------------------------------------------------------------------
def _sample_generator(samples, energies, chains, stride, transform, energy_fn, shuffle, seed):
    n_keep = samples.shape[0]
    idx = np.array([(k, c) for c in chains for k in range(0, n_keep, stride)])
    if shuffle:
        np.random.default_rng(seed).shuffle(idx)
    for k, c in idx:
        x = np.array(samples[k, c], dtype=np.float32)  # reads only this row from the memmap
        x = transform(x) if transform is not None else x
        if energy_fn is not None:
            y = float(energy_fn(torch.from_numpy(x)[None]).item())
        else:
            y = float(energies[k, c])
        yield x, y


def load_reference_splits(ref_dir: str, transform: Callable | None = None, fractions=(0.8, 0.1, 0.1),
                          stride: int = 1, seed: int = 0, shuffle: bool = True,
                          energy_fn: Callable | None = None):
    """Read reference samples from ``ref_dir`` and return three lazy generators.

    Each generator yields ``(x_i, y_i)`` with ``x_i`` a (transformed) configuration of shape (dim,)
    and ``y_i = E(x_i)``.  The arrays are opened as memory maps, so nothing is loaded until a
    sample is requested.  If ``energy_fn`` is given, y_i is recomputed on the transformed sample
    (useful as a check); otherwise the stored energy is returned (valid because all provided
    transforms leave E invariant).
    """
    samples = np.load(os.path.join(ref_dir, "samples.npy"), mmap_mode="r")
    energies = np.load(os.path.join(ref_dir, "energies.npy"), mmap_mode="r")
    split = chain_split(samples.shape[1], fractions, seed)
    return tuple(
        _sample_generator(samples, energies, split[name], stride, transform, energy_fn, shuffle, seed + i)
        for i, name in enumerate(("train", "val", "test"))
    )


def collect(gen: Iterator, n: int | None = None):
    """Materialise (part of) a generator into arrays X (N, dim), Y (N,)."""
    xs, ys = [], []
    for i, (x, y) in enumerate(gen):
        if n is not None and i >= n:
            break
        xs.append(x)
        ys.append(y)
    return np.stack(xs), np.array(ys)


def load_idem(system: str, split: str, root: str = "data/idem", center: bool = True):
    """Load an iDEM reference split (float32 array, shape (N, dim)); file read with allow_pickle=False."""
    name = {"dw4": "DW4", "lj13": "LJ13-1000"}[system.lower().replace("-", "")]
    x = np.load(os.path.join(root, f"{split}_split_{name}.npy"), allow_pickle=False).astype(np.float32)
    if center:
        n, d = (4, 2) if name == "DW4" else (13, 3)
        p = x.reshape(-1, n, d)
        x = (p - p.mean(1, keepdims=True)).reshape(x.shape)
    return x
