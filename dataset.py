import argparse
import json
import os
 
import numpy as np
import torch
 
from energy import DW_energy, LJ_energy, EnergyWrapper
from samplers import MH, HMC
from utils import (center, init_gaussian, init_cluster, prerelax, run_mcmc, save_reference,
                   load_run, is_complete)
 
def flatten(x):
    return x.reshape(x.shape[0], -1)
 
def batched(pairs, batch_size, drop_last=False):
    xs, ys = [], []
    for x, y in pairs:
        xs.append(x)
        ys.append(y)
        if len(xs) == batch_size:
            yield torch.stack(xs), torch.stack(ys)
            xs, ys = [], []
    if xs and not drop_last:
        yield torch.stack(xs), torch.stack(ys)
 
def take(pairs, k):
    X, Y = next(batched(pairs, k))
    return X, Y
 
 
class RandomRotation:
    "Rotation augmentation"
 
    def __init__(self, seed=0):
        self.g = torch.Generator().manual_seed(seed)
 
    def __call__(self, x):
        B, _, d = x.shape
        A = torch.randn(B, d, d, generator=self.g, dtype=torch.float64)
        Q, R = torch.linalg.qr(A)
        Q = Q * torch.sign(torch.diagonal(R, dim1=-2, dim2=-1)).unsqueeze(-2) 
        det = torch.linalg.det(Q)
        Q[det < 0, :, 0] *= -1
        return torch.einsum("bnd,bed->bne", x, Q.to(x.dtype))
 
 
class RandomPermutation:
    "Random particle permutation"
    def __init__(self, seed=0):
        self.g = torch.Generator().manual_seed(seed)
 
    def __call__(self, x):
        B, n, _ = x.shape
        idx = torch.argsort(torch.rand(B, n, generator=self.g), dim=1)
        return torch.gather(x, 1, idx[..., None].expand_as(x))
 
 
class Scale:
    "Constant scaling of all coordinates (normalisation); y stays the physical energy"
    def __init__(self, factor):
        self.factor = factor
 
    def __call__(self, x):
        return x * self.factor
 
def load_file(path):
    if os.path.isdir(path):
        X = np.load(os.path.join(path, "x.npy"), mmap_mode="r")
        get = lambda f: (np.load(os.path.join(path, f), mmap_mode="r") if os.path.exists(os.path.join(path, f)) else None)
        return X, get("chain.npy"), get("energy.npy")
    return np.load(path, mmap_mode="r", allow_pickle=False), None, None
 
 
def train_test_splits(N, fractions=(0.8, 0.1, 0.1), chain=None, seed=0):
    "train, val, test"
    fr = np.asarray(fractions, float) / np.sum(fractions)
    rng = np.random.default_rng(seed)
    if chain is None:
        perm = rng.permutation(N)
        cuts = np.round(np.cumsum(fr)[:-1] * N).astype(int)
        return tuple(np.sort(p) for p in np.split(perm, cuts))
    chain = np.asarray(chain)
    ids = rng.permutation(np.unique(chain))
    cuts = np.round(np.cumsum(fr)[:-1] * len(ids)).astype(int)
    return tuple(np.flatnonzero(np.isin(chain, part)) for part in np.split(ids, cuts))
 
 
def load_reference_splits(path, energy_fn, n, d, transforms=(center,), fractions=(0.8, 0.1, 0.1), seed=0, shuffle_train=True, epoch=0, chunk=1024, dtype=torch.float32, by_chain=True):
    "lazy generators (train, val, test), each with (x_i, y_i)"
    X, chain, E_stored = load_file(path)
    splits = train_test_splits(len(X), fractions, chain if by_chain else None, seed)
 
    def gen(idx, shuffle):
        if shuffle:
            idx = np.random.default_rng((seed, epoch)).permutation(idx)
        for s in range(0, len(idx), chunk):
            ib = idx[s:s + chunk]
            order = np.argsort(ib)                                   # sorted reads from the memmap
            raw = np.empty((len(ib),) + X.shape[1:], dtype=X.dtype)
            raw[order] = X[ib[order]]
            xb = torch.as_tensor(raw, dtype=torch.float64).reshape(-1, n, d)
            if energy_fn is not None:
                with torch.no_grad():
                    yb = energy_fn(xb)
            else:
                yb = torch.as_tensor(np.asarray(E_stored[ib]))
            for t in transforms:
                xb = t(xb)
            xb, yb = xb.to(dtype), yb.to(dtype)
            for i in range(len(ib)):
                yield xb[i], yb[i]
 
    train, val, test = splits
    return gen(train, shuffle_train), gen(val, False), gen(test, False)
 
 
 
# ---------------------------------------------------------------------------
# Generating reference data (MCMC), with an on-disk cache (Claude generated)
# ---------------------------------------------------------------------------
SYSTEMS = {
    #       energy, particles, dims, start configurations, tuned sampler settings, default steps
    "dw4": dict(energy=DW_energy, n=4, d=2,
                init=lambda B, seed: init_gaussian(B, 4, 2, scale=2.5, seed=seed),
                hmc=dict(step_size=0.1, traj_len=1.0), mh=dict(step_size=0.3), steps=20000),
    "lj13": dict(energy=LJ_energy, n=13, d=3,
                 init=lambda B, seed: init_cluster(B, 13, 3, seed=seed),
                 hmc=dict(step_size=0.02, traj_len=0.5), mh=dict(step_size=0.05), steps=10000),
}
 
 
def reference_config(system, sampler="hmc", chains=512, warmup=2000, steps=None, thin=10, seed=42,
                     burn_in=0, step_size=None, traj_len=None):
    """All settings of a reference run, with the system defaults filled in."""
    spec = SYSTEMS[system]
    kw = dict(spec[sampler])
    if step_size is not None:
        kw["step_size"] = step_size
    if traj_len is not None and sampler == "hmc":
        kw["traj_len"] = traj_len
    return dict(system=system, sampler=sampler, chains=chains, warmup=warmup,
                steps=steps or spec["steps"], thin=thin, seed=seed, burn_in=burn_in,
                **{"init_" + k: v for k, v in kw.items()})
 
 
def generate_reference(folder, config, log_every=1000):
    """Run MCMC for `config` (from reference_config) and save it to `folder`."""
    spec = SYSTEMS[config["system"]]
    target = EnergyWrapper(spec["energy"], p=spec["n"], d=spec["d"])
    kw = {k[5:]: v for k, v in config.items() if k.startswith("init_")}
    sampler = HMC(**kw) if config["sampler"] == "hmc" else MH(**kw)
    x0 = prerelax(target, spec["init"](config["chains"], config["seed"]))   # removes far-out starts
    res = run_mcmc(target, sampler, x0, warmup=config["warmup"], steps=config["steps"],
                   thin=config["thin"], seed=config["seed"], log_every=log_every)
    return save_reference(folder, res, spec["n"], spec["d"], energy_fn=spec["energy"],
                          burn_in=config["burn_in"], extra_meta={"config": config})
 
 
def get_reference(system, folder, force=False, samples=False, log_every=1000, **settings):
    """Load the reference run in `folder` if it is complete, otherwise run MCMC and save it.
    settings: see reference_config (sampler, chains, warmup, steps, thin, seed, ...).
    A complete folder is always loaded (force=True regenerates); if it was made with other
    settings, the differences are printed. Returns load_run(folder, samples)."""
    config = reference_config(system, **settings)
    if is_complete(folder) and not force:
        res = load_run(folder, samples=samples)
        stored = res["meta"].get("config")
        if stored is None:
            print(f"loaded {folder} (saved without its settings; force=True regenerates it)")
        else:
            diff = {k: (stored.get(k), v) for k, v in config.items() if stored.get(k) != v}
            print(f"loaded {folder}" + (f"; settings differ (stored, requested): {diff}, "
                                         "force=True regenerates it" if diff else ""))
        return res
    print(f"running {config['sampler'].upper()} for {system} -> {folder}")
    generate_reference(folder, config, log_every=log_every)
    return load_run(folder, samples=samples)
 
 
def main(argv=None):
    ap = argparse.ArgumentParser(description="Generate MCMC reference samples (skipped if the "
                                             "output folder is already complete).")
    ap.add_argument("system", choices=SYSTEMS)
    ap.add_argument("--out", required=True)
    ap.add_argument("--sampler", choices=["hmc", "mh"], default="hmc")
    ap.add_argument("--chains", type=int, default=512)
    ap.add_argument("--warmup", type=int, default=2000)
    ap.add_argument("--steps", type=int, default=None, help="default: 20000 (dw4) / 10000 (lj13)")
    ap.add_argument("--thin", type=int, default=10)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--burn-in", type=int, default=0, help="stored samples to drop per chain")
    ap.add_argument("--step-size", type=float, default=None, help="initial step size")
    ap.add_argument("--traj-len", type=float, default=None, help="HMC trajectory length")
    ap.add_argument("--force", action="store_true", help="regenerate even if complete")
    a = ap.parse_args(argv)
    res = get_reference(a.system, a.out, force=a.force, sampler=a.sampler, chains=a.chains,
                        warmup=a.warmup, steps=a.steps, thin=a.thin, seed=a.seed,
                        burn_in=a.burn_in, step_size=a.step_size, traj_len=a.traj_len)
    print(json.dumps(res["summary"], indent=1))
 
 
if __name__ == "__main__":
    main()