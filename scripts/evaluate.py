"""Compute all metrics for the Milestone 1 report and write them to results/metrics.json.

Every metric is computed on 3 independent random subsets (seed 0, 1, 2) and reported as mean ± std.
"""
import json
import os
import sys

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from adjsamp.data import chain_split, load_idem  # noqa: E402
from adjsamp.energies import get_energy  # noqa: E402
from adjsamp.mcmc import MH, BoltzmannTarget, run_chains  # noqa: E402
from adjsamp.metrics import energy_w2, geometric_w2  # noqa: E402
from scripts.run_mcmc import initial_state  # noqa: E402

DEV = "cuda" if torch.cuda.is_available() else "cpu"


def ref_split(path, split):
    s = np.load(os.path.join(path, "samples.npy"), mmap_mode="r")
    e = np.load(os.path.join(path, "energies.npy"), mmap_mode="r")
    idx = chain_split(s.shape[1])[split]
    return np.asarray(s[:, idx]).reshape(-1, s.shape[-1]), np.asarray(e[:, idx]).reshape(-1)


def metrics(x, e, x_ref, e_ref, n, d, n_geo, seeds=(0, 1, 2)):
    ew, gw = [], []
    for s in seeds:
        rng = np.random.default_rng(100 + s)
        ia = rng.choice(len(e), min(len(e), 10_000), replace=False)
        ib = rng.choice(len(e_ref), min(len(e_ref), 10_000), replace=False)
        ew.append(energy_w2(e[ia], e_ref[ib]))
        gw.append(geometric_w2(x, x_ref, n, d, max_samples=n_geo, seed=s, device=DEV))
    return {"energy_w2": [float(np.mean(ew)), float(np.std(ew))],
            "geom_w2": [float(np.mean(gw)), float(np.std(gw))]}


DW4_MINIMA = {"kite": -25.79, "triangle+tail": -25.31, "rectangle": -24.35, "rhombus": -23.47,
              "centred triangle": -21.07}


def dw4_basin_occupancy(x, n=2000, seed=0):
    """Fraction of samples whose steepest-descent minimum is each of the 5 DW-4 minima (else 'other')."""
    from adjsamp.plotting import quench
    x = np.asarray(x)
    k = np.random.default_rng(seed).choice(len(x), min(n, len(x)), replace=False)
    _, emin = quench(get_energy("dw4"), x[k], n_steps=6000, lr=5e-3, device=DEV)
    occ = {}
    for name, e0 in DW4_MINIMA.items():
        occ[name] = float(np.mean(np.abs(emin - e0) < 0.05))
    occ["other"] = 1 - sum(occ.values())
    return occ


def budget_matched_mh(system, n_evals, n_chains, seed=0):
    """Plain MH with the same number of energy evaluations as a trained sampler (final chain states)."""
    energy = get_energy(system)
    gen = torch.Generator().manual_seed(seed)
    torch.manual_seed(seed)
    x0 = initial_state(system, n_chains, gen, DEV)
    steps = max(n_evals // n_chains - 1, 1)
    res = run_chains(MH(0.1), BoltzmannTarget(energy), x0, n_warmup=steps, n_steps=1, thin=1, trace_steps=1)
    x = res["x_final"]
    return x, energy(torch.as_tensor(x)).numpy(), steps


def main():
    out = {}
    for system, n, d, n_geo, hmc in [("dw4", 4, 2, 1000, "data/reference/dw4_hmc"),
                                     ("lj13", 13, 3, 1000, "data/reference/lj13_hmc")]:
        energy = get_energy(system)
        x_te, e_te = ref_split(hmc, "test")
        x_va, e_va = ref_split(hmc, "val")
        x_mh, e_mh = ref_split(hmc.replace("_hmc", "_mh"), "test")
        x_id = load_idem(system, "test")
        e_id = energy(torch.as_tensor(x_id)).numpy()
        rows = {
            "HMC val vs HMC test (noise floor)": metrics(x_va, e_va, x_te, e_te, n, d, n_geo),
            "MH test vs HMC test": metrics(x_mh, e_mh, x_te, e_te, n, d, n_geo),
            "iDEM test vs HMC test": metrics(x_id, e_id, x_te, e_te, n, d, n_geo),
        }
        if system == "dw4":
            for arch in ("egnn", "mlp"):
                run = f"runs/dw4_{arch}_final"
                if os.path.exists(os.path.join(run, "samples.npy")):
                    xs, es = np.load(os.path.join(run, "samples.npy")), np.load(os.path.join(run, "energies.npy"))
                    cfg = json.load(open(os.path.join(run, "config.json")))
                    m = metrics(xs, es, x_te, e_te, n, d, n_geo)
                    m["energy_evals"] = cfg["energy_evals"]
                    rows[f"Adjoint Sampling prototype ({arch.upper()})"] = m
                    budget = cfg["energy_evals"]
            xb, eb, steps = budget_matched_mh("dw4", budget, 1024)
            m = metrics(xb, eb, x_te, e_te, n, d, n_geo)
            m["energy_evals"] = budget
            m["mh_steps_per_chain"] = steps
            rows["MH, same energy budget (1024 chains)"] = m
            samples = {"HMC val vs HMC test (noise floor)": x_va, "iDEM test vs HMC test": x_id,
                       "MH, same energy budget (1024 chains)": xb}
            for arch in ("egnn", "mlp"):
                samples[f"Adjoint Sampling prototype ({arch.upper()})"] = np.load(f"runs/dw4_{arch}_final/samples.npy")
            for k, xs in samples.items():
                rows[k]["basins"] = dw4_basin_occupancy(xs)
            rows["reference basins (HMC test)"] = {"basins": dw4_basin_occupancy(x_te)}
        out[system] = rows
        for k, v in rows.items():
            if "energy_w2" in v:
                print(f"{system:5s} {k:45s} E-W2 {v['energy_w2'][0]:.3f}±{v['energy_w2'][1]:.3f}  "
                      f"W2 {v['geom_w2'][0]:.3f}±{v['geom_w2'][1]:.3f}")
            if "basins" in v:
                print("      basins:", {b: round(f, 3) for b, f in v["basins"].items()})
    os.makedirs("results", exist_ok=True)
    with open("results/metrics.json", "w") as f:
        json.dump(out, f, indent=2)


if __name__ == "__main__":
    main()
