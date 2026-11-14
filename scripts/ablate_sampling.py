"""Ablation on DW-4: number of Euler-Maruyama steps at sampling time and the time-sampling scheme.

Re-samples trained models with K in {100, 200, 500, 1000} steps and writes results/ablation_sampling.json.
"""
import json
import os
import sys

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from adjsamp.energies import DoubleWell  # noqa: E402
from adjsamp.metrics import energy_w2  # noqa: E402
from adjsamp.model import Model  # noqa: E402
from adjsamp.sde import BaseProcess, GeometricSchedule, euler_maruyama  # noqa: E402
from scripts.evaluate import ref_split  # noqa: E402

RUNS = {"t ~ U[0,1] (paper), trained with K=200": "runs/dw4_egnn",
        "t uniform in nu, trained with K=200": "runs/dw4_egnn_tnu",
        "t ~ U[0,1] (paper), trained with K=1000": "runs/dw4_egnn_final"}


def main():
    _, e_ref = ref_split("data/reference/dw4_hmc", "test")
    E = DoubleWell()
    out = {"reference_median_energy": float(np.median(e_ref))}
    for name, run in RUNS.items():
        if not os.path.exists(os.path.join(run, "model.pt")):
            continue
        cfg = json.load(open(os.path.join(run, "config.json")))
        s = GeometricSchedule(cfg["sigma_min"], cfg["sigma_max"])
        base = BaseProcess(s, 4, 2)
        m = Model(4, 2, s, cfg["arch"], cfg["hidden"], cfg["layers"]).cuda()
        m.load_state_dict(torch.load(os.path.join(run, "model.pt")))
        m.eval()
        out[name] = {}
        for K in (100, 200, 500, 1000):
            torch.manual_seed(0)
            x = torch.cat([euler_maruyama(m, base, 2500, K, device="cuda") for _ in range(2)])
            e = E(x).cpu().numpy()
            out[name][K] = {"median_energy": float(np.median(e)), "energy_w2": energy_w2(e, e_ref),
                            "frac_E_gt_-10": float(np.mean(e > -10))}
            print(f"{name:42s} K={K:5d} median {np.median(e):7.2f}  E-W2 {out[name][K]['energy_w2']:.3f}")
    os.makedirs("results", exist_ok=True)
    with open("results/ablation_sampling.json", "w") as f:
        json.dump(out, f, indent=2)


if __name__ == "__main__":
    main()
