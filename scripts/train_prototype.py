"""Train the Adjoint Sampling prototype on DW-4 (Algorithm 1) and save model, history and samples.

Example:
    python scripts/train_prototype.py --arch egnn --n-outer 200 --n-inner 100
"""
import argparse
import json
import os
import sys
import time

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from adjsamp.data import collect, load_reference_splits  # noqa: E402
from adjsamp.energies import get_energy  # noqa: E402
from adjsamp.model import Model  # noqa: E402
from adjsamp.sde import BaseProcess, GeometricSchedule, euler_maruyama  # noqa: E402
from adjsamp.train import train_adjoint_sampling  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--system", default="dw4")
    ap.add_argument("--arch", default="egnn", choices=["egnn", "mlp"])
    ap.add_argument("--hidden", type=int, default=128)
    ap.add_argument("--layers", type=int, default=3)
    ap.add_argument("--n-outer", type=int, default=200)
    ap.add_argument("--n-inner", type=int, default=100)
    ap.add_argument("--n-per-outer", type=int, default=512)
    ap.add_argument("--batch-size", type=int, default=512)
    ap.add_argument("--buffer-size", type=int, default=10_000)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--sde-steps", type=int, default=200)
    ap.add_argument("--sigma-min", type=float, default=1e-4)
    ap.add_argument("--sigma-max", type=float, default=3.0)
    ap.add_argument("--max-grad-norm", type=float, default=100.0, help="per-sample clip of ||grad E||")
    ap.add_argument("--clip-param-grad", type=float, default=1.0)
    ap.add_argument("--t-sampling", default="uniform", choices=["uniform", "nu"])
    ap.add_argument("--n-final-samples", type=int, default=10_000)
    ap.add_argument("--ref", default="data/reference/dw4_hmc")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default=None)
    a = ap.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    energy = get_energy(a.system)
    sched = GeometricSchedule(a.sigma_min, a.sigma_max)
    base = BaseProcess(sched, energy.n_particles, energy.spatial_dim)
    model = Model(energy.n_particles, energy.spatial_dim, sched, arch=a.arch, hidden=a.hidden, n_layers=a.layers)
    n_params = sum(p.numel() for p in model.parameters())
    # validation energies (reference val split) to monitor training; the test split is kept for the final evaluation
    _, val_gen, _ = load_reference_splits(a.ref, stride=1)
    _, ref_e = collect(val_gen, 20_000)
    out = a.out or os.path.join("runs", f"{a.system}_{a.arch}_s{a.seed}")
    os.makedirs(out, exist_ok=True)
    print(f"{a.arch}: {n_params} parameters -> {out}")

    hist = train_adjoint_sampling(model, energy, base, n_outer=a.n_outer, n_inner=a.n_inner,
                                  n_per_outer=a.n_per_outer, batch_size=a.batch_size, buffer_size=a.buffer_size,
                                  lr=a.lr, n_sde_steps=a.sde_steps, max_grad_norm=a.max_grad_norm,
                                  clip_param_grad=a.clip_param_grad, t_sampling=a.t_sampling,
                                  ref_energies=ref_e, device=device, seed=a.seed)
    torch.save(model.state_dict(), os.path.join(out, "model.pt"))

    model.eval()
    t0 = time.perf_counter()
    xs = [euler_maruyama(model, base, 2000, a.sde_steps, device=device)
          for _ in range(a.n_final_samples // 2000)]
    x = torch.cat(xs)
    t_sample = time.perf_counter() - t0
    e = energy(x).cpu().numpy()
    np.save(os.path.join(out, "samples.npy"), x.cpu().numpy())
    np.save(os.path.join(out, "energies.npy"), e)
    _, traj = euler_maruyama(model, base, 64, a.sde_steps, device=device, return_traj=True, traj_every=5)
    np.save(os.path.join(out, "trajectories.npy"), traj.cpu().numpy())
    cfg = vars(a) | {"n_params": n_params, "sampling_seconds": t_sample,
                     "train_seconds": hist["seconds"][-1], "energy_evals": hist["n_energy_evals"][-1]}
    with open(os.path.join(out, "config.json"), "w") as f:
        json.dump(cfg, f, indent=2)
    with open(os.path.join(out, "history.json"), "w") as f:
        json.dump(hist, f)
    print(f"final samples: median E {np.median(e):.2f}, sampling {t_sample:.1f}s, train {hist['seconds'][-1]:.0f}s")


if __name__ == "__main__":
    main()
