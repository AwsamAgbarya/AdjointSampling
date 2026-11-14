"""Generate MCMC reference samples.

Example:
    python scripts/run_mcmc.py --system dw4 --method hmc --chains 1024 --warmup 5000 --steps 50000 --thin 10
"""
import argparse
import os
import sys

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from adjsamp.data import save_reference  # noqa: E402
from adjsamp.energies import get_energy  # noqa: E402
from adjsamp.mcmc import SAMPLERS, BoltzmannTarget, integrated_autocorr_time, run_chains, split_rhat  # noqa: E402


def initial_state(system, n_chains, gen, device):
    if system == "dw4":
        # broad Gaussian: particles spread over several pair-potential widths
        return 2.5 * torch.randn(n_chains, 8, generator=gen).to(device)
    # LJ-13: 13 random sites of a 3x3x3 lattice (spacing 1.1 ~ r_m) + jitter -> no overlapping atoms,
    # different random sub-lattice per chain so the chains start from distinct configurations
    grid = torch.stack(torch.meshgrid(*[torch.arange(3.0)] * 3, indexing="ij"), -1).reshape(-1, 3) * 1.1
    idx = torch.stack([torch.randperm(27, generator=gen)[:13] for _ in range(n_chains)])
    x = grid[idx] + 0.05 * torch.randn(n_chains, 13, 3, generator=gen)
    return x.reshape(n_chains, -1).to(device)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--system", default="dw4", choices=["dw4", "lj13"])
    ap.add_argument("--method", default="hmc", choices=list(SAMPLERS))
    ap.add_argument("--chains", type=int, default=1024)
    ap.add_argument("--warmup", type=int, default=5000)
    ap.add_argument("--steps", type=int, default=50000)
    ap.add_argument("--thin", type=int, default=10)
    ap.add_argument("--trace-steps", type=int, default=20000)
    ap.add_argument("--step-size", type=float, default=None)
    ap.add_argument("--n-leapfrog", type=int, default=10)
    ap.add_argument("--pre-mh", type=int, default=0, help="MH pre-equilibration steps before warm-up")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--out", default=None)
    a = ap.parse_args()

    torch.manual_seed(a.seed)
    gen = torch.Generator().manual_seed(a.seed)
    energy = get_energy(a.system)
    defaults = {"mh": 0.1, "mala": 1e-3, "hmc": 0.02}
    step = a.step_size or defaults[a.method]
    sampler = SAMPLERS[a.method](step, a.n_leapfrog) if a.method == "hmc" else SAMPLERS[a.method](step)
    target = BoltzmannTarget(energy)
    x0 = initial_state(a.system, a.chains, gen, a.device)
    pre = None
    if a.pre_mh > 0:
        # Gradient-based kernels are unstable far from equilibrium (stiff quartic / r^-12 walls give
        # huge gradients, every leapfrog trajectory overshoots and is rejected), so chains are first
        # relaxed with an adaptive random-walk MH run. These steps are discarded.
        pre_target = BoltzmannTarget(energy)
        pre = run_chains(SAMPLERS["mh"](0.05), pre_target, x0, a.pre_mh, a.thin, a.thin, trace_steps=1)
        x0 = torch.as_tensor(pre["x_final"], device=a.device)
        print(f"pre-equilibration: {a.pre_mh} MH steps, final mean E {pre['energies'][-1].mean():.2f}")
    print(f"{energy.name} / {a.method}: {a.chains} chains, {a.warmup} warm-up + {a.steps} steps, thin {a.thin}")
    res = run_chains(sampler, target, x0, a.warmup, a.steps, a.thin, log_every=max(a.steps // 10, 1),
                     trace_steps=a.trace_steps)
    tau = integrated_autocorr_time(res["trace"])
    rhat = split_rhat(res["energies"])
    n_grad = getattr(sampler, "grads_per_step", 0)
    summary = {
        "system": a.system, "chains": a.chains, "warmup": a.warmup, "steps": a.steps, "seed": a.seed,
        "n_leapfrog": a.n_leapfrog if a.method == "hmc" else None,
        "tau_int_energy_steps": tau, "rhat_energy": rhat,
        "ess_total": a.chains * a.steps / tau,
        "energy_evals_per_ess": res["energy_evals"] / (a.chains * a.steps / tau),
        "energy_mean": float(res["energies"].mean()), "energy_std": float(res["energies"].std()),
        "grads_per_step": n_grad, "pre_mh_steps": a.pre_mh,
        "pre_mh_energy_evals": pre["energy_evals_warmup"] + pre["energy_evals"] if pre else 0,
    }
    print({k: (round(v, 4) if isinstance(v, float) else v) for k, v in summary.items()})
    print(f"step size {res['step_size']:.4g}, accept {res['accept_rate'].mean():.3f}, "
          f"{res['seconds'] + res['seconds_warmup']:.0f}s")
    out = a.out or os.path.join("data", "reference", f"{a.system}_{a.method}")
    save_reference(out, res, summary)
    print("saved to", out)


if __name__ == "__main__":
    main()
