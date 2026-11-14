"""Reciprocal Adjoint Matching (RAM) with a replay buffer — Algorithm 1 of Havens et al. (2025)."""
from __future__ import annotations

import time

import numpy as np
import torch

from .metrics import energy_w2
from .sde import BaseProcess, euler_maruyama


class ReplayBuffer:
    """FIFO buffer of (X_1, grad g(X_1)) pairs with uniform sampling."""

    def __init__(self, max_size: int, dim: int, device=None):
        self.x = torch.empty(max_size, dim, device=device)
        self.g = torch.empty(max_size, dim, device=device)
        self.max_size, self.size, self.ptr = max_size, 0, 0

    def add(self, x, g):
        n = x.shape[0]
        idx = (self.ptr + torch.arange(n, device=x.device)) % self.max_size
        self.x[idx], self.g[idx] = x, g
        self.ptr = (self.ptr + n) % self.max_size
        self.size = min(self.size + n, self.max_size)

    def sample(self, m):
        idx = torch.randint(0, self.size, (m,), device=self.x.device)
        return self.x[idx], self.g[idx]


def terminal_cost_grad(energy, base: BaseProcess, x1, max_grad_norm: float | None = None):
    """grad g(x) = grad E(x)/tau - x/nu_1  (g = log p_1^base + E/tau), optionally norm-clipped per sample."""
    _, ge = energy.energy_and_grad(x1)
    ge = ge / energy.temperature
    n_clipped = 0
    if max_grad_norm is not None:
        norm = ge.norm(dim=-1, keepdim=True)
        n_clipped = int((norm > max_grad_norm).sum())
        ge = ge * torch.clamp(max_grad_norm / (norm + 1e-12), max=1.0)
    ge = torch.nan_to_num(ge)
    return base.project(ge) + base.grad_log_p1(x1), n_clipped  # grad log p1 = -x/nu_1


def sample_times(base: BaseProcess, m, device, t_sampling="uniform"):
    """t ~ U[0,1] (paper) or t = t(nu) with nu ~ U[0, nu_1] ("nu": uniform in base variance).

    Both have full support on [0,1], so they only re-weight the per-t regression problems and leave
    the minimiser (the fixed point u*) unchanged; "nu" puts more samples at small t, where the
    geometric schedule injects almost all of its variance."""
    u = torch.rand(m, device=device)
    if t_sampling == "uniform":
        return u
    return base.schedule.t_of_nu(u.double().cpu() * base.schedule.nu1).float().clamp(0, 1).to(device)


def ram_loss(model, base: BaseProcess, x1, grad_g, t_sampling="uniform"):
    """lambda(t)/2 ||u(X_t,t) + sigma(t) A grad g(X_1)||^2 with X_t ~ p^base_{t|1}, lambda = 1/sigma^2.

    With u = sigma * f this is 1/2 ||f(X_t, t) + A grad g(X_1)||^2 (see model.py)."""
    t = sample_times(base, x1.shape[0], x1.device, t_sampling)
    xt = base.sample_bridge(t, x1)
    res = model.f(xt, t) + base.project(grad_g)
    return 0.5 * (res**2).sum(-1).mean()


def train_adjoint_sampling(model, energy, base: BaseProcess, *, n_outer=200, n_inner=100, n_per_outer=512,
                           batch_size=512, buffer_size=10_000, lr=3e-4, n_sde_steps=200,
                           max_grad_norm: float | None = None, clip_param_grad: float | None = 1.0,
                           ref_energies=None, eval_every=10, t_sampling="uniform",
                           device="cuda", log=print, seed=0):
    torch.manual_seed(seed)
    model.to(device)
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    buf = ReplayBuffer(buffer_size, energy.dim, device)
    hist = {"outer": [], "loss": [], "energy_mean": [], "energy_w2": [], "n_energy_evals": [],
            "seconds": [], "n_clipped": []}
    n_evals, t0 = 0, time.perf_counter()
    for it in range(1, n_outer + 1):
        # ---- outer loop: simulate endpoints with the current (frozen) control, 1 energy grad each
        model.eval()
        x1 = euler_maruyama(model, base, n_per_outer, n_sde_steps, device=device)
        g, n_clip = terminal_cost_grad(energy, base, x1, max_grad_norm)
        n_evals += n_per_outer
        ok = torch.isfinite(x1).all(-1) & torch.isfinite(g).all(-1)
        buf.add(x1[ok], g[ok])
        # ---- inner loop: cheap regression steps on buffered pairs, fresh (t, X_t) every time
        model.train()
        losses = []
        for _ in range(n_inner):
            xb, gb = buf.sample(batch_size)
            loss = ram_loss(model, base, xb, gb, t_sampling)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            if clip_param_grad is not None:
                torch.nn.utils.clip_grad_norm_(model.parameters(), clip_param_grad)
            opt.step()
            losses.append(loss.item())
        with torch.no_grad():
            e = energy(x1).cpu().numpy()
        hist["outer"].append(it)
        hist["loss"].append(float(np.mean(losses)))
        hist["energy_mean"].append(float(np.median(e)))
        hist["n_energy_evals"].append(n_evals)
        hist["seconds"].append(time.perf_counter() - t0)
        hist["n_clipped"].append(n_clip)
        fin = np.isfinite(e)
        w2 = energy_w2(e[fin], ref_energies) if ref_energies is not None and fin.any() else float("nan")
        hist["energy_w2"].append(w2)
        if it % eval_every == 0 or it == 1:
            log(f"outer {it:4d} | loss {hist['loss'][-1]:9.3f} | median E {hist['energy_mean'][-1]:8.2f} | "
                f"E-W2(fresh) {w2:7.3f} | clipped {n_clip:3d} | {hist['seconds'][-1]:6.0f}s")
    return hist
