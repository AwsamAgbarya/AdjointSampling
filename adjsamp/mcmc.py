"""MCMC samplers (MH, MALA, HMC) for Boltzmann targets on the zero centre-of-mass subspace.

All energies here are translation invariant, so exp(-E/tau) is not normalisable on R^{nd}.
We therefore run every chain on X_CoM = {x : sum_i x_i = 0}: chains are initialised with zero
centre of mass and *all proposal noise is projected* onto X_CoM.  Gradients of a translation
invariant energy already have zero centre of mass, so MALA/HMC moves stay on X_CoM too.
"""
from __future__ import annotations

import math
import time

import numpy as np
import torch

from .energies import ParticleEnergy


class BoltzmannTarget:
    """mu(x) ∝ exp(-E(x)/tau) restricted to the zero centre-of-mass subspace."""

    def __init__(self, energy: ParticleEnergy):
        self.energy = energy
        self.n_energy_evals = 0  # counts single-configuration energy (or energy+grad) evaluations

    @property
    def dim(self):
        return self.energy.dim

    def U(self, x):
        with torch.no_grad():
            self.n_energy_evals += x.shape[0]
            return torch.nan_to_num(self.energy.reduced_energy(x), nan=math.inf)

    def U_and_grad(self, x):
        self.n_energy_evals += x.shape[0]
        e, g = self.energy.energy_and_grad(x)
        t = self.energy.temperature
        return torch.nan_to_num(e / t, nan=math.inf), torch.nan_to_num(g / t)

    def noise(self, like):
        return self.energy.remove_mean(torch.randn_like(like))


def _accept(state, proposal, log_alpha):
    log_alpha = torch.nan_to_num(log_alpha, nan=-math.inf)
    accepted = torch.log(torch.rand_like(log_alpha)) < log_alpha
    new = {}
    for k, v in state.items():
        mask = accepted.view(-1, *([1] * (v.dim() - 1)))
        new[k] = torch.where(mask, proposal[k], v)
    return new, log_alpha.clamp(max=0.0).exp(), accepted


class MH:
    """Random-walk Metropolis-Hastings: y = x + eps * xi, xi ~ N(0, A), A = CoM projection."""

    target_accept = 0.234
    grads_per_step = 0

    def __init__(self, step_size=0.3):
        self.step_size = step_size

    def init_state(self, target, x):
        return {"x": x, "U": target.U(x)}

    def step(self, target, s):
        y = s["x"] + self.step_size * target.noise(s["x"])
        Uy = target.U(y)
        return _accept(s, {"x": y, "U": Uy}, s["U"] - Uy)


class MALA:
    """Metropolis-adjusted Langevin: y = x - eps * grad U(x) + sqrt(2 eps) xi."""

    target_accept = 0.574
    grads_per_step = 1

    def __init__(self, step_size=1e-3):
        self.step_size = step_size

    def init_state(self, target, x):
        U, g = target.U_and_grad(x)
        return {"x": x, "U": U, "g": g}

    def _log_q(self, to, frm, g_frm):  # log N(to; frm - eps g, 2 eps I), up to a constant
        return -((to - frm + self.step_size * g_frm) ** 2).sum(-1) / (4 * self.step_size)

    def step(self, target, s):
        x, U, g = s["x"], s["U"], s["g"]
        y = x - self.step_size * g + math.sqrt(2 * self.step_size) * target.noise(x)
        Uy, gy = target.U_and_grad(y)
        log_alpha = U - Uy + self._log_q(x, y, gy) - self._log_q(y, x, g)
        return _accept(s, {"x": y, "U": Uy, "g": gy}, log_alpha)


class HMC:
    """Hamiltonian Monte Carlo with leapfrog integration and a jittered step size."""

    target_accept = 0.75

    def __init__(self, step_size=0.05, n_leapfrog=10, jitter=0.2):
        self.step_size = step_size
        self.n_leapfrog = n_leapfrog
        self.jitter = jitter

    @property
    def grads_per_step(self):
        return self.n_leapfrog

    def init_state(self, target, x):
        U, g = target.U_and_grad(x)
        return {"x": x, "U": U, "g": g}

    def step(self, target, s):
        x, U, g = s["x"], s["U"], s["g"]
        u = torch.rand(x.shape[0], 1, dtype=x.dtype, device=x.device)
        eps = self.step_size * (1 + self.jitter * (2 * u - 1))
        p = target.noise(x)  # momentum on the CoM subspace
        H0 = U + 0.5 * (p**2).sum(-1)
        y, gy, Uy = x, g, U
        p = p - 0.5 * eps * gy
        for l in range(self.n_leapfrog):
            y = y + eps * p
            Uy, gy = target.U_and_grad(y)
            if l < self.n_leapfrog - 1:
                p = p - eps * gy
        p = p - 0.5 * eps * gy
        H1 = Uy + 0.5 * (p**2).sum(-1)
        return _accept(s, {"x": y, "U": Uy, "g": gy}, H0 - H1)


SAMPLERS = {"mh": MH, "mala": MALA, "hmc": HMC}


def run_chains(sampler, target: BoltzmannTarget, x0: torch.Tensor, n_warmup: int, n_steps: int,
               thin: int = 10, adapt: bool = True, log_every: int = 0, adapt_rate: float = 0.05,
               trace_steps: int | None = None):
    """Run ``B = x0.shape[0]`` independent chains.

    Warm-up: the first 75% of warm-up steps adapt a single global log step size towards the
    sampler's target acceptance (Robbins-Monro); the final step size is the average of the
    second half of the adaptation phase.  Afterwards the kernel is fixed (valid MCMC).

    Returns a dict with ``samples`` (n_steps//thin, B, dim) and their ``energies``
    (n_steps//thin, B), a per-step energy ``trace`` of the first ``trace_steps`` sampling steps
    (for autocorrelation analysis), acceptance rates, timings and the number of energy evaluations.
    """
    trace_steps = n_steps if trace_steps is None else min(trace_steps, n_steps)
    x = target.energy.remove_mean(x0.clone())
    B = x.shape[0]
    state = sampler.init_state(target, x)
    warm_trace = np.empty((n_warmup, B), dtype=np.float32)
    n_adapt = int(0.75 * n_warmup) if adapt else 0
    log_eps, history = math.log(sampler.step_size), []
    t0 = time.perf_counter()
    for t in range(n_warmup):
        state, acc_prob, _ = sampler.step(target, state)
        warm_trace[t] = state["U"].float().cpu().numpy() * target.energy.temperature
        if t < n_adapt:
            log_eps += adapt_rate * (acc_prob.mean().item() - sampler.target_accept)
            sampler.step_size = math.exp(log_eps)
            if t >= n_adapt // 2:
                history.append(log_eps)
            if t == n_adapt - 1:
                sampler.step_size = math.exp(float(np.mean(history)))
    t_warm = time.perf_counter() - t0
    evals_warm = target.n_energy_evals

    n_keep = n_steps // thin
    samples = np.empty((n_keep, B, target.dim), dtype=np.float32)
    energies = np.empty((n_keep, B), dtype=np.float32)
    trace = np.empty((trace_steps, B), dtype=np.float32)
    n_acc = torch.zeros(B, device=x.device)
    t0 = time.perf_counter()
    for t in range(1, n_steps + 1):
        state, _, accepted = sampler.step(target, state)
        n_acc += accepted
        if t <= trace_steps:
            trace[t - 1] = (state["U"] * target.energy.temperature).float().cpu().numpy()
        if t % thin == 0:
            samples[t // thin - 1] = state["x"].float().cpu().numpy()
            energies[t // thin - 1] = (state["U"] * target.energy.temperature).float().cpu().numpy()
        if log_every and t % log_every == 0:
            print(f"  {type(sampler).__name__}: step {t}/{n_steps}, accept {n_acc.mean().item() / t:.3f}, "
                  f"E={(state['U'] * target.energy.temperature).mean().item():.2f}, "
                  f"{time.perf_counter() - t0:.0f}s", flush=True)
    return {
        "method": type(sampler).__name__,
        "samples": samples,
        "energies": energies,
        "trace": trace,
        "warmup_energies": warm_trace,
        "accept_rate": (n_acc / n_steps).cpu().numpy(),
        "step_size": sampler.step_size,
        "thin": thin,
        "seconds_warmup": t_warm,
        "seconds": time.perf_counter() - t0,
        "energy_evals_warmup": evals_warm,
        "energy_evals": target.n_energy_evals - evals_warm,
        "x_final": state["x"].float().cpu().numpy(),
    }


# ---------------------------------------------------------------------------------------------
# Diagnostics
# ---------------------------------------------------------------------------------------------
def autocorrelation(trace: np.ndarray, max_lag: int | None = None) -> np.ndarray:
    """Normalised autocorrelation of each chain (columns), averaged over chains. trace: (T, B)."""
    T = trace.shape[0]
    max_lag = max_lag or T // 2
    x = trace - trace.mean(0, keepdims=True)
    n = 1 << (2 * T - 1).bit_length()
    f = np.fft.rfft(x, n=n, axis=0)
    acf = np.fft.irfft(f * np.conj(f), n=n, axis=0)[:max_lag]
    acf /= np.maximum(acf[:1], 1e-12)
    return acf.mean(1)


def integrated_autocorr_time(trace: np.ndarray, c: float = 5.0) -> float:
    """Sokal's automatic-window estimate of the integrated autocorrelation time (in steps)."""
    rho = autocorrelation(trace)
    taus = 2 * np.cumsum(rho) - 1
    for m in range(1, len(taus)):
        if m >= c * taus[m]:
            return float(taus[m])
    return float(taus[-1])


def split_rhat(trace: np.ndarray) -> float:
    """Split-R̂ (Gelman et al.) of a scalar trace of shape (T, B)."""
    T = trace.shape[0] // 2
    chains = np.concatenate([trace[:T], trace[T:2 * T]], axis=1)
    W = chains.var(0, ddof=1).mean()
    Bv = T * chains.mean(0).var(ddof=1)
    var_hat = (T - 1) / T * W + Bv / T
    return float(np.sqrt(var_hat / W))
