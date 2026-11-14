"""Noise schedule, base process and Euler-Maruyama sampler of the controlled SDE

    dX_t = sigma(t) u(X_t, t) dt + sigma(t) A dB_t,   X_0 = 0,   t in [0, 1],

where A is the projection onto the zero centre-of-mass subspace.
"""
from __future__ import annotations

import math

import torch

from .energies import remove_mean


class GeometricSchedule:
    """Geometric (variance-exploding) schedule with decreasing noise level

        sigma(t) = sigma_min * r^(1-t) * sqrt(2 ln r),   r = sigma_max / sigma_min,
        nu_t     = int_0^t sigma(s)^2 ds = sigma_max^2 (1 - r^(-2t)),

    so nu_0 = 0 and nu_1 = sigma_max^2 - sigma_min^2.  Most of the variance is injected early;
    late times only add noise of order sigma_min (fine-scale refinement).
    """

    def __init__(self, sigma_min: float = 1e-4, sigma_max: float = 3.0):
        self.sigma_min, self.sigma_max = sigma_min, sigma_max
        self.r = sigma_max / sigma_min
        self.log_r = math.log(self.r)

    def sigma(self, t):
        t = torch.as_tensor(t)
        return self.sigma_min * torch.exp((1 - t) * self.log_r) * math.sqrt(2 * self.log_r)

    def nu(self, t):
        t = torch.as_tensor(t)
        return self.sigma_max**2 * (-torch.expm1(-2 * t * self.log_r))

    @property
    def nu1(self) -> float:
        return self.sigma_max**2 - self.sigma_min**2

    def t_of_nu(self, nu):
        """Inverse of nu(t): t = -log(1 - nu / sigma_max^2) / (2 ln r)."""
        nu = torch.as_tensor(nu)
        return -torch.log1p(-nu / self.sigma_max**2) / (2 * self.log_r)


class BaseProcess:
    """dX_t = sigma(t) A dB_t, X_0 = 0: Gaussian marginals and Brownian-bridge posterior."""

    def __init__(self, schedule: GeometricSchedule, n_particles: int, spatial_dim: int):
        self.schedule, self.n, self.d = schedule, n_particles, spatial_dim

    def project(self, x):
        return remove_mean(x, self.n, self.d)

    def sample_marginal(self, t, shape, device=None):
        """X_t ~ N(0, nu_t A)."""
        nu = self.schedule.nu(t).to(device).reshape(-1, 1)
        return self.project(torch.randn(shape, device=device)) * nu.sqrt()

    def log_p1(self, x):
        """log p_1^base(x) on X_CoM, up to an additive constant: -||x||^2 / (2 nu_1)."""
        return -(x**2).sum(-1) / (2 * self.schedule.nu1)

    def grad_log_p1(self, x):
        return -self.project(x) / self.schedule.nu1

    def sample_bridge(self, t, x1):
        """X_t ~ p^base_{t|1}(. | x1) = N( (nu_t/nu_1) x1, nu_t (nu_1 - nu_t) / nu_1 A )."""
        nu_t = self.schedule.nu(t).to(x1.device).reshape(-1, 1)
        nu_1 = self.schedule.nu1
        mean = nu_t / nu_1 * x1
        std = torch.sqrt(nu_t * (nu_1 - nu_t) / nu_1)
        return mean + std * self.project(torch.randn_like(x1))


def time_grid(schedule: GeometricSchedule, n_steps: int, grid: str = "nu", device=None):
    """Discretisation points 0 = t_0 < ... < t_K = 1, uniform in t or in the base variance nu_t."""
    if grid == "t":
        return torch.linspace(0.0, 1.0, n_steps + 1, device=device)
    nus = torch.linspace(0.0, schedule.nu1, n_steps + 1, dtype=torch.float64)
    ts = schedule.t_of_nu(nus).clamp(0, 1)
    ts[-1] = 1.0
    return ts.float().to(device)


@torch.no_grad()
def euler_maruyama(model, base: BaseProcess, n_samples: int, n_steps: int = 200, device=None,
                   return_traj: bool = False, traj_every: int = 10, grid: str = "nu"):
    """Simulate the controlled SDE from X_0 = 0 without gradients; returns X_1 (and optionally a trajectory).

    With u = sigma f, the SDE reads dX = f(X,t) dnu_t + A dW_{nu_t} in "variance time" nu_t, and one
    Euler-Maruyama step over [t_k, t_{k+1}] is  X += f(X_k, t_k) dnu_k + sqrt(dnu_k) A eps,
    dnu_k = nu_{t_{k+1}} - nu_{t_k}  (exact noise variance; reduces to sigma^2 dt for small steps).
    ``grid="nu"`` uses equal variance increments.  A uniform t-grid is unstable for the geometric
    schedule: ~99% of the variance is injected for t < 0.25, so the first steps have dnu ~ 1 and
    the drift step f dnu overshoots the stiff energy walls and diverges.
    """
    sched = base.schedule
    x = torch.zeros(n_samples, base.n * base.d, device=device)
    ts = time_grid(sched, n_steps, grid, device)
    nus = sched.nu(ts.double().cpu()).float().to(device)
    traj = [x.clone()] if return_traj else None
    for k in range(n_steps):
        t, dnu = ts[k], (nus[k + 1] - nus[k]).clamp_min(0)
        f = model.f(x, t.expand(n_samples))
        x = x + f * dnu + dnu.sqrt() * base.project(torch.randn_like(x))
        if return_traj and (k + 1) % traj_every == 0:
            traj.append(x.clone())
    return (x, torch.stack(traj)) if return_traj else x
