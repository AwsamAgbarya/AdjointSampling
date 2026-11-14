"""Energy functions for the synthetic particle systems DW-4 and LJ-13.

Both energies follow the conventions of the iDEM code base (Akhound-Sadegh et al., 2024),
which Adjoint Sampling reuses, so that our reference samples are comparable with the
published datasets:

* DW-4  (bgflow ``MultiDoubleWellPotential(a=0.9, b=-4, c=0, offset=4)``):
      E(x) = sum_{i<j} 0.9 (d_ij - 4)^4 - 4 (d_ij - 4)^2           (6 unique pairs)
* LJ-13 (iDEM ``LennardJonesPotential(eps=1, rm=1, oscillator_scale=1)``):
      E(x) = sum_{i!=j} [ (1/d_ij)^12 - 2 (1/d_ij)^6 ]  +  1/2 sum_i ||x_i - x_com||^2
  The pair sum runs over *ordered* pairs (156 terms), i.e. twice the textbook LJ energy,
  and distances are regularised as sqrt(||x_i - x_j||^2 + 1e-6) exactly as in bgflow.

All energies accept flat ``(B, n*d)`` or structured ``(B, n, d)`` tensors and return ``(B,)``.
"""
from __future__ import annotations

import math

import torch


def remove_mean(x: torch.Tensor, n_particles: int, spatial_dim: int) -> torch.Tensor:
    """Project a (flat or structured) configuration onto the zero centre-of-mass subspace."""
    shape = x.shape
    p = x.reshape(-1, n_particles, spatial_dim)
    return (p - p.mean(dim=1, keepdim=True)).reshape(shape)


class ParticleEnergy:
    """Base class: translation-invariant pair energy of ``n_particles`` points in ``spatial_dim`` D."""

    n_particles: int
    spatial_dim: int
    name: str

    def __init__(self, temperature: float = 1.0):
        self.temperature = temperature
        i, j = torch.triu_indices(self.n_particles, self.n_particles, offset=1)
        self._i, self._j = i, j

    @property
    def dim(self) -> int:
        return self.n_particles * self.spatial_dim

    # -- helpers ---------------------------------------------------------------------------
    def structured(self, x: torch.Tensor) -> torch.Tensor:
        return x.reshape(-1, self.n_particles, self.spatial_dim)

    def pair_distances(self, x: torch.Tensor, eps: float = 0.0) -> torch.Tensor:
        """Distances of the n(n-1)/2 unique pairs, shape (B, n_pairs)."""
        p = self.structured(x)
        i, j = self._i.to(p.device), self._j.to(p.device)
        diff = p[:, i] - p[:, j]
        return torch.sqrt((diff**2).sum(-1) + eps)

    def remove_mean(self, x: torch.Tensor) -> torch.Tensor:
        return remove_mean(x, self.n_particles, self.spatial_dim)

    # -- interface -------------------------------------------------------------------------
    def energy(self, x: torch.Tensor) -> torch.Tensor:  # pragma: no cover - abstract
        raise NotImplementedError

    def pair_potential(self, r):  # pragma: no cover - abstract
        raise NotImplementedError

    def __call__(self, x: torch.Tensor) -> torch.Tensor:
        return self.energy(x)

    def reduced_energy(self, x: torch.Tensor) -> torch.Tensor:
        """U(x) = E(x) / tau, the negative unnormalised log-density of the Boltzmann target."""
        return self.energy(x) / self.temperature

    def energy_and_grad(self, x: torch.Tensor):
        """Return E(x) of shape (B,) and dE/dx with the same shape as x (via autograd)."""
        with torch.enable_grad():
            x = x.detach().requires_grad_(True)
            e = self.energy(x)
            (g,) = torch.autograd.grad(e.sum(), x)
        return e.detach(), g.detach()


class DoubleWell(ParticleEnergy):
    """DW-4: 4 particles in 2D, d = 8."""

    name = "DW-4"

    def __init__(self, n_particles: int = 4, spatial_dim: int = 2, a: float = 0.0, b: float = -4.0,
                 c: float = 0.9, d0: float = 4.0, temperature: float = 1.0):
        self.n_particles, self.spatial_dim = n_particles, spatial_dim
        self.a, self.b, self.c, self.d0 = a, b, c, d0
        super().__init__(temperature)

    def pair_potential(self, r):
        s = r - self.d0
        return self.a * s + self.b * s**2 + self.c * s**4

    @property
    def well_radii(self):
        """Distances of the two pair-potential minima (for a = 0): d0 -/+ sqrt(-b / 2c)."""
        w = math.sqrt(-self.b / (2 * self.c))
        return self.d0 - w, self.d0 + w

    def energy(self, x: torch.Tensor) -> torch.Tensor:
        return self.pair_potential(self.pair_distances(x)).sum(-1)


class LennardJones(ParticleEnergy):
    """LJ-n with harmonic centre-of-mass confinement (LJ-13: 13 particles in 3D, d = 39)."""

    name = "LJ-13"

    def __init__(self, n_particles: int = 13, spatial_dim: int = 3, eps: float = 1.0, rm: float = 1.0,
                 oscillator_scale: float = 1.0, pair_factor: float = 2.0, dist_eps: float = 1e-6,
                 temperature: float = 1.0):
        self.n_particles, self.spatial_dim = n_particles, spatial_dim
        self.eps, self.rm = eps, rm
        self.oscillator_scale = oscillator_scale
        self.pair_factor = pair_factor  # 2.0 = iDEM convention (sum over ordered pairs)
        self.dist_eps = dist_eps
        self.name = f"LJ-{n_particles}"
        super().__init__(temperature)

    def pair_potential(self, r):
        """Single LJ pair term eps[(rm/r)^12 - 2(rm/r)^6] (minimum -eps at r = rm)."""
        q = (self.rm / r) ** 6
        return self.eps * (q**2 - 2 * q)

    def oscillator(self, x: torch.Tensor) -> torch.Tensor:
        p = self.structured(x)
        return 0.5 * ((p - p.mean(1, keepdim=True)) ** 2).sum((-2, -1))

    def energy(self, x: torch.Tensor) -> torch.Tensor:
        d = self.pair_distances(x, eps=self.dist_eps)
        e_lj = self.pair_factor * self.pair_potential(d).sum(-1)
        return e_lj + self.oscillator_scale * self.oscillator(x)


def get_energy(name: str) -> ParticleEnergy:
    name = name.lower().replace("-", "")
    if name == "dw4":
        return DoubleWell()
    if name == "lj13":
        return LennardJones()
    raise ValueError(f"unknown system {name!r}")
