"""Drift networks u_theta(X_t, t): an MLP baseline and an E(n)-equivariant GNN (Satorras et al., 2021).

Parameterisation.  The RAM regression target is -sigma(t) grad g(X_1) with lambda(t) = 1/sigma(t)^2,
so lambda ||u + sigma grad g||^2 = ||u/sigma + grad g||^2.  We therefore let the network predict
f_theta ≈ -grad g and return u_theta = sigma(t) * f_theta; the network output then has the same
scale (that of grad g) at every t, although sigma(t) spans four orders of magnitude.
"""
from __future__ import annotations

import math

import torch
import torch.nn as nn

from .energies import remove_mean


class TimeEmbedding(nn.Module):
    def __init__(self, dim: int = 32):
        super().__init__()
        self.register_buffer("freqs", torch.exp(torch.linspace(0, math.log(1000.0), dim // 2)))

    def forward(self, t):
        a = t.reshape(-1, 1) * self.freqs
        return torch.cat([torch.sin(a), torch.cos(a)], -1)


class MLPNet(nn.Module):
    """Plain MLP on the flattened coordinates: neither permutation- nor rotation-equivariant."""

    def __init__(self, dim: int, hidden: int = 256, n_layers: int = 4, t_dim: int = 32):
        super().__init__()
        self.temb = TimeEmbedding(t_dim)
        layers, d_in = [], dim + t_dim
        for _ in range(n_layers):
            layers += [nn.Linear(d_in, hidden), nn.SiLU()]
            d_in = hidden
        layers.append(nn.Linear(hidden, dim))
        self.net = nn.Sequential(*layers)

    def forward(self, x, t):
        return self.net(torch.cat([x, self.temb(t)], -1))


class EGNNLayer(nn.Module):
    def __init__(self, hidden: int):
        super().__init__()
        self.phi_e = nn.Sequential(nn.Linear(2 * hidden + 1, hidden), nn.SiLU(), nn.Linear(hidden, hidden), nn.SiLU())
        self.phi_x = nn.Sequential(nn.Linear(hidden, hidden), nn.SiLU(), nn.Linear(hidden, 1))
        self.phi_h = nn.Sequential(nn.Linear(2 * hidden, hidden), nn.SiLU(), nn.Linear(hidden, hidden))

    def forward(self, h, x):
        # h: (B, n, H), x: (B, n, d)
        n = x.shape[1]
        diff = x[:, :, None] - x[:, None]                       # (B, n, n, d)  x_i - x_j
        d2 = (diff**2).sum(-1, keepdim=True)                   # (B, n, n, 1)
        hi = h[:, :, None].expand(-1, -1, n, -1)
        hj = h[:, None].expand(-1, n, -1, -1)
        m = self.phi_e(torch.cat([hi, hj, d2], -1))            # (B, n, n, H)
        mask = 1 - torch.eye(n, device=x.device)[None, :, :, None]
        m = m * mask
        coef = self.phi_x(m) * mask
        # eps inside the sqrt: d2 = 0 on the diagonal and for all pairs at X_0 = 0, where the
        # derivative of sqrt is infinite (inf * 0 = NaN in the backward pass)
        x = x + (diff / (torch.sqrt(d2 + 1e-8) + 1.0) * coef).sum(2) / (n - 1)
        h = h + self.phi_h(torch.cat([h, m.sum(2)], -1))
        return h, x


class EGNNNet(nn.Module):
    """E(n)-equivariant drift: output transforms like the input under rotations, reflections and
    permutations, and is invariant to translations (it is a difference of coordinates)."""

    def __init__(self, n_particles: int, spatial_dim: int, hidden: int = 128, n_layers: int = 3, t_dim: int = 32):
        super().__init__()
        self.n, self.d = n_particles, spatial_dim
        self.temb = TimeEmbedding(t_dim)
        self.embed = nn.Linear(t_dim, hidden)
        self.layers = nn.ModuleList([EGNNLayer(hidden) for _ in range(n_layers)])

    def forward(self, x, t):
        B = x.shape[0]
        x0 = x.reshape(B, self.n, self.d)
        h = self.embed(self.temb(t))[:, None].expand(-1, self.n, -1)
        xh = x0
        for layer in self.layers:
            h, xh = layer(h, xh)
        return (xh - x0).reshape(B, -1)


class Model(nn.Module):
    """u_theta(X_t, t) = sigma(t) * A f_theta(X_t, t), with A the centre-of-mass projection."""

    def __init__(self, n_particles: int, spatial_dim: int, schedule, arch: str = "egnn",
                 hidden: int = 128, n_layers: int = 3):
        super().__init__()
        self.n, self.d, self.schedule = n_particles, spatial_dim, schedule
        if arch == "egnn":
            self.net = EGNNNet(n_particles, spatial_dim, hidden, n_layers)
        elif arch == "mlp":
            self.net = MLPNet(n_particles * spatial_dim, hidden, n_layers)
        else:
            raise ValueError(arch)

    def f(self, x, t):
        """Scaled output f_theta ≈ -grad g(X_1) (centre-of-mass free)."""
        return remove_mean(self.net(x, t), self.n, self.d)

    def forward(self, X, t):
        sig = self.schedule.sigma(t).to(X.device).reshape(-1, 1)
        return sig * self.f(X, t)
