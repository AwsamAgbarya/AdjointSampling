"""Sanity tests for energies, base process, model equivariance, MCMC and metrics.  Run: pytest -q"""
import math

import numpy as np
import torch

from adjsamp.energies import DoubleWell, LennardJones
from adjsamp.mcmc import HMC, MALA, MH, BoltzmannTarget, run_chains
from adjsamp.metrics import energy_w2, geometric_w2
from adjsamp.model import Model
from adjsamp.sde import BaseProcess, GeometricSchedule

torch.set_default_dtype(torch.float64)


def _random_orthogonal(d, g):
    q, r = torch.linalg.qr(torch.randn(d, d, generator=g))
    return q * torch.sign(torch.diag(r))


def _lj_start(n=13, seed=0):
    g = torch.Generator().manual_seed(seed)
    grid = torch.stack(torch.meshgrid(*[torch.arange(3.0)] * 3, indexing="ij"), -1).reshape(-1, 3)[:n] * 1.1
    return (grid + 0.05 * torch.randn(grid.shape, generator=g)).reshape(1, -1)


def test_reference_energies():
    dw = DoubleWell()
    kite = torch.tensor([[-1.6379, 0], [-1.0725, -2.6391], [-1.0725, 2.6391], [3.7830, 0]])
    assert abs(dw(kite[None]).item() + 25.7922) < 1e-3
    assert abs(dw(torch.zeros(1, 8)).item() - 998.4) < 1e-6  # 6 * v(0)


def test_gradients_match_finite_differences():
    g = torch.Generator().manual_seed(0)
    for E, x in [(DoubleWell(), 2 * torch.randn(3, 8, generator=g)), (LennardJones(), _lj_start().repeat(3, 1))]:
        _, grad = E.energy_and_grad(x)
        v = torch.randn(x.shape, generator=g)
        h = 1e-6
        fd = (E(x + h * v) - E(x - h * v)) / (2 * h)
        assert torch.allclose(fd, (grad * v).sum(-1), rtol=1e-5, atol=1e-5)


def test_energy_invariances():
    g = torch.Generator().manual_seed(1)
    for E in [DoubleWell(), LennardJones()]:
        n, d = E.n_particles, E.spatial_dim
        x = (_lj_start() if n == 13 else 2 * torch.randn(1, 8, generator=g)).reshape(1, n, d)
        R = _random_orthogonal(d, g)
        y = (x @ R.T)[:, torch.randperm(n, generator=g)] + torch.randn(1, 1, d, generator=g)
        assert torch.allclose(E(x), E(y), atol=1e-8)


def test_schedule_derivative_and_bridge():
    s = GeometricSchedule(1e-4, 3.0)
    t, h = torch.tensor([0.05, 0.3, 0.8]), 1e-6
    assert torch.allclose((s.nu(t + h) - s.nu(t - h)) / (2 * h), s.sigma(t) ** 2, rtol=1e-4)
    assert abs(s.nu(torch.tensor(1.0)).item() - s.nu1) < 1e-8
    base = BaseProcess(s, 4, 2)
    x1 = base.project(torch.randn(1, 8)).repeat(200_000, 1)
    tt = torch.full((200_000,), 0.05)
    xt = base.sample_bridge(tt, x1)
    nu_t, nu_1 = s.nu(torch.tensor(0.05)).item(), s.nu1
    assert torch.allclose(xt.mean(0), nu_t / nu_1 * x1[0], atol=0.02)
    # per-coordinate variance on the CoM subspace is (n-1)/n of the isotropic variance
    var = nu_t * (nu_1 - nu_t) / nu_1 * 3 / 4
    assert abs(xt.var(0).mean().item() - var) / var < 0.02
    # bridge started from base endpoints reproduces the base marginal
    x1 = base.sample_marginal(torch.ones(200_000), (200_000, 8))
    xt = base.sample_bridge(tt, x1)
    assert abs(xt.var(0).mean().item() - nu_t * 3 / 4) / (nu_t * 3 / 4) < 0.02


def test_egnn_equivariance():
    g = torch.Generator().manual_seed(2)
    for n, d in [(4, 2), (13, 3)]:
        m = Model(n, d, GeometricSchedule(), arch="egnn", hidden=32, n_layers=2)
        x = torch.randn(5, n, d, generator=g)
        t = torch.rand(5, generator=g)
        R = _random_orthogonal(d, g)
        P = torch.randperm(n, generator=g)
        shift = torch.randn(1, 1, d, generator=g)
        out = m(x.reshape(5, -1), t).reshape(5, n, d)
        out2 = m(((x @ R.T)[:, P] + shift).reshape(5, -1), t).reshape(5, n, d)
        assert torch.allclose((out @ R.T)[:, P], out2, atol=1e-8)
        assert out.sum(1).abs().max() < 1e-10  # zero centre of mass


def test_mcmc_stays_on_com_subspace():
    E = DoubleWell()
    for S in (MH(0.3), MALA(1e-3), HMC(0.05, 5)):
        tgt = BoltzmannTarget(E)
        res = run_chains(S, tgt, 2 * torch.randn(16, 8), n_warmup=50, n_steps=100, thin=10)
        com = res["samples"].reshape(-1, 16, 4, 2).mean(2)
        assert np.abs(com).max() < 1e-4


def test_metrics():
    rng = np.random.default_rng(0)
    e = rng.normal(size=5000)
    assert energy_w2(e, e + 1.0) - 1.0 < 1e-6
    x = rng.normal(size=(200, 4, 2))
    x -= x.mean(1, keepdims=True)
    th = rng.uniform(0, 2 * math.pi, size=200)
    R = np.stack([np.stack([np.cos(th), -np.sin(th)], -1), np.stack([np.sin(th), np.cos(th)], -1)], -2)
    y = np.einsum("bij,bnj->bni", R, x)[:, rng.permutation(4)]
    assert geometric_w2(x.reshape(200, -1), y.reshape(200, -1), 4, 2, device="cpu") < 1e-6
