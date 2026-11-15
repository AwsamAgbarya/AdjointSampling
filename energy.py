import torch
import math

# DW-4
DW_A = 0.0
DW_B = -4.0
DW_C = 0.9
DW_D0 = 4.0

# LJ-13 
LJ_RM = 1.0
LJ_EPS = 1.0
LJ_C = 1.0 

TEMP = 1.0

def DW_energy(x):
    "The input shape is batched (Bx4x2). Returns (B,)."
    n = x.shape[-2]
    i, j = torch.triu_indices(n, n, offset=1, device=x.device)
    d = (x[:, i] - x[:, j]).norm(dim=-1)
    return DW_pair_potential(d).sum(-1)

def DW_pair_potential(dij, a=DW_A, b=DW_B, c=DW_C, d_0=DW_D0):
    s = dij - d_0
    return a * s + b * s**2 + c * s**4

def LJ_energy(x, eps=LJ_EPS, rm=LJ_RM, c=LJ_C):
    "The input shape is batched (Bx13x3). Returns (B,)."
    n = x.shape[-2]
    i, j = torch.triu_indices(n, n, offset=1, device=x.device)
    r2 = (x[:, i] - x[:, j]).pow(2).sum(-1)
    E_lj = 2.0 * LJ_pair_potential(r2, eps, rm).sum(-1)
    x_c = x - x.mean(dim=1, keepdim=True)
    E_osc = 0.5 * x_c.pow(2).sum((-2, -1))
    return E_lj + c * E_osc

def LJ_pair_potential(dij2, eps=LJ_EPS, rm=LJ_RM):
    r3 = (rm**2 / dij2) ** 3
    return eps * (r3**2 - 2 * r3)

def energy_and_grad(energy_fn, x, **kwargs):
    "(E, dE/dx) for a batch x"
    with torch.enable_grad():
        x = x.detach().requires_grad_(True)
        E = energy_fn(x, **kwargs)
        (g,) = torch.autograd.grad(E.sum(), x)
    return E.detach(), g


class EnergyWrapper:
    """
    U is the negative unnormalised log-density
    mu(x) = exp(-U(x)) / Z with U = E / temp
    """

    def __init__(self, energy_fn, p=4, d=2, temp=TEMP):
        self.energy_fn = energy_fn
        self.dim_p, self.dim_d = p, d
        self.beta = 1.0 / temp

    def _U(self, x):
        E = self.energy_fn(x.view(-1, self.dim_p, self.dim_d)).reshape(-1)
        return torch.nan_to_num(self.beta * E, nan=math.inf)

    def U(self, x):
        with torch.no_grad():
            return self._U(x)

    def log_prob_unnorm(self, x):
        "log mu(x) + log Z = -U(x)."
        return -self.U(x)

    def U_and_grad(self, x):
        with torch.enable_grad():
            x = x.detach().requires_grad_(True)
            U = self._U(x)
            (g,) = torch.autograd.grad(U.sum(), x)
        return U.detach(), g

    def noise(self, like):
        "centered Gaussian noise (for translation invariance)"
        noise = torch.randn_like(like)
        shape_of_noise = noise.shape
        noise = noise.reshape(-1, self.dim_p, self.dim_d)
        return (noise - noise.mean(dim=1, keepdim=True)).reshape(shape_of_noise)
    