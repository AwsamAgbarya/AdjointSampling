import math

import torch

def _accept(state, proposal, log_alpha):
    """Per-chain accept/reject. NaN log-ratios count as rejections."""
    log_alpha = torch.nan_to_num(log_alpha, nan=-math.inf)
    accepted = torch.log(torch.rand_like(log_alpha)) < log_alpha
    new = {}
    for k, v in state.items():
        mask = accepted.view(-1, *([1] * (v.dim() - 1)))
        new[k] = torch.where(mask, proposal[k], v)
    return new, log_alpha.clamp(max=0.0).exp(), accepted


class MH:
    """Random-walk Metropolis-Hastings.
    y = x + step_size * xi,  xi ~ N(0, I) projected to zero CoM (symmetric proposal)
    alpha = min(1, exp(U(x) - U(y)))
    """
    target_accept = 0.30

    def __init__(self, step_size=0.3):
        self.step_size = step_size

    def init_state(self, target, x):
        return {"x": x, "U": target.U(x)}

    def step(self, target, s):
        y = s["x"] + self.step_size * target.noise(s["x"])
        Uy = target.U(y)
        return _accept(s, {"x": y, "U": Uy}, s["U"] - Uy)

    def describe(self):
        return {"step_size": self.step_size}


class HMC:
    """Hamiltonian Monte Carlo, H(x, p) = U(x) + |p|^2 / 2, leapfrog integrator.
    step size eps * Uniform(1 - jitter, 1 + jitter).
    """
    target_accept = 0.75

    def __init__(self, step_size=0.05, n_leapfrog=10, jitter=0.2, traj_len=None, max_leapfrog=200):
        self.step_size = step_size
        self._n_leapfrog = n_leapfrog
        self.jitter = jitter
        self.traj_len = traj_len
        self.max_leapfrog = max_leapfrog

    @property
    def n_leapfrog(self):
        if self.traj_len is None:
            return self._n_leapfrog
        return int(min(max(math.ceil(self.traj_len / self.step_size), 1), self.max_leapfrog))

    def init_state(self, target, x):
        U, g = target.U_and_grad(x)
        return {"x": x, "U": U, "g": g}

    def step(self, target, s):
        x, U, g = s["x"], s["U"], s["g"]
        L = self.n_leapfrog
        u = torch.rand(x.shape[0], 1, dtype=x.dtype, device=x.device)
        eps = self.step_size * (1 + self.jitter * (2 * u - 1))
        p = target.noise(x)
        H0 = U + 0.5 * (p**2).sum(-1)
        y, gy = x, g
        p = p - 0.5 * eps * gy
        for l in range(L):
            y = y + eps * p
            Uy, gy = target.U_and_grad(y)
            if l < L - 1:
                p = p - eps * gy
        p = p - 0.5 * eps * gy
        H1 = Uy + 0.5 * (p**2).sum(-1)
        return _accept(s, {"x": y, "U": Uy, "g": gy}, H0 - H1)

    def describe(self):
        return {"step_size": self.step_size, "n_leapfrog": self.n_leapfrog, "traj_len": self.traj_len, "jitter": self.jitter}