import math
import torch
import torch.nn as nn

from utils import center


class TimeEmbedding(nn.Module):
    """Fourier features of t in [0, 1]."""
    def __init__(self, dim=32, max_freq=64.0):
        super().__init__()
        self.register_buffer("freqs", math.pi * torch.logspace(0, math.log10(max_freq), dim // 2))
 
    def forward(self, t):
        a = t.reshape(-1, 1) * self.freqs
        return torch.cat([torch.sin(a), torch.cos(a)], dim=-1)

class Model(nn.Module):
    """
    Basic MLP
    Input is flattened n*d and centered
    """
    def __init__(self, n_particles, dim, schedule, hidden=256, n_layers=4, t_dim=32):
        super().__init__()
        self.n, self.d, self.schedule = n_particles, dim, schedule
        self.temb = TimeEmbedding(t_dim)

        layers = []
        sizes = [n_particles * dim + t_dim] + [hidden] * n_layers + [n_particles * dim] # DW4 (8+32, 256, 256, 256, 256, 8)
        for i in range(len(sizes) - 1):
            layers.append(nn.Linear(sizes[i], sizes[i + 1]))
            if i < len(sizes) - 2:
                layers.append(nn.SiLU())
        nn.init.zeros_(layers[-1].weight)
        nn.init.zeros_(layers[-1].bias)
        self.net = nn.Sequential(*layers)
 
    def score(self, X, t):
        B = X.shape[0]
        h = torch.cat([X.reshape(B, -1), self.temb(t.expand(B) if t.dim() == 0 else t)], -1)
        return center(self.net(h).reshape(B, self.n, self.d)) # -E[grad g(X_1) | X_t]
 
    def forward(self, X, t):
        t = torch.as_tensor(t, dtype=X.dtype, device=X.device)
        t = t.expand(X.shape[0]) if t.dim() == 0 else t
        return self.schedule.sigma(t).reshape(-1, 1, 1).to(X.dtype) * self.score(X, t)