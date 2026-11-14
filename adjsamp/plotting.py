"""Report figures: configuration grids (2D/3D), sample statistics with overlays, diagnostics.

The palette is shared with ``visualizations.py`` so that all figures of the report look alike.
Every plotting function draws into given axes (``ax``/``axes``) and returns them, so panels can be
combined freely and further layers (model samples, trajectories, ...) can be added later.
"""
from __future__ import annotations

import itertools
import os
import sys
from typing import Callable, Sequence

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from visualizations import AXIS, ENERGY_CMAP, GRID, INK, INK2, MUTED, SERIES, SURFACE  # noqa: E402

REF_COLOR = "#9a9993"          # reference (ground truth) histograms: neutral grey fill
TEXTWIDTH_IN = 6.3             # LaTeX article, 2.5 cm margins


def set_style():
    mpl.rcParams.update({
        "font.size": 8, "axes.titlesize": 8.5, "axes.labelsize": 8, "legend.fontsize": 7,
        "xtick.labelsize": 7, "ytick.labelsize": 7, "axes.edgecolor": AXIS, "axes.labelcolor": INK2,
        "xtick.color": MUTED, "ytick.color": MUTED, "axes.grid": True, "grid.color": GRID,
        "grid.linewidth": 0.5, "axes.spines.top": False, "axes.spines.right": False,
        "figure.dpi": 150, "savefig.bbox": "tight", "pdf.fonttype": 42, "axes.titlecolor": INK,
        "legend.frameon": False, "lines.linewidth": 1.3,
    })


def save(fig, path):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    fig.savefig(path)
    plt.close(fig)


# ---------------------------------------------------------------------------------------------
# Configurations
# ---------------------------------------------------------------------------------------------
def _bonds(p, cutoff):
    for i, j in itertools.combinations(range(len(p)), 2):
        r = np.linalg.norm(p[i] - p[j])
        if r < cutoff:
            yield i, j, r


def draw_configuration(ax, x, n_particles, spatial_dim, bond_cutoff, title=None, lim=None,
                       color=INK, alpha=1.0, label_particles=True):
    """Draw one configuration (centred). 2D: plain axes; 3D: ``ax`` must be a 3D axes.
    Bonds connect particles closer than ``bond_cutoff`` (e.g. the DW barrier d0 or 1.3 r_m for LJ)."""
    p = np.asarray(x, dtype=float).reshape(n_particles, spatial_dim)
    p = p - p.mean(0)
    for i, j, r in _bonds(p, bond_cutoff):
        ax.plot(*p[[i, j]].T, color=SERIES[0], lw=1.2, alpha=0.7 * alpha, zorder=1)
    if spatial_dim == 2:
        ax.scatter(*p.T, s=70, color=color, edgecolor=SURFACE, linewidth=1.5, zorder=3, alpha=alpha)
        if label_particles:
            for k, (px, py) in enumerate(p):
                ax.annotate(str(k), (px, py), ha="center", va="center", color="white", fontsize=6,
                            fontweight="bold", zorder=4)
        lim = lim or np.abs(p).max() + 1.0
        ax.set(xlim=(-lim, lim), ylim=(-lim, lim), aspect="equal", xticks=[], yticks=[])
        ax.grid(False)
        for s in ax.spines.values():
            s.set_visible(False)
    else:
        z = p[:, 2]
        shade = (z - z.min()) / (np.ptp(z) + 1e-9)  # depth cue: far = light, near = dark
        cols = ENERGY_CMAP(0.15 + 0.7 * (1 - shade))
        ax.scatter(*p.T, s=110, c=cols, edgecolor=SURFACE, linewidth=0.8, depthshade=False, alpha=alpha)
        lim = lim or np.abs(p).max() + 0.3
        ax.set(xlim=(-lim, lim), ylim=(-lim, lim), zlim=(-lim, lim))
        ax.set_box_aspect((1, 1, 1))
        ax.set_axis_off()
    if title:
        ax.set_title(title, fontsize=7.5, pad=2)
    return ax


def plot_configuration_grid(X, n_particles, spatial_dim, energies=None, ncols=5, bond_cutoff=None,
                            titles=None, panel_size=1.25, lim=None, overlays: Sequence[Callable] = ()):
    """Grid of configurations. ``overlays``: callables ``f(ax, i, x_i)`` drawn on top of panel i
    (e.g. a model sample over a reference sample, or a sampling trajectory)."""
    X = np.asarray(X)
    n = len(X)
    nrows = int(np.ceil(n / ncols))
    kw = {"projection": "3d"} if spatial_dim == 3 else {}
    fig = plt.figure(figsize=(panel_size * ncols, panel_size * nrows * (1.05 if spatial_dim == 3 else 1.0)))
    axes = []
    for i in range(n):
        ax = fig.add_subplot(nrows, ncols, i + 1, **kw)
        t = titles[i] if titles is not None else (f"E = {energies[i]:.2f}" if energies is not None else None)
        draw_configuration(ax, X[i], n_particles, spatial_dim, bond_cutoff or np.inf, title=t, lim=lim)
        for f in overlays:
            f(ax, i, X[i])
        axes.append(ax)
    return fig, axes


# ---------------------------------------------------------------------------------------------
# Sample statistics with overlays
# ---------------------------------------------------------------------------------------------
def pair_distances(x, n, d):
    p = np.asarray(x).reshape(-1, n, d)
    i, j = np.triu_indices(n, 1)
    return np.linalg.norm(p[:, i] - p[:, j], axis=-1).ravel()


def plot_sample_stats(sets: dict, n_particles, spatial_dim, axes=None, e_range=None, r_range=None, bins=80,
                      reference: str | None = None, density=True):
    """Energy histogram and pair-distance histogram for several sample sets.

    ``sets`` maps a label to ``{"x": (N, dim) array, "e": (N,) array}``.  The set named
    ``reference`` is drawn as a grey filled histogram, all others as coloured step lines, on
    identical bins so the panels can be compared directly."""
    if axes is None:
        _, axes = plt.subplots(1, 2, figsize=(TEXTWIDTH_IN, 2.0), layout="constrained")
    all_e = np.concatenate([np.asarray(s["e"]) for s in sets.values()])
    e_range = e_range or tuple(np.quantile(all_e, [0.001, 0.995]))
    e_bins = np.linspace(*e_range, bins)
    r_all = {k: pair_distances(s["x"], n_particles, spatial_dim) for k, s in sets.items()}
    r_range = r_range or (0, np.quantile(np.concatenate(list(r_all.values())), 0.999))
    r_bins = np.linspace(*r_range, bins)
    ci = 0
    for label, s in sets.items():
        if label == reference:
            style = dict(histtype="stepfilled", color=REF_COLOR, alpha=0.45, lw=0)
        else:
            style = dict(histtype="step", color=SERIES[ci % len(SERIES)], lw=1.3)
            ci += 1
        axes[0].hist(np.clip(s["e"], *e_range), bins=e_bins, density=density, label=label, **style)
        axes[1].hist(np.clip(r_all[label], *r_range), bins=r_bins, density=density, label=label, **style)
    axes[0].set(xlabel="energy $E(x)$", ylabel="density", xlim=e_range)
    axes[1].set(xlabel="pair distance $d_{ij}$", ylabel="density", xlim=r_range)
    axes[0].legend(loc="upper right")
    return axes


# ---------------------------------------------------------------------------------------------
# Landscape
# ---------------------------------------------------------------------------------------------
def plot_plane_contour(energy_fn, x0, k, n_particles, spatial_dim, ax, e1=None, e2=None, extent=2.5, n=200,
                       clip=30.0, title=None, colorbar=True, slab=0.5, xlabel="$u$", ylabel="$v$"):
    """Energy as particle k is placed at c + u e1 + v e2 (c = centroid, all other particles fixed).

    Works in 2D and 3D; by default e1, e2 are the x and y axes.  Fixed particles within ``slab`` of
    the plane are drawn (orange); the open circle marks the original position of particle k."""
    import torch
    from matplotlib.ticker import MaxNLocator
    p = np.asarray(x0, dtype=float).reshape(n_particles, spatial_dim)
    p = p - p.mean(0)
    e1 = np.eye(spatial_dim)[0] if e1 is None else np.asarray(e1, float) / np.linalg.norm(e1)
    e2 = np.eye(spatial_dim)[1] if e2 is None else np.asarray(e2, float) / np.linalg.norm(e2)
    us = np.linspace(-extent, extent, n)
    U, V = np.meshgrid(us, us)
    X = np.repeat(p[None], U.size, 0)
    X[:, k] = U.ravel()[:, None] * e1 + V.ravel()[:, None] * e2
    with torch.no_grad():
        E = energy_fn(torch.tensor(X).reshape(U.size, -1)).numpy().reshape(U.shape)
    vmin = E.min()
    levels = np.linspace(vmin, vmin + clip, 25)
    cs = ax.contourf(U, V, np.minimum(E, vmin + clip), levels=levels, cmap=ENERGY_CMAP)
    ax.contour(U, V, E, levels=levels[::3], colors="white", linewidths=0.4, alpha=0.6)
    others = [j for j in range(n_particles) if j != k]
    normal = np.cross(e1, e2) if spatial_dim == 3 else None
    near = [j for j in others if spatial_dim == 2 or abs(p[j] @ normal) < slab]
    ax.scatter(p[near] @ e1, p[near] @ e2, s=35, color=SERIES[1], edgecolor="white", lw=1.0, zorder=4)
    ax.scatter([p[k] @ e1], [p[k] @ e2], s=45, facecolor="none", edgecolor="white", lw=1.4, zorder=4)
    ax.set(aspect="equal", xlabel=xlabel, ylabel=ylabel, xlim=(-extent, extent), ylim=(-extent, extent))
    ax.grid(False)
    if title:
        ax.set_title(title)
    if colorbar:
        cb = plt.colorbar(cs, ax=ax, shrink=0.85, pad=0.02)
        cb.set_label("energy", fontsize=7)
        cb.outline.set_visible(False)
        cb.locator = MaxNLocator(5)
        cb.update_ticks()
        cb.ax.tick_params(labelsize=6)
    return ax


def quench(energy, x, n_steps=4000, lr=1e-3, device="cpu"):
    """Batched steepest descent to the nearest local minimum (inherent structure) of each sample."""
    import torch
    x = torch.as_tensor(np.asarray(x), dtype=torch.float64, device=device).clone()
    opt = torch.optim.Adam([x.requires_grad_(True)], lr=lr)
    for _ in range(n_steps):
        opt.zero_grad()
        energy(x).sum().backward()
        opt.step()
    with torch.no_grad():
        return x.detach().cpu().numpy(), energy(x).cpu().numpy()
