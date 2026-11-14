"""Produce all report figures (vector PDF) in report/figures/ and summary numbers in results/.

Run after run_mcmc.py (dw4/lj13 x mh/hmc), train_prototype.py (egnn, mlp) and evaluate.py.
"""
import json
import os
import sys

import matplotlib.pyplot as plt
import numpy as np
import torch

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
from adjsamp import plotting as P  # noqa: E402
from adjsamp.data import chain_split, collect, load_idem, load_reference_splits  # noqa: E402
from adjsamp.energies import DoubleWell, LennardJones  # noqa: E402
from adjsamp.mcmc import autocorrelation, integrated_autocorr_time  # noqa: E402
from visualizations import SERIES, plot_pair_potential  # noqa: E402

FIG = os.path.join(ROOT, "report", "figures")
DW, LJ = DoubleWell(), LennardJones()
DW_MINIMA = {  # local minima of DW-4 (verified: energies reproduce E(x) to 1e-4)
    "kite": [[-1.6379, 0.0], [-1.0725, -2.6391], [-1.0725, 2.6391], [3.7830, 0.0]],
    "triangle + tail": [[-3.3160, 0.0], [-0.4669, 0.0], [1.8914, 1.2385], [1.8914, -1.2385]],
    "rectangle": [[-2.5808, -1.2062], [-2.5808, 1.2062], [2.5808, 1.2062], [2.5808, -1.2062]],
    "rhombus": [[-2.4297, 0.0], [0.0, 1.1923], [0.0, -1.1923], [2.4297, 0.0]],
    "centred triangle": [[-2.8775, 1.0207], [0.0, 0.0], [0.5548, -3.0024], [2.3227, 1.9816]],
}


def ref(path, split="test"):
    s = np.load(os.path.join(ROOT, path, "samples.npy"), mmap_mode="r")
    e = np.load(os.path.join(ROOT, path, "energies.npy"), mmap_mode="r")
    idx = chain_split(s.shape[1])[split]
    return np.asarray(s[:, idx]).reshape(-1, s.shape[-1]), np.asarray(e[:, idx]).reshape(-1)


def meta(path):
    return json.load(open(os.path.join(ROOT, path, "meta.json")))


def fig_pair_potentials():
    fig, ax = plt.subplots(1, 2, figsize=(P.TEXTWIDTH_IN, 2.0), layout="constrained")
    r_s, r_l = DW.well_radii
    r = np.linspace(0, 8, 800)
    ax[0].plot(r, DW.pair_potential(r), color=SERIES[0])
    ax[0].axhline(0, color=P.AXIS, lw=0.8)
    for rr, txt in [(r_s, f"min\n$r={r_s:.2f}$"), (DW.d0, "barrier\n$r=d_0=4$"), (r_l, f"min\n$r={r_l:.2f}$")]:
        ax[0].axvline(rr, color=P.MUTED, lw=0.8, ls="--", zorder=0)
        ax[0].annotate(txt, (rr, 4.0), ha="center", va="bottom", fontsize=6.5, color=P.INK2,
                       bbox=dict(fc="white", ec="none", pad=0.5))
    ax[0].set(xlim=(0, 8), ylim=(-6, 20), title="DW-4: $v(r)=0.9(r-4)^4-4(r-4)^2$",
              xlabel="pair distance $r$", ylabel="$v(r)$")
    r = np.linspace(0.85, 3.0, 400)
    ax[1].plot(r, LJ.pair_factor * LJ.pair_potential(r), color=SERIES[0], label="pair term $2\\,v_{LJ}(r)$")
    ax[1].axhline(0, color=P.AXIS, lw=0.8)
    ax[1].axvline(1.0, color=P.MUTED, lw=0.8, ls="--")
    ax[1].annotate("min $r=r_m=1$", (1.0, 3.6), xytext=(3, 0), textcoords="offset points", fontsize=7, color=P.INK2)
    ax[1].set(xlim=(0.85, 3.0), ylim=(-2.5, 4), title="LJ-13: $v_{LJ}(r)=(1/r)^{12}-2(1/r)^{6}$",
              xlabel="pair distance $r$", ylabel="energy")
    ax[1].legend(loc="upper right")
    P.save(fig, os.path.join(FIG, "pair_potentials.pdf"))


def fig_landscape(lj_min):
    """Row 1: DW-4 minima. Row 2: DW-4 contour, LJ-13 ground state (3D), LJ-13 contour."""
    fig = plt.figure(figsize=(P.TEXTWIDTH_IN, 3.9), layout="constrained")
    sub = fig.subfigures(2, 1, height_ratios=[1, 1.45])
    top = sub[0].subplots(1, 5)
    for ax, (name, x) in zip(top, DW_MINIMA.items()):
        e = DW(torch.tensor([x], dtype=torch.float64)).item()
        P.draw_configuration(ax, x, 4, 2, DW.d0, title=f"{name}\n$E={e:.2f}$", lim=4.6)
    bot = sub[1].subfigures(1, 3, width_ratios=[1.15, 0.8, 1.15])
    a = bot[0].subplots()
    P.plot_plane_contour(DW, np.array(DW_MINIMA["kite"]), 3, 4, 2, a, extent=6.5, clip=30,
                         title="DW-4: particle 3", xlabel="$x$", ylabel="$y$")
    a3 = bot[1].add_subplot(projection="3d")
    P.draw_configuration(a3, lj_min, 13, 3, 1.3, title=f"LJ-13 global min.\n$E={LJ(torch.tensor(lj_min)[None]).item():.2f}$")
    b = bot[2].subplots()
    p = lj_min.reshape(13, 3) - lj_min.reshape(13, 3).mean(0)
    k = int(np.argmax(p[:, 2]))           # a surface atom; plane through the centre containing its radial axis
    e1 = p[k] / np.linalg.norm(p[k])
    j = int(np.argmin([np.linalg.norm(p[i] - p[k]) if i != k and np.linalg.norm(p[i]) > 0.5 else 9 for i in range(13)]))
    e2 = p[j] - (p[j] @ e1) * e1         # direction towards a neighbouring surface atom
    P.plot_plane_contour(LJ, p, k, 13, 3, b, e1=e1, e2=e2, extent=2.2, n=220, clip=25, slab=0.35,
                         title=f"LJ-13: atom {k}", xlabel="radial $u$", ylabel="tangential $v$")
    P.save(fig, os.path.join(FIG, "landscape.pdf"))


def fig_mcmc():
    fig, ax = plt.subplots(1, 3, figsize=(P.TEXTWIDTH_IN, 2.05), layout="constrained")
    rows = {}
    for col, sys_ in enumerate(["dw4", "lj13"]):
        for c, method in enumerate(["mh", "hmc"]):
            path = f"data/reference/{sys_}_{method}"
            tr = np.load(os.path.join(ROOT, path, "energy_trace.npy"))
            m = meta(path)
            gps = max(m["grads_per_step"], 1)
            max_lag = min(tr.shape[0] // 2, int(8 * m["tau_int_energy_steps"]) + 10)
            acf = autocorrelation(tr, max_lag)
            lags = np.arange(1, len(acf))
            ax[col].plot(lags * gps, acf[1:], color=SERIES[c],
                         label=f"{method.upper()} ($\\tau_{{int}}$={m['tau_int_energy_steps'] * gps:.0f} evals)")
            rows[f"{sys_}_{method}"] = m
        ax[col].axhline(0, color=P.AXIS, lw=0.8)
        ax[col].set(xscale="log", xlabel="lag [energy evaluations]", ylabel="ACF of $E$", ylim=(-0.05, 1.0),
                    title=f"{'DW-4' if sys_ == 'dw4' else 'LJ-13'}: energy autocorrelation")
        ax[col].legend(loc="lower left")
    # warm-up traces: LJ-13 MH equilibration and the DW-4 HMC failure without pre-equilibration
    w = np.load(os.path.join(ROOT, "data/reference/dw4_hmc_no_preequil/warmup_trace.npy"))
    e = np.load(os.path.join(ROOT, "data/reference/dw4_hmc_no_preequil/energies.npy"))
    stuck = e.mean(0) > 0
    t = np.arange(w.shape[0])
    ax[2].plot(t, np.median(w[:, ~stuck], 1), color=SERIES[0], label=f"{(~stuck).sum()} chains (median)")
    ax[2].plot(t, np.median(w[:, stuck], 1), color=SERIES[1], label=f"{stuck.sum()} stuck chains (median)")
    ax[2].set_yscale("symlog", linthresh=30)
    ax[2].set_yticks([-20, 0, 100, 1000, 10000])
    ax[2].set_yticklabels(["-20", "0", "$10^2$", "$10^3$", "$10^4$"])
    ax[2].set(xlabel="HMC warm-up step", ylabel="energy", title="DW-4 HMC, no pre-equilibration")
    ax[2].legend(loc="center right")
    P.save(fig, os.path.join(FIG, "mcmc_diagnostics.pdf"))
    return rows


def fig_reference_vs_idem():
    fig, ax = plt.subplots(2, 2, figsize=(P.TEXTWIDTH_IN, 3.6), layout="constrained")
    for row, (sys_, E, n, d, er, rr) in enumerate([("dw4", DW, 4, 2, (-26, -12), (0, 8)),
                                                   ("lj13", LJ, 13, 3, (-70, -15), (0.7, 4.6))]):
        x_id = load_idem(sys_, "test")
        sets = {"iDEM data": {"x": x_id, "e": E(torch.tensor(x_id)).numpy()}}
        for m in ("mh", "hmc"):
            x, e = ref(f"data/reference/{sys_}_{m}")
            k = np.random.default_rng(0).choice(len(e), 20_000, replace=False)
            sets[f"ours, {m.upper()}"] = {"x": x[k], "e": e[k]}
        P.plot_sample_stats(sets, n, d, axes=ax[row], e_range=er, r_range=rr, reference="iDEM data")
        ax[row, 0].set_title(f"{'DW-4' if sys_ == 'dw4' else 'LJ-13'}: energy")
        ax[row, 1].set_title(f"{'DW-4' if sys_ == 'dw4' else 'LJ-13'}: pair distances")
    P.save(fig, os.path.join(FIG, "reference_vs_idem.pdf"))


def fig_reference_samples():
    fig = plt.figure(figsize=(P.TEXTWIDTH_IN, 2.6), layout="constrained")
    sub = fig.subfigures(2, 1)
    tr, _, _ = load_reference_splits(os.path.join(ROOT, "data/reference/dw4_hmc"), seed=3)
    X, Y = collect(tr, 6)
    axs = sub[0].subplots(1, 6)
    for ax, x, y in zip(axs, X, Y):
        P.draw_configuration(ax, x, 4, 2, DW.d0, title=f"$E={y:.2f}$", lim=4.6)
    tr, _, _ = load_reference_splits(os.path.join(ROOT, "data/reference/lj13_hmc"), seed=3)
    X, Y = collect(tr, 6)
    for i, (x, y) in enumerate(zip(X, Y)):
        ax = sub[1].add_subplot(1, 6, i + 1, projection="3d")
        P.draw_configuration(ax, x, 13, 3, 1.3, title=f"$E={y:.2f}$", lim=1.9)
    P.save(fig, os.path.join(FIG, "reference_samples.pdf"))


def fig_basins():
    """Quench reference samples to their local minimum and count basin occupancies."""
    out = {}
    fig, ax = plt.subplots(1, 2, figsize=(P.TEXTWIDTH_IN, 1.9), layout="constrained")
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    for i, (sys_, E, lr, bins) in enumerate([("dw4", DW, 5e-3, np.arange(-26.0, -20.5, 0.05)),
                                             ("lj13", LJ, 1e-3, np.linspace(-84, -70, 120))]):
        x, _ = ref(f"data/reference/{sys_}_hmc")
        k = np.random.default_rng(0).choice(len(x), 4000, replace=False)
        _, emin = P.quench(E, x[k], n_steps=6000, lr=lr, device=dev)
        ax[i].hist(emin, bins=bins, color=SERIES[0], weights=np.full(len(emin), 1 / len(emin)))
        ax[i].set(xlabel="energy of local minimum", ylabel="fraction of samples",
                  title=f"{'DW-4' if sys_ == 'dw4' else 'LJ-13'}: basins of reference samples")
        vals, cnt = np.unique(np.round(emin, 1), return_counts=True)
        order = np.argsort(-cnt)[:8]
        out[sys_] = {f"{vals[j]:.1f}": float(cnt[j] / len(emin)) for j in order}
    P.save(fig, os.path.join(FIG, "basins.pdf"))
    return out


def fig_prototype():
    runs = {a: os.path.join(ROOT, f"runs/dw4_{a}_final") for a in ("egnn", "mlp")}
    fig, ax = plt.subplots(1, 3, figsize=(P.TEXTWIDTH_IN, 2.0), layout="constrained")
    for c, (a, r) in enumerate(runs.items()):
        h = json.load(open(os.path.join(r, "history.json")))
        ev = np.array(h["n_energy_evals"]) / 1e3
        w = np.array(h["energy_w2"])
        ax[0].plot(ev, w, color=SERIES[c], label=a.upper(), lw=1.0)
    ax[0].set(yscale="log", xlabel="energy evaluations [$10^3$]", ylabel="energy $W_2$ (fresh batch)",
              title="training progress")
    ax[0].legend()
    x_te, e_te = ref("data/reference/dw4_hmc")
    k = np.random.default_rng(0).choice(len(e_te), 10_000, replace=False)
    sets = {"reference (HMC)": {"x": x_te[k], "e": e_te[k]}}
    for a, r in runs.items():
        sets[f"AS prototype ({a.upper()})"] = {"x": np.load(os.path.join(r, "samples.npy")),
                                               "e": np.load(os.path.join(r, "energies.npy"))}
    P.plot_sample_stats(sets, 4, 2, axes=ax[1:], e_range=(-26, -10), r_range=(0, 8), reference="reference (HMC)")
    h, lab = ax[1].get_legend_handles_labels()
    ax[1].get_legend().remove()
    fig.legend(h, lab, loc="outside lower center", ncol=3)
    ax[1].set_title("energy")
    ax[2].set_title("pair distances")
    P.save(fig, os.path.join(FIG, "prototype.pdf"))

    # sample configurations + one trajectory per panel (overlay hook of the grid function)
    traj = np.load(os.path.join(runs["egnn"], "trajectories.npy"))  # (T, 64, 8)
    xs = traj[-1, :5]
    es = DW(torch.tensor(xs)).numpy()

    def overlay(axh, i, _x):
        tr = traj[:, i].reshape(-1, 4, 2)
        for j in range(4):
            axh.plot(tr[:, j, 0] - tr[-1].mean(0)[0], tr[:, j, 1] - tr[-1].mean(0)[1], color=SERIES[j], lw=0.6, alpha=0.8)

    fig, _ = P.plot_configuration_grid(xs, 4, 2, energies=es, ncols=5, bond_cutoff=DW.d0, lim=6.5,
                                       overlays=[overlay])
    P.save(fig, os.path.join(FIG, "prototype_samples.pdf"))


def main():
    P.set_style()
    lj_min = np.load(os.path.join(ROOT, "data/lj13_minimum.npy")).reshape(-1)
    fig_pair_potentials()
    fig_landscape(lj_min)
    mc = fig_mcmc()
    fig_reference_vs_idem()
    fig_reference_samples()
    basins = fig_basins()
    if os.path.exists(os.path.join(ROOT, "runs/dw4_mlp_final/history.json")):
        fig_prototype()
    os.makedirs(os.path.join(ROOT, "results"), exist_ok=True)
    with open(os.path.join(ROOT, "results", "summary.json"), "w") as f:
        json.dump({"mcmc": mc, "basins": basins}, f, indent=2)
    print(json.dumps({"basins": basins}, indent=1))


if __name__ == "__main__":
    main()
