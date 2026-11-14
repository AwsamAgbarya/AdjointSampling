# AdjointSampling

From-scratch PyTorch re-implementation of *Adjoint Sampling: Highly Scalable Diffusion Samplers via
Adjoint Matching* (Havens et al., ICML 2025) for the TU Berlin course *Project Machine Learning*.

**Milestone 1** (theory, energy functions, prototype) covers DW-4 and LJ-13 energies, MCMC reference
samples, data pipeline, metrics and a first Adjoint Sampling prototype trained on DW-4.

## Layout

```
adjsamp/            library code
  energies.py       DW-4 and LJ-13 energies (iDEM conventions), E and grad E via autograd
  mcmc.py           MH, MALA, HMC on the zero-centre-of-mass subspace + diagnostics (ACF, tau_int, R-hat)
  data.py           reference storage, chain-wise train/val/test split, lazy (x, E(x)) generators, transforms
  metrics.py        energy W2, symmetry-aware (rotation + permutation) W2
  sde.py            geometric noise schedule, base process (marginals, bridge), Euler-Maruyama sampler
  model.py          Model(nn.Module): u_theta(X_t, t) with MLP or EGNN backbone
  train.py          replay buffer, Reciprocal Adjoint Matching loss, Algorithm 1 training loop
  plotting.py       report figures (configuration grids with overlays, histograms, contours)
visualizations.py   DW-4 landscape plots (pair potential, 1D slices, 2D contours)
scripts/            run_mcmc.py, train_prototype.py, evaluate.py, ablate_sampling.py, make_figures.py
tests/              pytest sanity checks (gradients, invariances, bridge, EGNN equivariance, metrics)
report/figures/     figures written by make_figures.py (the report itself is kept on Overleaf)
testing.ipynb       exploration notebook
```

## Setup

```bash
python -m venv .venv
.venv/Scripts/activate            # Windows;  source .venv/bin/activate on Linux
pip install torch --index-url https://download.pytorch.org/whl/cu128
pip install -r requirements.txt
pytest -q tests
```

iDEM reference data (for comparison only) goes into `data/idem/`:
`{train,val,test}_split_DW4.npy`, `{train,val,test}_split_LJ13-1000.npy` from
<https://github.com/jarridrb/DEM/tree/main/data>.

## Reproduce Milestone 1

```bash
# reference samples (≈ 1 min / 3 min / 1 min / 8 min on an RTX 5080 laptop GPU)
python scripts/run_mcmc.py --system dw4  --method mh  --chains 1024 --warmup 5000  --steps 50000  --thin 10  --step-size 0.3
python scripts/run_mcmc.py --system dw4  --method hmc --chains 1024 --pre-mh 3000  --warmup 2000 --steps 20000 --thin 4  --step-size 0.1
python scripts/run_mcmc.py --system lj13 --method mh  --chains 1024 --warmup 20000 --steps 100000 --thin 100 --step-size 0.05
python scripts/run_mcmc.py --system lj13 --method hmc --chains 1024 --pre-mh 20000 --warmup 5000 --steps 30000 --thin 10 --step-size 0.01
# Adjoint Sampling prototype on DW-4
# (≈ 22 min EGNN / 6 min MLP; evaluate.py and make_figures.py read runs/dw4_{egnn,mlp}_final)
python scripts/train_prototype.py --arch egnn --n-outer 500 --n-inner 200 --sde-steps 1000 --out runs/dw4_egnn_final
python scripts/train_prototype.py --arch mlp --hidden 256 --layers 4 --n-outer 500 --n-inner 200 --sde-steps 1000 --out runs/dw4_mlp_final
# metrics, sampling-steps ablation and figures (results/*.json, report/figures/*.pdf)
python scripts/evaluate.py
python scripts/ablate_sampling.py
python scripts/make_figures.py
```

Energy conventions follow the iDEM code exactly (needed to compare with their data): DW-4 sums over
the 6 unique pairs; LJ-13 sums the LJ term over *ordered* pairs (= 2x the textbook energy) plus
`0.5 * sum ||x_i - x_com||^2`.

> Cluster note: coding agents must not be used on the ML cluster. All scripts are plain Python and
> can be submitted as ordinary jobs.
