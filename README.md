# Affine-invariant Fisher-Rao convergence

NumPy and SciPy code for *Affine Invariant Convergence of Fisher-Rao Gradient
Flows and Their Discretizations for Gaussian Variational Inference* by Zheyuan
Lai and Yifan Chen.

The repository contains the Gaussian variational inference algorithms,
numerical checks, experiment configurations, and figures used in the paper.

## Installation

The project requires Python 3.11 or newer.

```bash
git clone https://github.com/zheyuanlai/gvi-fisher-rao.git
cd gvi-fisher-rao
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip setuptools wheel
python -m pip install -e ".[dev]"
python -m pip check
```

## Quick validation

From the repository root, run:

```bash
source .venv/bin/activate
python -m fr_gvi.experiments.datasets --fetch --write-manifest
python -m pytest -q
./scripts/run_smoke.sh
```

## Methods

| Method | Description |
| --- | --- |
| FR-R | Fisher-Rao Riemannian gradient update |
| FR-KL | Fisher-Rao Bregman/KL update |
| FR-R-STL, FR-KL-STL | Sticking-the-landing stochastic updates |
| BW-FB, S-BW-FB | Bures-Wasserstein baselines |
| Sq-NGVI | Square-root natural-gradient variational inference |
| Price-BBVI | Bonnet-Price black-box variational inference |
| BBVI-STL | Projected black-box variational inference with STL |
| Laplace | Logistic-regression Laplace approximation |

## Experiments in the paper

### Figure 1: covariance recovery and affine invariance

Gaussian targets test covariance recovery, affine equivariance, and coordinate
anisotropy. The Fisher-Rao methods are compared with BW-FB after non-orthogonal
coordinate changes and on an anisotropic Gaussian with condition number `10^9`.

### Figure 2: local rates on shifted log-cosh targets

Shifted log-cosh targets test global contraction, iteration counts across
dimensions and curvature, and agreement with the linearized local rate.

### Figure 3: Monte Carlo scaling and initialization

The stochastic experiments measure Gaussian pathwise cancellation, the stationary
gap as batch size and step size vary, and inverse-Hessian covariance
initialization with a Newton mean correction.

### Figure 4: Bayesian logistic regression

The benchmark uses pima, wdbc, ionosphere, sonar, and spambase. It reports
objective gaps, held-out predictive negative log-likelihood, and update time to
a relative objective gap of `10^-4`.

## Running the manuscript experiments

Activate the virtual environment, then run the following commands from the
repository root:

```bash
source .venv/bin/activate
python -m fr_gvi.experiments.datasets --fetch --write-manifest
python -m fr_gvi.experiments.manuscript --pilot
python -m fr_gvi.experiments.campaign configs/manuscript/pilot --budget-hours 0.5
python -m fr_gvi.experiments.pilot --write
python -m fr_gvi.experiments.tuning --batch-size 16
python -m fr_gvi.experiments.manuscript
./scripts/run_manuscript.sh
python -m fr_gvi.plotting.manuscript_figures
python -m fr_gvi.plotting.audit --figures results/figures/manuscript
python -m fr_gvi.experiments.manuscript_tables
python -m fr_gvi.experiments.manuscript_audit
```

`run_manuscript.sh` can be resumed. Set `MANUSCRIPT_JOBS`,
`MANUSCRIPT_THREADS`, and `MANUSCRIPT_BUDGET_HOURS` for the machine.

## Repository layout

```text
src/fr_gvi/       algorithms, targets, diagnostics, experiments, plotting
configs/          smoke and manuscript experiment configurations
scripts/          setup, campaign, plotting, and audit entry points
results/          figures, tables, manifests, and processed experiment data
reports/          numerical audit, protocol, implementation, and reproduction notes
tests/            unit, integration, and regression tests
```
