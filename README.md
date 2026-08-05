# Gaussian GVI via Fisher--Rao forward--backward algorithms

This repository is a float64 NumPy/SciPy reference implementation and reproducible experiment campaign for Gaussian variational inference under Fisher--Rao geometry. It implements exactly the methods admitted by the scientific protocol:

- deterministic FR--R and FR--KL;
- stochastic Price/Hessian--STL FR--R--STL and FR--KL--STL;
- deterministic FB--GVI and minibatch S--FB--GVI from Diao et al.;
- Laplace only as a noniterative logistic-regression approximation baseline.

It does not contain BWGD, BW--SGD, covariance-projected Fisher--Rao schemes, or mixed geometries. Invalid covariance updates are recorded as failures; only logged roundoff-scale repairs bounded by `100 * eps * matrix_scale` are allowed.

The completed active campaign contains 45 smoke and 119 core algorithm jobs, with no failures or pending jobs. Two bump-train configs are intentionally skipped because the exact construction is absent from the supplied sources. The expanded full/appendix grids remain unrun.

## Quick start

```bash
./scripts/bootstrap_env.sh
./scripts/run_smoke.sh
```

## Campaign commands

```bash
# Ten-hour default, resumable and idempotent
./scripts/run_core_overnight.sh

# Verify or resume matching core jobs
OVERNIGHT_BUDGET_HOURS=10 ./scripts/resume_campaign.sh

# Expanded and appendix configurations
OVERNIGHT_BUDGET_HOURS=10 ./scripts/run_full.sh

# Artifact-only operations
./scripts/make_all_figures.sh
make tables
.venv/bin/python -m fr_gvi.experiments.pilot_summary
./scripts/audit_results.sh --allow-failed
```

Every run stores its config, numerical source hash, reference hashes, Git state, platform/package information, seeds, operation accounting, status, and output paths under `results/manifests/`. Raw trajectories retain individual seeds. The runner stops launching jobs at its budget and records the pending count atomically.

## Numerical and plotting policy

All SPD functions use symmetric eigendecompositions; solves use Cholesky factors. Deterministic methods share fixed expectation designs per job. Update and evaluation QMC designs use independent seeds. Stochastic methods use paired streams where common random numbers are meaningful.

Figures follow the referenced academic plotting protocol: exact 7-inch width, Matplotlib, accessible colors plus redundant encodings, median and 10--90% bands, paired PDF/PNG output, embedded fonts, caption drafts, and saved processed input rows. The five manuscript composites are `results/figures/main_figure_1.*` through `main_figure_5.*`.

See [implementation notes](reports/IMPLEMENTATION_NOTES.md), [numerical audit](reports/NUMERICAL_AUDIT.md), [overnight summary](reports/OVERNIGHT_SUMMARY.md), and [reproduction guide](reports/REPRODUCIBILITY.md).

