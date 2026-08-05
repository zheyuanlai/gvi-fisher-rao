# Overnight campaign summary

## Execution status

The active smoke-plus-core campaign finished on 2026-08-05 with 164 completed jobs, zero failed, zero interrupted, zero pending, and two intentional skips. The smoke tier accounts for 45 completed jobs and one skip; the core tier accounts for 119 completed jobs and one skip. The final resumability pass reproduced that exact accounting in `results/manifests/campaign_state.json`.

The two skips are the smoke and core forms of Experiment E. The supplied manuscript and numerical plan name a bump-train construction but do not provide its potential, smoothing rule, or constants. No surrogate target was invented. See `reports/BLOCKED_EXPERIMENTS.md`.

The expanded `full` and `appendix` grids have not been run. In particular, the complete condition-number, dimension, stepsize-stability, feature-conditioning, and 30-seed stochastic sweeps remain future work. Composite figures label single-cell pilot panels and pending grids explicitly.

## Main numerical observations

These are core-pilot observations, not claims about the unrun full grids.

- Covariance burn-in is logarithmic at the tested hard starts. For FR--R, `N_cov h` is 11.5 and 20.7 at initial covariance scales `1e-8` and `1e-12`, compared with `log[1/(beta lambda_0)]` values 11.513 and 20.723. FR--KL gives 12.1 and 21.8.
- In the transformed Gaussian test with `cond(A A^T)=1e4`, maximum covariance discrepancies are `1.22e-13` for FR--R and `3.82e-13` for FR--KL, versus `5.19e-1` for FB--GVI. This is a direct iterate-level affine-equivariance test, not a cross-geometry gradient-norm comparison.
- On the single Diao anisotropic Gaussian instance and chosen core stepsizes, terminal exact KL gaps are `6.17e-16` (FR--R), `1.70e-11` (FR--KL), and `2.73e1` (FB--GVI). The ordering is conditional on this stepsize and 80-pair budget. FR--R+QR and FR--KL+QR both recover the Gaussian solution after exactly one gradient--Hessian query and are kept out of the competitive curves.
- On the shifted log-cosh core cell, terminal objective gaps after 100 iterations are 1.30 (FR--R), 1.40 (FR--KL), and 4.22 (FB--GVI). The complete stepsize grid is unrun, so this is not a general performance ranking.
- In the exact-Gaussian local experiment, terminal KL gaps are `3.53e-10` (FR--R) and `3.97e-8` (FR--KL). In the near-Gaussian spectral pilot, the predicted `2 gamma_star` is 1.996; fitted tail rates are 2.166 and 1.885.
- With covariance matched to a Gaussian target, maximum across-seed objective spreads are below `8.0e-15` for the two FR--STL methods, while S--FB--GVI retains a spread of 0.354. This compares complete algorithms with their native estimators; it is not a geometry-only comparison.
- The STL/raw intrinsic variance ratio ranges from 0.0358 to 0.1867 over the tested optimizer-relative distances. FR--STL tail floors fall from about `5.1e-3` at `B=1` to `1.3e-4` at `B=64`; the S--FB--GVI core cell falls from 0.0459 to 0.0306 under its fixed selected step.
- Decreasing-step log--log tail slopes are -0.903 (FR--R--STL) and -0.891 (FR--KL--STL), consistent with but not proof of the predicted inverse-iteration regime.
- In the single logistic core cell, Laplace has the smallest approximate Gaussian-VI objective gap (0.0302). The lowest terminal predictive NLL is obtained by FR--R--STL (0.6813), closely followed by FR--KL--STL (0.6815), while their 50-iteration objective gaps remain larger than those of the deterministic methods. The reference is a finite-QMC optimizer with squared FR residual `2.34e-5`; small objective differences should therefore not be overinterpreted.

The machine-readable values, including every terminal metric and stochastic quantile, are in `reports/PILOT_RESULTS.json`.

## Failures and numerical concerns

No active smoke or core algorithm job failed, no job was interrupted, and no pending job remains. Strict SPD checks were active throughout. Roundoff-scale negative Gaussian gaps (approximately `-8e-16`) in the rescue verification are numerical zero, not negative KL values.

Unresolved scientific limitations are the missing theorem-specific bump-train formula, the finite-QMC logistic reference residual, and the unrun full/appendix sweeps. These limitations are visible in captions and reports rather than hidden.

## Artifacts

- Manuscript figures: `results/figures/main_figure_1.{pdf,png}` through `main_figure_5.{pdf,png}`.
- Caption drafts: `results/figures/main_figure_1.md` through `main_figure_5.md`.
- Figure-level processed inputs: `results/processed/main_figure_1.csv` through `main_figure_5.csv`.
- Individual experiment figures and captions: `results/figures/experiment_*`.
- Terminal tables: `results/tables/terminal_summary.csv` and `terminal_summary.tex`.
- Numerical and plot audits: `reports/AUDIT_RESULTS.json`, `reports/PLOT_AUDIT.json`, and `reports/NUMERICAL_AUDIT.md`.

## Resume commands

The active core campaign is complete, so this command now verifies hashes and skips matching jobs:

```bash
OVERNIGHT_BUDGET_HOURS=10 ./scripts/resume_campaign.sh
```

Run the expanded and appendix configurations with:

```bash
OVERNIGHT_BUDGET_HOURS=10 ./scripts/run_full.sh
```

