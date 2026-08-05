# Gaussian GVI via Fisher--Rao forward--backward algorithms

This repository is a float64 NumPy/SciPy reference implementation and reproducible experiment campaign for Gaussian variational inference under Fisher--Rao geometry. It implements exactly the methods admitted by the scientific protocol:

- deterministic FR--R and FR--KL;
- stochastic Price/Hessian--STL FR--R--STL and FR--KL--STL;
- deterministic FB--GVI and minibatch S--FB--GVI from Diao et al.;
- Laplace only as a noniterative logistic-regression approximation baseline.

It does not contain BWGD, BW--SGD, covariance-projected Fisher--Rao schemes, or mixed geometries. Invalid covariance updates are recorded as failures; only roundoff-scale repairs bounded by `100 * eps * matrix_scale` are permitted and logged.

## Quick start

```bash
./scripts/bootstrap_env.sh
./scripts/run_smoke.sh
```

The smoke command runs the test suite, experiments A--L at tiny scale, figure regeneration, table generation, and the result audit. Experiment E is an intentional skipped manifest because no exact bump-train formula appears in the supplied manuscript.

## Campaign commands

```bash
# Ten-hour default, resumable and idempotent
./scripts/run_core_overnight.sh

# Resume matching jobs without rerunning completed config+code hashes
OVERNIGHT_BUDGET_HOURS=4 ./scripts/resume_campaign.sh

# Expanded and appendix configurations
./scripts/run_full.sh

# Artifact-only operations
./scripts/make_all_figures.sh
make tables
./scripts/audit_results.sh --allow-failed
```

Every run stores its config, source hash, reference hashes, Git state, platform/package information, seeds, accounting, status, and output paths under `results/manifests/`. Raw trajectories retain individual seeds under `results/raw/`. The runner stops launching new jobs when its budget expires and leaves an exact pending count in `results/manifests/campaign_state.json`.

## Numerical policy

All SPD functions use symmetric eigendecompositions; solves use Cholesky factors. Deterministic methods share fixed expectation designs per target/config. Scrambled Sobol designs for updates and evaluation use independent seeds. Stochastic methods share run seeds across methods where common random numbers are meaningful. The minimum covariance eigenvalue is recorded at every iteration.

See [implementation notes](reports/IMPLEMENTATION_NOTES.md), [numerical audit](reports/NUMERICAL_AUDIT.md), [overnight summary](reports/OVERNIGHT_SUMMARY.md), and [reproduction guide](reports/REPRODUCIBILITY.md).

