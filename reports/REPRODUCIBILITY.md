# Reproducibility

## Clean machine

Requirements: CPython 3.11 or newer, a POSIX shell, Git, and a C/Fortran-compatible wheel platform for NumPy/SciPy.

```bash
git clone <repository-url> gvi-fisher-rao
cd gvi-fisher-rao
./scripts/bootstrap_env.sh
./scripts/run_smoke.sh
```

The bootstrap creates `.venv`, installs the exact versions in `requirements-lock.txt`, installs the package in editable mode, and runs `pip check`.

## Reproduce tiers

```bash
# Smoke, including validation, figures, and audit
./scripts/run_smoke.sh

# Resumable core campaign; override the ten-hour default if desired
OVERNIGHT_BUDGET_HOURS=10 ./scripts/run_core_overnight.sh

# Expanded/appendix jobs
OVERNIGHT_BUDGET_HOURS=10 ./scripts/run_full.sh
```

Completed jobs are skipped only when their serialized config hash and source hash match. Delete no state to resume: run `./scripts/resume_campaign.sh`.

## Regenerate without experiments

```bash
./scripts/make_all_figures.sh
make tables
./scripts/audit_results.sh --allow-failed
```

PNG/PDF pairs and caption drafts are in `results/figures/`; processed plot data are in `results/processed/`; terminal tables are in `results/tables/`.

## Determinism

Each job owns a master seed. `numpy.random.SeedSequence` derives separate target, update-QMC, evaluation-QMC, reference, and paired stochastic streams. BLAS thread variables are set to one by campaign shell scripts. Manifests capture Python/package/platform/BLAS information, Git state, reference hashes, and the exact command.

