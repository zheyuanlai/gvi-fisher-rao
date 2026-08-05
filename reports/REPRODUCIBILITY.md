# Reproducibility

## Clean machine

Requirements are CPython 3.11 or newer, a POSIX shell, Git, and a platform supported by the pinned NumPy/SciPy wheels.

```bash
git clone <repository-url> gvi-fisher-rao
cd gvi-fisher-rao
./scripts/bootstrap_env.sh
./scripts/run_smoke.sh
```

The bootstrap creates `.venv`, installs the exact versions in `requirements-lock.txt`, installs the package in editable mode, runs `pip check`, and executes the tests. The smoke wrapper runs validation, the smoke campaign, individual figures, tables, the numerical audit, and the physical plot audit.

## Reproduce campaign tiers

```bash
# Core campaign; safe to rerun and ten hours by default
OVERNIGHT_BUDGET_HOURS=10 ./scripts/run_core_overnight.sh

# Resume only config/source mismatches or unfinished jobs
OVERNIGHT_BUDGET_HOURS=10 ./scripts/resume_campaign.sh

# Expanded and appendix configurations
OVERNIGHT_BUDGET_HOURS=10 ./scripts/run_full.sh
```

Completed jobs are skipped only when the serialized config hash, numerical source hash, and raw output all match. Atomic JSON updates leave exact status after interruption. The numerical hash excludes plotting and report-only modules, so changing a caption does not rerun algorithms.

## Regenerate artifacts without experiments

```bash
./scripts/make_all_figures.sh
make tables
.venv/bin/python -m fr_gvi.experiments.pilot_summary
./scripts/audit_results.sh --allow-failed
```

`make_all_figures.sh` creates individual and five composite manuscript figures, writes matching PDF/PNG files and caption drafts, saves figure-level processed CSV inputs, and runs size/font audits. Outputs are located as follows:

- raw trajectories: `results/raw/<tier>/`;
- run and reference manifests: `results/manifests/`;
- processed figure inputs: `results/processed/`;
- figures and captions: `results/figures/`;
- tables: `results/tables/`;
- campaign, numerical, and plot audits: `reports/`.

## Determinism and provenance

Each job owns a master seed. `numpy.random.SeedSequence` derives target, update-QMC, evaluation-QMC, reference, and paired stochastic streams. Campaign scripts set BLAS thread variables to one. Manifests record UTC times, config, source and reference hashes, Git commit/dirty status, Python/package/platform/CPU/BLAS information, seeds, exact command, operation counts, outputs, status, and any failure reason or traceback.

Core/full raw trajectories are intentionally ignored by Git because they are regenerable; smoke raw data and manifests are retained as regression artifacts. The complete current core raw data remain in the workspace.

