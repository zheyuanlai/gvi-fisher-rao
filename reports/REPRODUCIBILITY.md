# Reproduction guide

## Requirements

- Linux or macOS, Python 3.11--3.13, no GPU.
- Roughly 8 GB of RAM for the campaign; more cores shorten wall time roughly
  linearly up to the number of config files.
- Optional: `poppler-utils` (`pdfinfo`, `pdffonts`) so the plot audit can check
  physical figure width and font embedding.

## 1. Environment

```bash
git clone <repository> gvi-fisher-rao
cd gvi-fisher-rao
./scripts/bootstrap_env.sh      # creates .venv and installs the pinned lock file
make test                       # unit, integration and regression tests
```

`requirements-lock.txt` pins every direct dependency. The test suite must pass
before any campaign is launched; it contains the numerical-validation gates
(exact Gaussian fixed points, one-query quadratic rescue, affine-equivariance
regression, estimator unbiasedness, no-silent-clipping, and the closed-form
curvature constants).

## 2. Smoke tier

```bash
make smoke
```

Finishes in under a minute and exercises every experiment end to end, including
the data, plotting and reporting paths.

## 3. Configuration grids

The full-tier configs under `configs/full/` are generated programmatically and
are committed, so this step is only needed if a grid definition changes:

```bash
make configs        # instantiates each cell and solves its reference
```

Generation is slower than the campaign itself because it solves one reference per
cell in order to compute the certified stepsizes that the sweep is built around.

## 4. Full campaign

```bash
CAMPAIGN_JOBS=64 OVERNIGHT_BUDGET_HOURS=8 make full
```

The runner is resumable and idempotent: a job is skipped when its config hash and
the numerical source hash both match a completed run and its CSV exists. Any edit
to a hashed source file (everything under `src/` except the plotting modules)
invalidates previous runs by design, so restyling figures is free but changing an
algorithm forces a re-run.

Interrupted campaigns resume with:

```bash
CAMPAIGN_JOBS=64 make resume
```

Parallelism is at the config-file level. Each worker owns a private state shard
under `results/manifests/state_shards/`, and the parent merges them into
`results/manifests/campaign_state.json`, so no two processes ever write the same
JSON. Use one BLAS thread per worker (the scripts already export
`OMP_NUM_THREADS=1` and friends) to avoid oversubscription.

## 5. Figures, tables and audits

```bash
make figures        # regenerates every figure and table from saved results only
make tables
make audit
```

`make figures` never reruns an experiment. It reads `results/raw/full/`, writes
paired PDF and PNG files plus the exact processed CSV behind each figure and a
caption draft, then runs the physical plot audit.

## 6. What is and is not in version control

Committed: the source, the configs, the reports, every figure with its caption
draft and its processed input CSV, and the smoke tier's raw data and manifests.

Not committed, because it is regenerable and large: `results/raw/full/`,
`results/raw/appendix/`, the full- and appendix-tier manifests,
`results/manifests/reference_*.json` (which carry full covariance matrices),
`results/manifests/campaign_state.json`, `results/tables/*`, and logs. All of it
is written by `make full`, `make figures` and `make tables`.

Manifests record the working tree's dirty-path count rather than the full
`git status` listing; the commit hash, the dirty flag and the numerical source
hash — the parts that actually identify the code — are recorded in full.

## 7. What to check

- `reports/AUDIT_RESULTS.json` — manifest statuses, forbidden-method scan,
  covariance positivity, and the pathwise covariance-band check of Lemma 4.4.
- `reports/PLOT_AUDIT.json` — PDF/PNG pairing, physical width, font embedding.
- `results/tables/reference_quality.csv` — the certification of every reference
  used to compute an objective gap.
- `results/manifests/full/*.json` — per-run provenance: config, config hash,
  numerical source hash, git commit and dirty flag, Python and package versions,
  platform, CPU count, BLAS configuration, seeds, curvature constants, operation
  counts, status and failure reason.

## 8. Determinism

Every run derives its seeds from `numpy.random.SeedSequence(master_seed,
spawn_key=(stream, repeat))`, and both the master seed and the derived run seed
are stored in the manifest. Re-running a completed job with `--force` reproduces
its CSV bit-for-bit on the same platform and package versions. Results are not
guaranteed bit-identical across BLAS implementations; the reported quantities are
stable far beyond the precision at which they are interpreted.

## 9. Known non-reproducible quantities

Wall-clock timings are machine specific and are recorded for within-run
comparison only. Peak RSS is process-level and reflects the worker, not a single
trajectory.
