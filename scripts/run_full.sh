#!/usr/bin/env bash
# Run the full-tier campaign in parallel.  Safe to rerun: jobs whose config and
# code hashes match a completed run are skipped.
set -euo pipefail

REPO_ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)

# One BLAS thread per worker avoids oversubscription across processes.
export OPENBLAS_NUM_THREADS=1
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1
export VECLIB_MAXIMUM_THREADS=1

CORES=$(nproc)
JOBS=${CAMPAIGN_JOBS:-$(( CORES > 64 ? 64 : CORES ))}
OVERNIGHT_BUDGET_HOURS=${OVERNIGHT_BUDGET_HOURS:-10}

mkdir -p "${REPO_ROOT}/results/logs"
"${REPO_ROOT}/.venv/bin/python" -m fr_gvi.experiments.campaign \
  "${REPO_ROOT}/configs/full" "${REPO_ROOT}/configs/appendix" \
  --jobs "${JOBS}" \
  --budget-hours "${OVERNIGHT_BUDGET_HOURS}" "$@" \
  2>&1 | tee "${REPO_ROOT}/results/logs/full_$(date -u +%Y%m%dT%H%M%SZ).log"

"${REPO_ROOT}/.venv/bin/python" -m fr_gvi.plotting.figures
"${REPO_ROOT}/.venv/bin/python" -m fr_gvi.plotting.main_figures
"${REPO_ROOT}/.venv/bin/python" -m fr_gvi.experiments.tables
"${REPO_ROOT}/.venv/bin/python" -m fr_gvi.plotting.audit
