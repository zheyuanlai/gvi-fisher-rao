#!/usr/bin/env bash
# Resume an interrupted campaign.  Jobs whose config hash and numerical source
# hash match a completed run are skipped; everything else is re-executed.
set -euo pipefail

REPO_ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
export OPENBLAS_NUM_THREADS=1
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1

CORES=$(nproc)
JOBS=${CAMPAIGN_JOBS:-$(( CORES > 64 ? 64 : CORES ))}
OVERNIGHT_BUDGET_HOURS=${OVERNIGHT_BUDGET_HOURS:-10}

"${REPO_ROOT}/.venv/bin/python" -m fr_gvi.experiments.campaign \
  "${REPO_ROOT}/configs/full" \
  --jobs "${JOBS}" \
  --budget-hours "${OVERNIGHT_BUDGET_HOURS}" "$@"
