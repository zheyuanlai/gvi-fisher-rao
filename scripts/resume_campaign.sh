#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
export OPENBLAS_NUM_THREADS=1
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
OVERNIGHT_BUDGET_HOURS=${OVERNIGHT_BUDGET_HOURS:-10}

"${REPO_ROOT}/.venv/bin/python" -m fr_gvi.experiments.campaign "${REPO_ROOT}/configs/core" --budget-hours "${OVERNIGHT_BUDGET_HOURS}"

