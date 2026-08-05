#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
export OPENBLAS_NUM_THREADS=1
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1

"${REPO_ROOT}/.venv/bin/python" -m pytest -q
"${REPO_ROOT}/.venv/bin/python" -m fr_gvi.experiments.campaign "${REPO_ROOT}/configs/smoke" --budget-hours 0.5
"${REPO_ROOT}/.venv/bin/python" -m fr_gvi.plotting.figures --raw-root "${REPO_ROOT}/results/raw/smoke"
"${REPO_ROOT}/.venv/bin/python" -m fr_gvi.experiments.tables
"${REPO_ROOT}/.venv/bin/python" -m fr_gvi.experiments.audit
"${REPO_ROOT}/.venv/bin/python" -m fr_gvi.plotting.audit

