#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
"${REPO_ROOT}/.venv/bin/python" -m fr_gvi.plotting.figures --raw-root "${REPO_ROOT}/results/raw"
"${REPO_ROOT}/.venv/bin/python" -m fr_gvi.plotting.main_figures
"${REPO_ROOT}/.venv/bin/python" -m fr_gvi.plotting.audit

