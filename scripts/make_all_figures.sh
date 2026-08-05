#!/usr/bin/env bash
# Regenerate every figure and table from saved results, without rerunning anything.
set -euo pipefail

REPO_ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
"${REPO_ROOT}/.venv/bin/python" -m fr_gvi.plotting.figures
"${REPO_ROOT}/.venv/bin/python" -m fr_gvi.plotting.main_figures
"${REPO_ROOT}/.venv/bin/python" -m fr_gvi.experiments.tables
"${REPO_ROOT}/.venv/bin/python" -m fr_gvi.plotting.audit
