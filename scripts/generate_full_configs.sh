#!/usr/bin/env bash
# Regenerate every full-tier config from the programmatic grid definitions.
# This instantiates each cell, solves its reference and records the certified
# stepsizes, so it is noticeably slower than running the campaign itself.
set -euo pipefail

REPO_ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
"${REPO_ROOT}/.venv/bin/python" -m fr_gvi.experiments.grids \
  --destination "${REPO_ROOT}/configs/full" "$@"
