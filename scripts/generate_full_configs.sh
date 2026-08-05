#!/usr/bin/env bash
# Regenerate every full-tier config from the programmatic grid definitions.
set -euo pipefail
cd "$(dirname "$0")/.."

export OMP_NUM_THREADS=8
export OPENBLAS_NUM_THREADS=8
export MKL_NUM_THREADS=8

source .venv/bin/activate
python -m fr_gvi.experiments.grids --destination configs/full "$@"
