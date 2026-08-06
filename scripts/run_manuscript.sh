#!/usr/bin/env bash
# Run the reduced manuscript campaign: 134 deterministic trajectories over the
# six preregistered config groups.  Safe to rerun: jobs whose config and code
# hashes match a completed run are skipped.
set -euo pipefail

REPO_ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)

CONFIGS="${REPO_ROOT}/configs/manuscript"
if [[ ! -d "${CONFIGS}" ]]; then
  echo "missing ${CONFIGS}; run 'make manuscript-configs' first" >&2
  exit 1
fi

# Threads per worker.  The campaign parallelizes over configs, so each worker
# gets a modest BLAS pool rather than the whole machine.
THREADS=${MANUSCRIPT_THREADS:-4}
export OPENBLAS_NUM_THREADS="${THREADS}"
export OMP_NUM_THREADS="${THREADS}"
export MKL_NUM_THREADS="${THREADS}"
export NUMEXPR_NUM_THREADS="${THREADS}"
export VECLIB_MAXIMUM_THREADS="${THREADS}"

CORES=$(nproc)
JOBS=${MANUSCRIPT_JOBS:-$(( CORES / THREADS > 32 ? 32 : (CORES / THREADS > 1 ? CORES / THREADS : 1) ))}
BUDGET_HOURS=${MANUSCRIPT_BUDGET_HOURS:-4}

mkdir -p "${REPO_ROOT}/results/logs"
"${REPO_ROOT}/.venv/bin/python" -m fr_gvi.experiments.campaign \
  "${CONFIGS}/figure1_gaussian_burnin" \
  "${CONFIGS}/figure1_affine_equivariance" \
  "${CONFIGS}/figure1_anisotropic_gaussian" \
  "${CONFIGS}/figure2_logcosh_global" \
  "${CONFIGS}/figure2_logcosh_local" \
  "${CONFIGS}/figure3_logistic" \
  --jobs "${JOBS}" \
  --budget-hours "${BUDGET_HOURS}" "$@" \
  2>&1 | tee "${REPO_ROOT}/results/logs/manuscript_$(date -u +%Y%m%dT%H%M%SZ).log"
