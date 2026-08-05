#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
PYTHON_BIN=${PYTHON_BIN:-python3}

"${PYTHON_BIN}" -m venv "${REPO_ROOT}/.venv"
"${REPO_ROOT}/.venv/bin/python" -m pip install --upgrade pip setuptools wheel
"${REPO_ROOT}/.venv/bin/python" -m pip install -e "${REPO_ROOT}[dev]"
"${REPO_ROOT}/.venv/bin/python" -m pip check

echo "Environment ready: ${REPO_ROOT}/.venv"

