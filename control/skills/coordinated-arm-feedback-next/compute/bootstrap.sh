#!/usr/bin/env bash
set -euo pipefail
ROOT="$(pwd)"
HERE="$ROOT/control/skills/coordinated-arm-feedback-next/compute"
PYTHON_BIN="${PYTHON_BIN:-python3.12}"
if ! command -v "$PYTHON_BIN" >/dev/null 2>&1; then PYTHON_BIN=python3; fi
"$PYTHON_BIN" -c 'import sys; assert sys.version_info[:2] == (3,12)'
export PIP_DISABLE_PIP_VERSION_CHECK=1 PIP_NO_INPUT=1 PIP_RETRIES=0 WANDB_MODE=offline WANDB_SILENT=true PYTHONNOUSERSITE=1 PYTHONDONTWRITEBYTECODE=1
unset PYTHONPATH PYTHONHOME
"$PYTHON_BIN" -B "$HERE/environment.py" stage-sources
test ! -e "$ROOT/.venv-worker"
"$PYTHON_BIN" -m venv "$ROOT/.venv-worker"
PY="$ROOT/.venv-worker/bin/python"
"$PY" -m pip install --only-binary=:all: 'torch==2.9.1' 'torchvision==0.24.1' --index-url https://download.pytorch.org/whl/cu128
"$PY" -m pip install --only-binary=:all: --no-deps -r "$ROOT/control/common/compute/requirements-worker.txt"
"$PY" -B "$HERE/environment.py" bind-sources
"$PY" -m pip check > "$ROOT/evidence/bootstrap/pip-check.txt" 2>&1 || { cat "$ROOT/evidence/bootstrap/pip-check.txt"; exit 1; }
"$PY" -B "$HERE/environment.py" attest
"$PY" -B "$HERE/environment.py" cuda-attest > "$ROOT/worker-attestation.json"
"$PY" -m pip freeze > "$ROOT/worker-pip-freeze.txt"
