#!/usr/bin/env bash
# Isolated worker venv: pinned CUDA torch plus the R3 walking worker closure
# (requirements-worker.txt = forward-urdf-retry closure, sha256 6132ca53...;
# mjlab 1.3.0, mujoco 3.13.0, mujoco-warp 3.8.1, warp 1.12.0, rsl-rl 5.0.1).
# No task plugins are installed, so mjlab runs unpatched.
set -euo pipefail
ROOT="$(pwd)"
HERE="$ROOT/control/wbc/compute"
PYTHON_BIN="${PYTHON_BIN:-python3.12}"
command -v "$PYTHON_BIN" >/dev/null 2>&1 || PYTHON_BIN=python3
"$PYTHON_BIN" -c 'import sys; assert sys.version_info[:2] == (3, 12)'
export PIP_DISABLE_PIP_VERSION_CHECK=1 PIP_NO_INPUT=1 PIP_RETRIES=0 PYTHONNOUSERSITE=1 PYTHONDONTWRITEBYTECODE=1
unset PYTHONPATH PYTHONHOME
test ! -e "$ROOT/.venv-worker"
"$PYTHON_BIN" -m venv "$ROOT/.venv-worker"
PY="$ROOT/.venv-worker/bin/python"
"$PY" -m pip install --only-binary=:all: 'torch==2.9.1' 'torchvision==0.24.1' --index-url https://download.pytorch.org/whl/cu128
"$PY" -m pip install --only-binary=:all: --no-deps -r "$HERE/requirements-worker.txt"
"$PY" -m pip check
"$PY" -m pip freeze > "$ROOT/evidence/bootstrap/pip-freeze.txt"
"$PY" -c 'import torch, mjlab, mujoco_warp; assert torch.cuda.is_available(); print(torch.cuda.get_device_name(0))'
