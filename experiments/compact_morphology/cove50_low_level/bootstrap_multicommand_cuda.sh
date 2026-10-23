#!/usr/bin/env bash
# Pinned A40 runtime for the bounded Direct Frame multi-command CUDA pilot.
set -euo pipefail
repo_root=/root/robot-learning-lab
venv=/root/cove50-multicommand-venv
export MUJOCO_GL=egl
export NVIDIA_DRIVER_CAPABILITIES=compute,utility,graphics
if [[ -e "$venv" ]]; then
  echo "Refusing to overlay an existing CUDA runtime" >&2
  exit 1
fi
timeout 300 apt-get update -qq
timeout 300 apt-get install -y --no-install-recommends \
  libegl1=1.7.0-1build1 libopengl0=1.7.0-1build1
python3.12 -m venv --system-site-packages "$venv"
py="$venv/bin/python"
"$py" -m pip install --no-input --disable-pip-version-check \
  -r "$repo_root/experiments/compact_morphology/cove50_stand_recovery/runtime-inputs-v31.txt"
"$py" -m pip install --no-input --disable-pip-version-check --no-deps \
  'git+https://github.com/Rhoban/bam.git@62bd8ce12154340be97e06f7f41a0ca8f116d967'
"$py" -m pip install --no-input --disable-pip-version-check --no-deps \
  "$repo_root/.venv-control/src/microduck_rl"
"$py" -m pip install --no-input --disable-pip-version-check --no-deps \
  "$repo_root/experiments/compact_morphology/mjlab_ape"
"$py" -m pip check
"$py" - <<'PY'
import importlib.metadata as m
import ctypes
import torch
for name in ('warp-lang','mjlab','rsl-rl-lib','mujoco','mujoco-warp'):
    print(name, m.version(name))
import bam
print('bam module', bam.__file__)
ctypes.CDLL('libEGL.so.1')
print('torch', torch.__version__, 'cuda', torch.cuda.is_available())
assert torch.__version__ == '2.9.1+cu128' and torch.cuda.is_available()
PY
