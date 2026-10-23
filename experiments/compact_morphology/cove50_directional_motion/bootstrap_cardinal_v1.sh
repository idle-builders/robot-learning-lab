#!/usr/bin/env bash
set -euo pipefail
repo_root="/root/robot-learning-lab"
venv_path="/root/cove50-cardinal-v1-venv"
check="$repo_root/experiments/compact_morphology/cove50_directional_motion/check_cardinal_v1.py"
inputs="$repo_root/experiments/compact_morphology/cove50_stand_recovery/runtime-inputs-v31.txt"
pollen_source="$repo_root/.venv-control/src/microduck_rl"
export PYTHONPATH="$repo_root/experiments/compact_morphology/mjlab_ape/src:$repo_root"
export MUJOCO_GL=egl
if [[ -e "$venv_path" ]]; then
  echo "Fresh cardinal runtime required" >&2
  exit 1
fi
if [[ ",${NVIDIA_DRIVER_CAPABILITIES:-}," != *",graphics,"* ]]; then
  echo "NVIDIA graphics capability missing" >&2
  exit 1
fi
export DEBIAN_FRONTEND=noninteractive
timeout 300 apt-get update -qq
timeout 300 apt-get install -y --no-install-recommends \
  libegl1=1.7.0-1build1 libopengl0=1.7.0-1build1
python3.12 - <<'PY'
import ctypes
import json
import subprocess
from pathlib import Path
for package in ('libegl1', 'libopengl0'):
    result = subprocess.check_output(('dpkg-query', '-W', '-f=${Status} ${Version}', package), text=True).strip()
    if result != 'install ok installed 1.7.0-1build1':
        raise RuntimeError(f'OS GLVND package differs: {package}: {result}')
for library in ('libEGL.so.1', 'libOpenGL.so.0', 'libEGL_nvidia.so.0'):
    ctypes.CDLL(library)
vendor = Path('/usr/share/glvnd/egl_vendor.d/10_nvidia.json')
if json.loads(vendor.read_text())['ICD']['library_path'] != 'libEGL_nvidia.so.0':
    raise RuntimeError('NVIDIA EGL vendor hook differs')
print('Ubuntu GLVND and NVIDIA EGL loaders attested before MuJoCo import')
PY
python3.12 "$check"
python3.12 -m venv --system-site-packages "$venv_path"
python_bin="$venv_path/bin/python"
"$python_bin" -m pip install --no-input --disable-pip-version-check -r "$inputs"
"$python_bin" -m pip install --no-input --disable-pip-version-check --no-deps \
  'git+https://github.com/Rhoban/bam.git@62bd8ce12154340be97e06f7f41a0ca8f116d967'
"$python_bin" -m pip install --no-input --disable-pip-version-check --no-deps "$pollen_source"
"$python_bin" -m pip install --no-input --disable-pip-version-check --no-deps \
  "$repo_root/experiments/compact_morphology/mjlab_ape"
"$python_bin" -m pip check
"$python_bin" "$check" --runtime
