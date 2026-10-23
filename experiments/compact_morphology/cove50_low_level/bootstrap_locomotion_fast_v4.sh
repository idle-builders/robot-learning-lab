#!/usr/bin/env bash
# Install one V4 worker from pinned source; old V3e one-shot contract is not used.
set -euo pipefail
repo_root="/root/robot-learning-lab"
venv_path="/root/cove50-v4-venv"
check="$repo_root/experiments/compact_morphology/cove50_low_level/check_locomotion_fast_v4.py"
inputs="$repo_root/experiments/compact_morphology/cove50_stand_recovery/runtime-inputs-v31.txt"
pollen_source="$repo_root/.venv-control/src/microduck_rl"
export PYTHONPATH="$repo_root/experiments/compact_morphology/mjlab_ape/src:$repo_root"
if [[ -e "$venv_path" ]]; then
  echo "V4 runtime path already exists; refusing overlay" >&2
  exit 1
fi
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
