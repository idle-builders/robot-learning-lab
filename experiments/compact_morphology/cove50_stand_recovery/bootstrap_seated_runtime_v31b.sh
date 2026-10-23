#!/usr/bin/env bash
# Frozen V31b install on runpod/pytorch:1.0.3-cu1281-torch291-ubuntu2404.
# This prepares only the disjoint CUDA sentinel. No original or optimizer runs.
set -euo pipefail

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
venv_path="/root/cove50-v31b-venv"
pollen_source="$repo_root/.venv-control/src/microduck_rl"
inputs="$repo_root/experiments/compact_morphology/cove50_stand_recovery/runtime-inputs-v31.txt"

if [[ "$(git -C "$pollen_source" rev-parse HEAD)" != "0bf989701f14d4008c866ffd6f09067870f99379" ]]; then
  echo "Pinned Pollen source commit missing" >&2
  exit 1
fi
if [[ -n "$(git -C "$pollen_source" status --porcelain)" ]]; then
  echo "Pinned Pollen source tree is modified" >&2
  exit 1
fi
if [[ -e "$venv_path" ]]; then
  echo "V31b runtime path already exists; refusing an in-place overlay" >&2
  exit 1
fi
python3.12 -m venv --system-site-packages "$venv_path"
python_bin="$venv_path/bin/python"

# The image's Torch/Triton are requirements, not upgrade targets. These direct
# pins constrain the resolver; the later attestation rejects an overlay.
"$python_bin" -m pip install --no-input --disable-pip-version-check -r "$inputs"
"$python_bin" -m pip install --no-input --disable-pip-version-check --no-deps \
  'git+https://github.com/Rhoban/bam.git@62bd8ce12154340be97e06f7f41a0ca8f116d967'
"$python_bin" -m pip install --no-input --disable-pip-version-check --no-deps "$pollen_source"
"$python_bin" -m pip install --no-input --disable-pip-version-check --no-deps \
  "$repo_root/experiments/compact_morphology/mjlab_ape"
"$python_bin" -m pip check

echo "V31b packages installed. Run the frozen attestation, then independent live-host review, then the unscored CUDA sentinel."
