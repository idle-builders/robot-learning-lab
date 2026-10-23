#!/usr/bin/env bash
# One-use Ubuntu 24.04 A40-r4 environment readiness bootstrap. No score or fit.
set -euo pipefail

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
preflight="$repo_root/experiments/compact_morphology/cove50_stand_recovery/preflight_seated_os_egl_v35_a40_r4.py"

if [[ "${MUJOCO_GL:-}" != "egl" || ",${NVIDIA_DRIVER_CAPABILITIES:-}," != *",graphics,"* ]]; then
  echo "Pinned EGL and NVIDIA graphics environment missing before apt/Python" >&2
  exit 1
fi
if [[ -e "$repo_root/artifacts/ape-cove50-stand-recovery/seated-bc-eval-v35-a40-r4/os-egl-loader-attestation.json" ]]; then
  echo "First OS loader attestation already exists; refusing overlay" >&2
  exit 1
fi

export DEBIAN_FRONTEND=noninteractive
timeout 300 apt-get update -qq
timeout 300 apt-get install -y --no-install-recommends \
  libegl1=1.7.0-1build1 libopengl0=1.7.0-1build1

# stdlib-only OS/loader proof happens before installing or importing any policy runtime.
python3.12 "$preflight" write

# Reuse the exact frozen 227-distribution V31b Python runtime recipe.
bash "$repo_root/experiments/compact_morphology/cove50_stand_recovery/bootstrap_seated_runtime_v31b.sh"

echo "A40-r4 OS loader and Python runtime installed; run checkout attest/runtime, then bounded environment-only visits."
