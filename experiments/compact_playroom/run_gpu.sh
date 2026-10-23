#!/usr/bin/env bash
set -euo pipefail
export OMNI_KIT_ACCEPT_EULA=YES
export VK_ICD_FILENAMES=/workspace/nvidia-headless-icd.json
export VK_DRIVER_FILES=/workspace/nvidia-headless-icd.json
cd "${PLAYROOM_ROOT:-/runpod-volume/robot-learning-lab/berkeley-playroom-isaac51}/Berkeley-Humanoid-Lite"
exec /workspace/berkeley-venv/bin/python "$@"
