#!/usr/bin/env bash
set -euo pipefail

# Run on a Runpod NVIDIA GPU. The simulator source may be on a network volume
# or on the pod disk when the original GPU class is temporarily unavailable.
root=${PLAYROOM_ROOT:-/runpod-volume/robot-learning-lab/berkeley-playroom-isaac51}
mkdir -p "$root"
apt-get update -qq
DEBIAN_FRONTEND=noninteractive apt-get install -y -qq libxt6 libglu1-mesa libxrender1 libxext6 libsm6 libxi6 libxrandr2 libxfixes3 libxkbcommon0 vulkan-tools
# The stock manifest points to GLX; Runpod headless hosts need NVIDIA's EGL ICD.
printf '{"file_format_version":"1.0.0","ICD":{"library_path":"libEGL_nvidia.so.0","api_version":"1.3.0"}}\n' >/workspace/nvidia-headless-icd.json
export VK_ICD_FILENAMES=/workspace/nvidia-headless-icd.json
export VK_DRIVER_FILES=/workspace/nvidia-headless-icd.json
vulkaninfo --summary >/tmp/berkeley-vulkaninfo.txt
# Many tiny Python files are slow on Runpod's network volume. Keep the active
# interpreter and environment on the pod disk; persist data/checkpoints there.
export UV_CACHE_DIR=/workspace/uv-cache
export UV_PYTHON_INSTALL_DIR=/workspace/uv-python
export UV_PROJECT_ENVIRONMENT=/workspace/berkeley-venv
# Upstream records public submodules with git@ URLs; this pod has no GitHub SSH key.
git config --global url.https://github.com/.insteadOf git@github.com:

if [ ! -d "$root/Berkeley-Humanoid-Lite/.git" ]; then
  git clone --recurse-submodules https://github.com/HybridRobotics/Berkeley-Humanoid-Lite.git "$root/Berkeley-Humanoid-Lite"
fi
cd "$root/Berkeley-Humanoid-Lite"
git fetch origin 984741a3623c93b0583ccfdc479f1f8b1c4d900e
git checkout --detach 984741a3623c93b0583ccfdc479f1f8b1c4d900e
git submodule update --init --recursive

uv python install 3.11
uv sync --frozen --no-dev
if [ ! -d "$root/IsaacLab-Arena-0.1.1/.git" ]; then
  git clone --depth 1 --branch release/0.1.1 https://github.com/isaac-sim/IsaacLab-Arena.git "$root/IsaacLab-Arena-0.1.1"
fi
uv pip install --python "$UV_PROJECT_ENVIRONMENT/bin/python" --no-deps -e "$root/IsaacLab-Arena-0.1.1"
uv pip install --python "$UV_PROJECT_ENVIRONMENT/bin/python" lightwheel-sdk
printf 'READY %s\n' "$(git rev-parse HEAD)"
