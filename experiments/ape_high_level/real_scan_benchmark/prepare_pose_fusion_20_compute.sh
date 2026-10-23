#!/usr/bin/env bash
# Disposable compute runtime. Source, frozen cohort and models arrive from the local repo.
set -euo pipefail

repo=/root/robot-learning-lab
mkdir -p "$repo/assets/real_indoor_scenes/habitat_test_scenes"
apt-get update -qq
DEBIAN_FRONTEND=noninteractive apt-get install -y -qq \
  libegl-dev libglew2.2 libgl1 libglib2.0-0 libopencv-core406t64 \
  ffmpeg zstd >/root/ape-pose-fusion-apt.log 2>&1
python -m venv --system-site-packages /root/ape-venv
/root/ape-venv/bin/python -m pip install -q \
  numpy==2.2.5 open3d==0.20.0 trimesh==5.1.1 \
  opencv-python-headless imageio imageio-ffmpeg pillow \
  transformers==4.56.2 safetensors scikit-image mujoco \
  >/root/ape-pose-fusion-pip.log 2>&1
/root/ape-venv/bin/python -m pip install -q --no-deps torchvision==0.24.1 \
  --index-url https://download.pytorch.org/whl/cu128 \
  >>/root/ape-pose-fusion-pip.log 2>&1
echo POSE_FUSION_20_COMPUTE_READY
