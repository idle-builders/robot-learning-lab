#!/usr/bin/env bash
# Ephemeral Runpod runtime; official archives and built SDK remain on the volume.
set -euo pipefail

repo=/root/robot-learning-lab
cache=/workspace/robot-learning-lab/replica-source-cache-20261015
mkdir -p "$repo/assets/real_indoor_scenes/habitat_test_scenes"
apt-get update -qq
DEBIAN_FRONTEND=noninteractive apt-get install -y -qq \
  libegl-dev libglew2.2 libgl1 libglib2.0-0 libopencv-core406t64 \
  zstd ffmpeg >/root/ape-policy-apt.log 2>&1
python -m venv --system-site-packages /root/ape-venv
/root/ape-venv/bin/python -m pip install -q \
  numpy==2.2.5 open3d==0.20.0 trimesh==5.1.1 \
  opencv-python-headless imageio imageio-ffmpeg pillow \
  transformers==4.56.2 safetensors scikit-image mujoco \
  >/root/ape-policy-pip.log 2>&1
mkdir -p /root/replica-source
if [[ ! -e /root/replica-sdk-src ]]; then
  ln -s "$cache/sdk-built" /root/replica-sdk-src
fi
for scene in "$@"; do
  if [[ ! -f /root/replica-source/$scene/mesh.ply ]]; then
    tar --zstd -xf "$cache/$scene.tar.zst" -C /root/replica-source
  fi
  if [[ ! -f /root/replica-source/$scene/mesh_yup.ply ]]; then
    /root/ape-venv/bin/python \
      "$repo/experiments/ape_high_level/real_scan_benchmark/transform_replica_mesh_to_habitat.py" \
      "/root/replica-source/$scene/mesh.ply" \
      "/root/replica-source/$scene/mesh_yup.ply"
  fi
done
echo REPLICA_POLICY_COMPUTE_READY
