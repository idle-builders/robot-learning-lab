#!/usr/bin/env bash
set -euo pipefail

# Temporary Runpod compute setup. Source and artifacts remain on the local host.
mkdir -p /root/robot-learning-lab
tar --no-same-owner --warning=no-unknown-keyword -xzf /root/cove50-swing-input.tgz -C /root/robot-learning-lab
apt-get update -qq
apt-get install -y -qq libegl1 libgles2 git >/root/apt.log 2>&1
python -m venv --system-site-packages /root/cove50-venv
printf 'torch==2.9.1+cu128\ntorchvision==0.24.1\n' >/root/constraints.txt
/root/cove50-venv/bin/python -m pip install -c /root/constraints.txt \
  mjlab==1.3.0 mujoco-warp==3.8.1 warp-lang==1.12.0 \
  better-actuator-models==1.0.2 scipy colorama PyYAML >/root/pip-main.log 2>&1
git clone -q https://github.com/pollen-robotics/microduck_rl.git /root/microduck_rl
git -C /root/microduck_rl checkout -q 4757d6aed51ff059232a66cdb8a4ee3a77fa2ce8
git -C /root/microduck_rl apply \
  /root/robot-learning-lab/experiments/compact_morphology/mjlab_ape/pollen-bam-compat.patch
/root/cove50-venv/bin/python -m pip install --no-deps -e /root/microduck_rl \
  -e /root/robot-learning-lab/experiments/compact_morphology/mjlab_ape >/root/pip-project.log 2>&1
cd /root/robot-learning-lab
/root/cove50-venv/bin/python -c 'import ape_walk.task, mjlab_microduck; print("imports OK")'
