#!/usr/bin/env bash
set -euo pipefail

mkdir -p /root/robot-learning-lab /workspace/native-pinch-v3
tar -xzf /root/cove50-pinch-v3-input.tgz -C /root/robot-learning-lab
apt-get update -qq > /workspace/native-pinch-v3/apt-update.log 2>&1
DEBIAN_FRONTEND=noninteractive apt-get install -y -qq libegl1 libgles2 \
  > /workspace/native-pinch-v3/apt-install.log 2>&1
python -m venv --system-site-packages /root/cove50-venv
/root/cove50-venv/bin/python -m pip install \
  -r /root/robot-learning-lab/experiments/compact_morphology/cove50_stand_recovery/runtime-inputs-v31.txt \
  > /workspace/native-pinch-v3/pip.log 2>&1
/root/cove50-venv/bin/python -m pip install --no-deps \
  'git+https://github.com/Rhoban/bam.git@62bd8ce12154340be97e06f7f41a0ca8f116d967' \
  > /workspace/native-pinch-v3/bam.log 2>&1
/root/cove50-venv/bin/python -m pip install --no-deps \
  -e /root/robot-learning-lab/microduck_rl \
  -e /root/robot-learning-lab/experiments/compact_morphology/mjlab_ape \
  > /workspace/native-pinch-v3/project.log 2>&1
/root/cove50-venv/bin/python -m pip check > /workspace/native-pinch-v3/pip-check.log 2>&1
cd /root/robot-learning-lab
/root/cove50-venv/bin/python -c 'import torch, mjlab, ape_walk.task; assert torch.cuda.is_available(); print(torch.cuda.get_device_name(0))'
