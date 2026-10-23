#!/usr/bin/env bash
set -euo pipefail

mkdir -p /root/robot-learning-lab
tar -xzf /root/cove50-pinch-v3-input.tgz -C /root/robot-learning-lab
tar -xzf /root/cove50-handoff-v6-extra.tgz -C /root/robot-learning-lab
# The pinch packet carried a different Microduck revision. Pin this task's
# compatible source and dereferenced upright JSON assets before task import.
rm -rf /root/robot-learning-lab/microduck_rl
tar -xzf /root/cove50-compatible-microduck.tgz -C /root/robot-learning-lab
tar -xzf /root/cove50-handoff-v6-runtime-json.tgz -C /root/robot-learning-lab
apt-get update -qq
DEBIAN_FRONTEND=noninteractive apt-get install -y -qq libegl1 libgles2 libopengl0 >/root/cove50-v6-egl.log 2>&1
python -m venv --system-site-packages /root/cove50-venv
/root/cove50-venv/bin/python -m pip install -q \
  -r /root/robot-learning-lab/experiments/compact_morphology/cove50_stand_recovery/runtime-inputs-v31.txt
/root/cove50-venv/bin/python -m pip install -q --no-deps \
  'git+https://github.com/Rhoban/bam.git@62bd8ce12154340be97e06f7f41a0ca8f116d967'
/root/cove50-venv/bin/python -m pip install -q --no-deps \
  -e /root/robot-learning-lab/microduck_rl \
  -e /root/robot-learning-lab/experiments/compact_morphology/mjlab_ape
cd /root/robot-learning-lab
MUJOCO_GL=egl /root/cove50-venv/bin/python -c \
  'import torch, mjlab, ape_walk.task; assert torch.cuda.is_available(); print(torch.cuda.get_device_name(0))'
