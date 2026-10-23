#!/usr/bin/env bash
set -euo pipefail

# Disposable Runpod compute only. The task's authoritative files are local.
mkdir -p /root/robot-learning-lab
tar -xzf /root/cove50-v5-input.tgz -C /root/robot-learning-lab
python -m venv --system-site-packages /root/cove50-venv
/root/cove50-venv/bin/python -m pip install \
  -r /root/robot-learning-lab/experiments/compact_morphology/cove50_stand_recovery/runtime-inputs-v31.txt \
  >/root/cove50-v5-pip.log 2>&1
/root/cove50-venv/bin/python -m pip install --no-deps \
  'git+https://github.com/Rhoban/bam.git@62bd8ce12154340be97e06f7f41a0ca8f116d967' \
  >/root/cove50-v5-bam.log 2>&1
/root/cove50-venv/bin/python -m pip install --no-deps \
  -e /root/robot-learning-lab/microduck_rl \
  -e /root/robot-learning-lab/experiments/compact_morphology/mjlab_ape \
  >/root/cove50-v5-project.log 2>&1
/root/cove50-venv/bin/python -m pip check >/root/cove50-v5-pip-check.log 2>&1
cd /root/robot-learning-lab
/root/cove50-venv/bin/python -c 'import torch, mjlab, ape_walk.task; assert torch.cuda.is_available(); print(torch.cuda.get_device_name(0))'
