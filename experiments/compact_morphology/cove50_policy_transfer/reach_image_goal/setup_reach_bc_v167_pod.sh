#!/usr/bin/env bash
set -euo pipefail

# Disposable CUDA compute. Local artifacts remain authoritative after sync.
mkdir -p /root/robot-learning-lab /workspace/reach-bc-v167
tar -xzf /root/cove50-v167-input.tgz -C /root/robot-learning-lab
python -m venv --system-site-packages /root/cove50-venv
/root/cove50-venv/bin/python -m pip install \
  -r /root/robot-learning-lab/experiments/compact_morphology/cove50_stand_recovery/runtime-inputs-v31.txt \
  >/workspace/reach-bc-v167/pip.log 2>&1
/root/cove50-venv/bin/python -m pip install --no-deps \
  'git+https://github.com/Rhoban/bam.git@62bd8ce12154340be97e06f7f41a0ca8f116d967' \
  >/workspace/reach-bc-v167/bam.log 2>&1
/root/cove50-venv/bin/python -m pip install --no-deps \
  -e /root/robot-learning-lab/microduck_rl \
  -e /root/robot-learning-lab/experiments/compact_morphology/mjlab_ape \
  >/workspace/reach-bc-v167/project.log 2>&1
/root/cove50-venv/bin/python -m pip check >/workspace/reach-bc-v167/pip-check.log 2>&1
cd /root/robot-learning-lab
/root/cove50-venv/bin/python -c 'import torch, mjlab, ape_walk.task; assert torch.cuda.is_available(); print(torch.cuda.get_device_name(0))'
