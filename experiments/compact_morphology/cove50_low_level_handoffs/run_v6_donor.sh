#!/usr/bin/env bash
set -euo pipefail
cd /root/robot-learning-lab
base=/workspace/broad-handoff-v5/v6-donor
mkdir -p "$base"
test ! -e "$base/seed-54301.json"
test ! -e "$base/seed-54301.npz"
python - <<'PY'
import hashlib, json, pathlib
root = pathlib.Path('/root/robot-learning-lab')
method = json.loads((root / 'artifacts/ape-low-level-training-20261015/broad-handoff-v6/method.json').read_text())
for rel, expected in method['paths'].items():
    if rel.endswith('/seed-54301-rail-10-delay-0.json'):
        continue  # local source original; the matched training donor remains saved locally
    actual = hashlib.sha256((root / rel).read_bytes()).hexdigest()
    if actual != expected:
        raise RuntimeError(f'Frozen V6 source changed: {rel}')
PY
MUJOCO_GL=egl OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 \
PYTHONPATH=experiments/compact_morphology/mjlab_ape/src \
timeout 900 /root/cove50-venv/bin/python \
  experiments/compact_morphology/cove50_low_level_handoffs/evaluate_v4_shared_bc.py \
  --body artifacts/ape-cove50-transfer-20261008/cove50-direct-training.xml \
  --student-actor artifacts/ape-low-level-training-20261015/broad-handoff-v6/shared-actor.pt \
  --seed 54301 --rail-v 10 --delay-ms 0 --walk-velocity 0.2 \
  --device cuda:0 --native-evaluation \
  --out "$base/seed-54301.json" --trace "$base/seed-54301.npz"
