#!/usr/bin/env bash
set -euo pipefail

# Run on the provisioned Runpod GPU. Keep outputs on its ephemeral disk, then
# copy the JSON/video evidence back locally before removing the pod.
cd /workspace/robot-learning-lab/experiments/compact_morphology/mjlab_ape
export PYTHONPATH=/workspace/robot-learning-lab/experiments/compact_morphology/mjlab_ape/src:/workspace/microduck_rl/src
ckpt=/workspace/robot-learning-lab/microduck_rl/logs/rsl_rl/ape_velocity/2026-09-28_22-02-40_ape_dt2_from_scratch/model_499.pt
for model in boot_geometry_locked boot_cad_mass; do
  for seed in 143 144 145 146; do
    /tmp/ape-mjlab-venv/bin/python evaluate.py "$ckpt" \
      --task Mjlab-Velocity-Flat-Ape-Elbows --legacy-policy-on-elbow \
      --worlds 1 --seed "$seed" --speed 0.35 --physics-dt 0.002 \
      --body-xml "/tmp/ape-boot-models/$model.xml" \
      --out "/tmp/$model-$seed.json" >"/tmp/$model-$seed.log" 2>&1
    python3 - "$model" "$seed" <<'PY'
import json, sys
model, seed = sys.argv[1:]
w = json.load(open(f"/tmp/{model}-{seed}.json"))["worlds"][0]
print(model, seed, "duration", w["duration_s"], "forward", w["forward_m"],
      "lateral", w["lateral_m"], "hand force", w["hand_force_n"], flush=True)
PY
  done
done
