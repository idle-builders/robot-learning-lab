#!/usr/bin/env bash
set -euo pipefail

# Run on the provisioned Runpod GPU. Arguments: checkpoint, output prefix.
checkpoint=$1
label=$2
cd /workspace/robot-learning-lab/experiments/compact_morphology/mjlab_ape
export PYTHONPATH=/workspace/robot-learning-lab/experiments/compact_morphology/mjlab_ape/src:/workspace/microduck_rl/src
for seed in 142 143 144 145 146; do
  /tmp/ape-mjlab-venv/bin/python evaluate.py "$checkpoint" \
    --task Mjlab-Velocity-Flat-Ape-Elbows --worlds 1 --seed "$seed" \
    --speed 0.35 --physics-dt 0.002 \
    --body-xml /tmp/ape-boot-models/boot_cad_mass.xml \
    --out "/tmp/$label-$seed.json" --state-out "/tmp/$label-$seed.npz" \
    >"/tmp/$label-$seed.log" 2>&1
  python3 - "$label" "$seed" <<'PY'
import json, sys
label, seed = sys.argv[1:]
w = json.load(open(f"/tmp/{label}-{seed}.json"))["worlds"][0]
print(label, seed, "duration", w["duration_s"], "forward", w["forward_m"],
      "lateral", w["lateral_m"], "yaw", w["final_yaw_rad"],
      "hand N", w["hand_force_n"], flush=True)
PY
done
