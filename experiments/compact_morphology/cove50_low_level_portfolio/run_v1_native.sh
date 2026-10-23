#!/usr/bin/env bash
set -euo pipefail
cd /root/robot-learning-lab
base=/workspace/modular-portfolio-v1
mkdir -p "$base"
body=artifacts/ape-cove50-transfer-20261008/cove50-direct-training.xml
gait=artifacts/ape-low-level-training-20261015/shared-forward-turn/native/bc/bc-best.pt
stop=artifacts/ape-cove50-recovery-20261008/ankleonly100-model844.pt
squat=artifacts/ape-low-level-training-20261015/multicommand-cpu-pilot-v1/student-squat-varied-history.pt
arms=artifacts/ape-low-level-training-20261015/multicommand-cpu-pilot-v1/student-arms-only.pt

run_visit() {
  local kind="$1" seed="$2" rail="$3" delay="$4" speed="$5"
  local script=experiments/compact_morphology/cove50_low_level_portfolio/evaluate_portfolio.py
  if [[ "$kind" == source ]]; then
    script=experiments/compact_morphology/cove50_low_level_handoffs/evaluate_walk_squat_arms.py
  fi
  local stem="$base/$kind-$seed"
  test ! -e "$stem.json" && test ! -e "$stem.npz" && test ! -e "$stem.mp4"
  MUJOCO_GL=egl OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 PYTHONPATH=experiments/compact_morphology/mjlab_ape/src \
    /root/cove50-venv/bin/python "$script" --body "$body" \
    --gait-actor "$gait" --stop-actor "$stop" --squat-actor "$squat" --arm-actor "$arms" \
    --seed "$seed" --rail-v "$rail" --delay-ms "$delay" --walk-velocity "$speed" \
    --device cuda:0 --native-evaluation --out "$stem.json" --trace "$stem.npz" \
    --video "$stem.mp4" >"$stem.log" 2>&1
}

# Each pair is complete before the next start; the four case parameters are
# predeclared in artifacts/ape-low-level-training-20261015/modular-portfolio-v1/method.json.
run_visit source 54700 10 0 0.20
run_visit portfolio 54700 10 0 0.20
run_visit source 54701 10 20 0.25
run_visit portfolio 54701 10 20 0.25
run_visit source 54702 12 0 0.30
run_visit portfolio 54702 12 0 0.30
run_visit source 54703 12 20 0.35
run_visit portfolio 54703 12 20 0.35
