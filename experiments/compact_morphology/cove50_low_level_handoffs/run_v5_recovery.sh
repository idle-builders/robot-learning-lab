#!/usr/bin/env bash
set -euo pipefail
cd /root/robot-learning-lab
base=/workspace/broad-handoff-v5
script=experiments/compact_morphology/cove50_low_level_handoffs/evaluate_v4_shared_bc.py
body=artifacts/ape-cove50-transfer-20261008/cove50-direct-training.xml
student=artifacts/ape-low-level-training-20261015/broad-handoff-v4-l4-fit/shared-actor.pt
gait=artifacts/ape-low-level-training-20261015/shared-forward-turn/native/bc/bc-best.pt
stop=artifacts/ape-cove50-recovery-20261008/ankleonly100-model844.pt
squat=artifacts/ape-low-level-training-20261015/multicommand-cpu-pilot-v1/student-squat-varied-history.pt
arms=artifacts/ape-low-level-training-20261015/multicommand-cpu-pilot-v1/student-arms-only.pt

run_one() {
  local seed="$1" rail="$2" delay="$3" speed="$4" switch="$5" label="$6"
  test ! -e "$base/$label.json"
  OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 PYTHONPATH=experiments/compact_morphology/mjlab_ape/src \
    /root/cove50-venv/bin/python "$script" --body "$body" --student-actor "$student" \
    --gait-actor "$gait" --stop-actor "$stop" --squat-actor "$squat" --arm-actor "$arms" \
    --teacher-recovery-step "$switch" --seed "$seed" --rail-v "$rail" --delay-ms "$delay" \
    --walk-velocity "$speed" --device cuda:0 --native-evaluation \
    --out "$base/$label.json" --trace "$base/$label.npz" \
    >"$base/$label.log" 2>&1
}

test ! -e "$base/recovery-sentinel.json"
OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 PYTHONPATH=experiments/compact_morphology/mjlab_ape/src \
  /root/cove50-venv/bin/python "$script" --body "$body" --student-actor "$student" \
  --gait-actor "$gait" --stop-actor "$stop" --squat-actor "$squat" --arm-actor "$arms" \
  --teacher-recovery-step 1 --seed 54539 --rail-v 12 --delay-ms 0 \
  --walk-velocity 0.25 --max-steps 2 --device cuda:0 --native-evaluation \
  --out "$base/recovery-sentinel.json" --trace "$base/recovery-sentinel.npz" \
  >"$base/recovery-sentinel.log" 2>&1

run_one 54540 10 0 0.20 25 recovery-54540 & p1=$!
run_one 54541 10 20 0.25 200 recovery-54541 & p2=$!
run_one 54542 12 0 0.30 300 recovery-54542 & p3=$!
run_one 54543 12 20 0.35 800 recovery-54543 & p4=$!
set +e
wait "$p1"; r1=$?
wait "$p2"; r2=$?
wait "$p3"; r3=$?
wait "$p4"; r4=$?
printf 'exit codes: %s %s %s %s\n' "$r1" "$r2" "$r3" "$r4"
test "$r1" -eq 0 && test "$r2" -eq 0 && test "$r3" -eq 0 && test "$r4" -eq 0
