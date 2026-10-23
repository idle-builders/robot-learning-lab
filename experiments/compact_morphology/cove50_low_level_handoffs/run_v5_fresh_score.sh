#!/usr/bin/env bash
set -euo pipefail
cd /root/robot-learning-lab
base=/workspace/broad-handoff-v5/score
mkdir -p "$base"
body=artifacts/ape-cove50-transfer-20261008/cove50-direct-training.xml
student=artifacts/ape-low-level-training-20261015/broad-handoff-v5/shared-actor.pt
gait=artifacts/ape-low-level-training-20261015/shared-forward-turn/native/bc/bc-best.pt
stop=artifacts/ape-cove50-recovery-20261008/ankleonly100-model844.pt
squat=artifacts/ape-low-level-training-20261015/multicommand-cpu-pilot-v1/student-squat-varied-history.pt
arms=artifacts/ape-low-level-training-20261015/multicommand-cpu-pilot-v1/student-arms-only.pt
teacher_script=experiments/compact_morphology/cove50_low_level_handoffs/evaluate_walk_squat_arms.py
student_script=experiments/compact_morphology/cove50_low_level_handoffs/evaluate_v4_shared_bc.py

run_pair() {
  local seed="$1" rail="$2" delay="$3" speed="$4"
  test ! -e "$base/teacher-$seed.json" && test ! -e "$base/student-$seed.json"
  OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 PYTHONPATH=experiments/compact_morphology/mjlab_ape/src \
    /root/cove50-venv/bin/python "$teacher_script" --body "$body" \
    --gait-actor "$gait" --stop-actor "$stop" --squat-actor "$squat" --arm-actor "$arms" \
    --seed "$seed" --rail-v "$rail" --delay-ms "$delay" --walk-velocity "$speed" \
    --device cuda:0 --native-evaluation --out "$base/teacher-$seed.json" \
    --trace "$base/teacher-$seed.npz" >"$base/teacher-$seed.log" 2>&1 & local tp=$!
  OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 PYTHONPATH=experiments/compact_morphology/mjlab_ape/src \
    /root/cove50-venv/bin/python "$student_script" --body "$body" --student-actor "$student" \
    --seed "$seed" --rail-v "$rail" --delay-ms "$delay" --walk-velocity "$speed" \
    --device cuda:0 --native-evaluation --out "$base/student-$seed.json" \
    --trace "$base/student-$seed.npz" >"$base/student-$seed.log" 2>&1 & local sp=$!
  wait "$tp" && wait "$sp"
}

run_pair 54530 10 0 0.20 & p1=$!
run_pair 54531 10 20 0.25 & p2=$!
run_pair 54532 12 0 0.30 & p3=$!
run_pair 54533 12 20 0.35 & p4=$!
set +e
wait "$p1"; r1=$?
wait "$p2"; r2=$?
wait "$p3"; r3=$?
wait "$p4"; r4=$?
printf 'pair exit codes: %s %s %s %s\n' "$r1" "$r2" "$r3" "$r4"
test "$r1" -eq 0 && test "$r2" -eq 0 && test "$r3" -eq 0 && test "$r4" -eq 0
