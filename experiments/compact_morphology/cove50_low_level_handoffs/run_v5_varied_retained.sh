#!/usr/bin/env bash
set -euo pipefail

cd /root/robot-learning-lab
base=/workspace/broad-handoff-v5
body=artifacts/ape-cove50-transfer-20261008/cove50-direct-training.xml
gait=artifacts/ape-low-level-training-20261015/shared-forward-turn/native/bc/bc-best.pt
stop=artifacts/ape-cove50-recovery-20261008/ankleonly100-model844.pt
squat=artifacts/ape-low-level-training-20261015/multicommand-cpu-pilot-v1/student-squat-varied-history.pt
arms=artifacts/ape-low-level-training-20261015/multicommand-cpu-pilot-v1/student-arms-only.pt
script=experiments/compact_morphology/cove50_low_level_handoffs/collect_v5_batched_teacher.py

run_one() {
  local seed="$1" rail="$2" delay="$3" out="$4"
  test ! -e "$base/$out"
  OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 PYTHONPATH=experiments/compact_morphology/mjlab_ape/src \
    /root/cove50-venv/bin/python "$script" \
    --body "$body" --gait-actor "$gait" --stop-actor "$stop" \
    --squat-actor "$squat" --arm-actor "$arms" \
    --worlds 64 --seed "$seed" --rail-v "$rail" --delay-ms "$delay" \
    --steps 1200 --device cuda:0 --retain-whole-traces --out "$base/$out" \
    >"$base/$out.log" 2>&1
}

run_one 54520 10 0 retained-10v-0ms & p1=$!
run_one 54521 10 20 retained-10v-20ms & p2=$!
run_one 54522 12 0 retained-12v-0ms & p3=$!
run_one 54523 12 20 retained-12v-20ms & p4=$!
set +e
wait "$p1"; r1=$?
wait "$p2"; r2=$?
wait "$p3"; r3=$?
wait "$p4"; r4=$?
printf 'exit codes: %s %s %s %s\n' "$r1" "$r2" "$r3" "$r4"
test "$r1" -eq 0 && test "$r2" -eq 0 && test "$r3" -eq 0 && test "$r4" -eq 0
