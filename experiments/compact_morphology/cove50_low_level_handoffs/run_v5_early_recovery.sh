#!/usr/bin/env bash
set -euo pipefail
cd /root/robot-learning-lab
base=/workspace/broad-handoff-v5
script=experiments/compact_morphology/cove50_low_level_handoffs/evaluate_v4_shared_bc.py
run_one() {
  local seed="$1" rail="$2" speed="$3" switch="$4"
  test ! -e "$base/recovery-$seed.json"
  OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 PYTHONPATH=experiments/compact_morphology/mjlab_ape/src \
    /root/cove50-venv/bin/python "$script" \
    --body artifacts/ape-cove50-transfer-20261008/cove50-direct-training.xml \
    --student-actor artifacts/ape-low-level-training-20261015/broad-handoff-v4-l4-fit/shared-actor.pt \
    --gait-actor artifacts/ape-low-level-training-20261015/shared-forward-turn/native/bc/bc-best.pt \
    --stop-actor artifacts/ape-cove50-recovery-20261008/ankleonly100-model844.pt \
    --squat-actor artifacts/ape-low-level-training-20261015/multicommand-cpu-pilot-v1/student-squat-varied-history.pt \
    --arm-actor artifacts/ape-low-level-training-20261015/multicommand-cpu-pilot-v1/student-arms-only.pt \
    --teacher-recovery-step "$switch" --seed "$seed" --rail-v "$rail" --delay-ms 0 \
    --walk-velocity "$speed" --device cuda:0 --native-evaluation \
    --out "$base/recovery-$seed.json" --trace "$base/recovery-$seed.npz" \
    >"$base/recovery-$seed.log" 2>&1
}
run_one 54544 10 0.20 5 & p1=$!
run_one 54545 12 0.30 10 & p2=$!
set +e
wait "$p1"; r1=$?
wait "$p2"; r2=$?
printf 'early recovery exit codes: %s %s\n' "$r1" "$r2"
test "$r1" -eq 0 && test "$r2" -eq 0
