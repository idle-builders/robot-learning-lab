#!/usr/bin/env bash
set -euo pipefail

start="${1:?a or b}"
label="${2:?fresh run label}"
case "$start" in
  a) source_run=g1-cart-robustness-baseline-20260925/start-a-rep1 ;;
  b) source_run=g1-cart-robustness-baseline-20260925/start-b-rep1 ;;
  *) echo 'start must be a or b' >&2; exit 2 ;;
esac

root=/workspace/i4h-workflows
volume=/runpod-volume/robot-learning-lab/v1
out="$volume/g1-cart-approach-hil-20260925/$label"
checkpoint="${I4H_HIL_CHECKPOINT:-$volume/g1-cart-hil-full-action-20260925/hil-correction400}"
mkdir -p "$out"
cd "$root"
export OMNI_KIT_ACCEPT_EULA=YES
export VK_DRIVER_FILES=/workspace/nvidia-headless-icd.json
export VK_ICD_FILENAMES=/workspace/nvidia-headless-icd.json
export I4H_STRICT_NATIVE_SUCCESS=1
export I4H_HANDOFF_HDF5="$volume/$source_run/handoff.hdf5"
export I4H_HIL_FULL_ACTION=1
export I4H_HIL_STEP_MODE=1
export I4H_HIL_PORT=8766
export I4H_HIL_OPERATOR=assistant
export I4H_HIL_TRACE="$out/intervention.jsonl"

policy_pid=''
cleanup() {
  if [[ -n "$policy_pid" ]]; then
    kill "$policy_pid" 2>/dev/null || true
    wait "$policy_pid" 2>/dev/null || true
  fi
}
trap cleanup EXIT

workflows/agentic/policy/gr00t_n16/run.sh \
  --env locomanip_push_cart --model-path "$checkpoint" \
  >"$out/policy.log" 2>&1 &
policy_pid=$!
echo "$policy_pid" >"$out/policy.pid"

ready=0
for _ in $(seq 1 180); do
  if ! kill -0 "$policy_pid" 2>/dev/null; then
    echo 'policy process exited before readiness' >&2
    exit 1
  fi
  if python3 - <<'PY'
import json
from urllib.request import urlopen
try:
    with urlopen('http://127.0.0.1:8768/healthz', timeout=2) as response:
        state = json.load(response).get('state')
    raise SystemExit(0 if state in ('waiting_for_samples', 'ready', 'running') else 1)
except Exception:
    raise SystemExit(1)
PY
  then
    ready=1
    break
  fi
  sleep 2
done
if [[ "$ready" != 1 ]]; then
  echo 'policy did not become ready' >&2
  exit 1
fi

timeout 3600s workflows/agentic/arena/run.sh \
  --env locomanip_push_cart --episodes 1 --max-timesteps 1400 \
  --seed 100 --headless --num_envs 1 \
  --record-to "$out/rollout.hdf5" --save-all-episodes \
  >"$out/arena.log" 2>&1

sha256sum "$out/rollout.hdf5" "$out/arena.log" "$out/policy.log" "$out/intervention.jsonl" \
  >"$out/sha256.txt"
