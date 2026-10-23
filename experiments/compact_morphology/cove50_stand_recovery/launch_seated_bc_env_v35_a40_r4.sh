#!/usr/bin/env bash
# One-shot, source-bound worker sequence. Launch only after the local provider
# guard and relay are live; all outputs are placed on the Pod's persistent volume.
set -euo pipefail

repo=/root/robot-learning-lab
art="$repo/artifacts/ape-cove50-stand-recovery/seated-bc-eval-v35-a40-r4"
saved=/workspace/cove50-seated-bc-eval-v35-a40-r4
source_dir="$repo/experiments/compact_morphology/cove50_stand_recovery"
python=/root/cove50-v31b-venv/bin/python
gate="$source_dir/checkout_import_gate_v35_a40_r4.py"
required_env=(
  MUJOCO_GL=egl
  NVIDIA_DRIVER_CAPABILITIES=compute,utility,graphics
  PYTHONPATH="$repo/experiments/compact_morphology/mjlab_ape/src:$repo"
)

if [[ ! -d "$repo" || ! -d "$art" || -L "$art" ]]; then
  echo "Fresh extracted A40-r4 worker checkout is required" >&2
  exit 1
fi
if ! mountpoint -q /workspace || [[ -e "$saved" ]]; then
  echo "A fresh mounted /workspace persistent volume is required" >&2
  exit 1
fi
mv "$art" "$saved"
ln -s "$saved" "$art"

run_logged() {
  local stem="$1"
  shift
  local log="$art/worker-$stem.log"
  local status="$art/worker-$stem-exit-status.json"
  if [[ -e "$log" || -e "$status" ]]; then
    echo "A40-r4 stage $stem was already attempted" >&2
    exit 1
  fi
  printf 'stage=%s\n' "$stem" > "$log"
  printf 'command=' >> "$log"
  printf '%q ' "$@" >> "$log"
  printf '\n' >> "$log"
  set +e
  env "${required_env[@]}" "$@" 2>&1 | tee -a "$log"
  local results=("${PIPESTATUS[@]}")
  set -e
  printf '{"stage":"%s","command_exit_code":%d,"tee_exit_code":%d}\n' \
    "$stem" "${results[0]}" "${results[1]}" > "$status"
  sync "$log" "$status"
  if (( results[0] != 0 || results[1] != 0 )); then
    exit 1
  fi
}

run_logged bootstrap bash "$source_dir/bootstrap_seated_runtime_v35_a40_r4.sh"
run_logged os-egl-check python3.12 "$source_dir/preflight_seated_os_egl_v35_a40_r4.py" check
run_logged checkout-attest "$python" "$gate" attest
run_logged checkout-runtime "$python" "$gate" runtime

attestation="$art/fresh-pod-attestation.json"
state="$art/fresh-pod-create-guard.json"
run_logged guard-relay-ready bash -c '
  for (( attempt=0; attempt<60; attempt++ )); do
    [[ -s "$1" && -s "$2" ]] && exit 0
    sleep 2
  done
  echo "Ordered, relayed provider guard evidence did not arrive" >&2
  exit 1
' bash "$attestation" "$state"
request_at=$(env "${required_env[@]}" "$python" -c \
  'import json,sys; print(json.load(open(sys.argv[1]))["local_start_request_at_utc"])' \
  "$attestation")
run_logged runner "$python" "$gate" runner \
  --guard-attestation "$attestation" --guard-state "$state" \
  --local-start-request-at-utc "$request_at"
