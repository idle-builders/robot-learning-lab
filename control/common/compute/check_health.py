"""Read-only Runpod/model inventory. No retries, provisioning or credential output.

Run: .venv/bin/python -B control/common/compute/check_health.py --out NEW.json
Use --observed RECEIPT.json to package already captured CLI observations offline.
"""
import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import subprocess

ROOT = Path(__file__).resolve().parents[3]


def capture(command):
    try:
        result = subprocess.run(command, capture_output=True, text=True, timeout=45)
    except (OSError, subprocess.TimeoutExpired) as error:
        return {"command": command, "exit_code": None, "response": {"error_type": type(error).__name__}}
    raw = result.stdout.strip() or result.stderr.strip()
    try:
        value = json.loads(raw)
    except json.JSONDecodeError:
        value = {"message": raw[:2000]}
    if command == ["runpodctl", "user"] and result.returncode == 0:
        value = {k: value.get(k) for k in ["clientBalance", "currentSpendPerHr", "spendLimit"]}
    if result.returncode == 0 and command[1:] in (["pod", "list"], ["network-volume", "list"]):
        # Inventory is evidence of resource state; do not retain environment,
        # authentication or connection payloads if CLI output grows later.
        allowed = {"id", "name", "runtimeStatus", "desiredStatus", "costPerHr", "gpuId",
                   "gpuCount", "dataCenterId", "size", "createdAt", "pods", "networkVolumes"}
        def inventory(item):
            if isinstance(item, list): return [inventory(x) for x in item]
            if isinstance(item, dict): return {k: inventory(v) for k, v in item.items() if k in allowed}
            return item
        value = inventory(value)
    return {"command": command, "exit_code": result.returncode, "response": value}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--observed", type=Path)
    args = parser.parse_args()
    pointer = ROOT / "robot/model/current-model.json"
    selected = json.loads(pointer.read_text())
    inputs = {}
    for key in ["rigid_URDF", "rigid_MJCF", "actuators", "sensors", "timing"]:
        item = selected[key]
        path = ROOT / item["path"]
        digest = hashlib.file_digest(path.open("rb"), "sha256").hexdigest() if path.is_file() else None
        inputs[key] = {"path": item["path"], "expected_sha256": item["sha256"], "actual_sha256": digest}
    if args.observed:
        observations = json.loads(args.observed.read_text())
    else:
        observations = {"capture_kind": "live_read_only_cli", "checks": [capture(command) for command in [
            ["runpodctl", "version"], ["runpodctl", "user"],
            ["runpodctl", "pod", "list"], ["runpodctl", "network-volume", "list"]]]}
    report = {"schema": "cove50.forward-urdf-retry-compute.v1",
              "recorded_at_utc": datetime.now(timezone.utc).isoformat(),
              "selected_model_version": selected["version"],
              "selected_model_pointer_sha256": hashlib.sha256(pointer.read_bytes()).hexdigest(),
              "selected_inputs": inputs, "observations": observations,
              "new_paid_resources": 0, "new_training_transitions": 0, "new_optimizer_updates": 0,
              "limitation": "Account authentication is separate from pod inventory and actual GPU task readiness. A daily-window 429 is not a policy or robot outcome."}
    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open("x") as stream:
        json.dump(report, stream, indent=2)
        stream.write("\n")
    print(args.out)


if __name__ == "__main__":
    main()
