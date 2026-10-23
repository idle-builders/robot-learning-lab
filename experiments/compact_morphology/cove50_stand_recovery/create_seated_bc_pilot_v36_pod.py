"""Create one bounded A40 pilot pod and arm an independent local stop timer."""

from __future__ import annotations

import argparse
from datetime import datetime, timedelta, timezone
import json
import os
from pathlib import Path
import subprocess
import sys
import time
import uuid


ROOT = Path(__file__).resolve().parents[3]
ART = ROOT / "artifacts/ape-cove50-stand-recovery/seated-bc-pilot-v36"
PLAN = ART / "plan.json"
KEY = Path.home() / ".runpod/ssh/runpodctl-ssh-key"


def cli(*parts: str, timeout: int = 45) -> dict | list:
    result = subprocess.run(["runpodctl", *parts], text=True, capture_output=True,
                            timeout=timeout, check=False)
    if result.returncode:
        raise RuntimeError(f"runpodctl {parts[:2]} exited {result.returncode}")
    return json.loads(result.stdout)


def write(path: Path, payload: dict) -> None:
    temporary = path.with_name(path.name + ".incoming")
    with temporary.open("x") as output:
        json.dump(payload, output, indent=2, sort_keys=True)
        output.write("\n")
        output.flush()
        os.fsync(output.fileno())
    temporary.replace(path)


def create() -> None:
    plan = json.loads(PLAN.read_text())
    if (plan["gpu"] != "NVIDIA A40" or plan["max_pod_wall_s"] != 3600
            or plan["max_listed_gpu_rental_usd"] != 0.59
            or not KEY.is_file() or not KEY.with_suffix(".pub").is_file()
            or (ART / "pod-create.json").exists()):
        raise RuntimeError("V36 create contract or one-attempt marker differs")
    user = cli("user")
    pods = cli("pod", "list")
    if user["clientBalance"] < 2 or any("seated-bc-pilot-v36" in p["name"] for p in pods):
        raise RuntimeError("Insufficient balance or existing active V36 pod")
    requested = datetime.now(timezone.utc)
    name = "cove50-seated-bc-pilot-v36-" + uuid.uuid4().hex[:12]
    marker = {"schema": "robot-learning-lab.seated-bc-pilot-v36.create.v1",
              "name": name, "requested_at_utc": requested.isoformat(),
              "hard_stop_at_utc": (requested + timedelta(seconds=3600)).isoformat(),
              "status": "attempt_consumed_before_provider_create"}
    ART.mkdir(parents=True, exist_ok=True)
    with (ART / "pod-create.json").open("x") as output:
        json.dump(marker, output, indent=2, sort_keys=True)
        output.write("\n")
        output.flush()
        os.fsync(output.fileno())
    environment = {"MUJOCO_GL": "egl",
                   "NVIDIA_DRIVER_CAPABILITIES": "compute,utility,graphics",
                   "PUBLIC_KEY": KEY.with_suffix(".pub").read_text().strip()}
    try:
        created = cli("pod", "create", "--image",
                      "runpod/pytorch:1.0.3-cu1281-torch291-ubuntu2404",
                      "--gpu-id", "NVIDIA A40", "--cloud-type", "SECURE",
                      "--name", name, "--ports", "22/tcp", "--volume-in-gb", "10",
                      "--container-disk-in-gb", "30", "--env", json.dumps(environment),
                      timeout=100)
        pod_id = created["id"]
    except Exception:
        # The named attempt remains consumed. Discover and stop an unknown
        # provider outcome by exact name; never send a second create.
        try:
            matches = [p for p in cli("pod", "list", "--all") if p["name"] == name]
            for pod in matches:
                cli("pod", "stop", pod["id"])
        finally:
            raise
    marker.update({"status": "created_stop_timer_arming", "pod_id": pod_id,
                   "provider_create": {"id": pod_id, "name": name,
                                       "costPerHr": created.get("costPerHr")}})
    write(ART / "pod-created.json", marker)
    watchdog_log = (ART / "watchdog.log").open("a")
    process = subprocess.Popen(
        [sys.executable, str(Path(__file__)), "watchdog", "--pod-id", pod_id,
         "--deadline", marker["hard_stop_at_utc"]],
        stdin=subprocess.DEVNULL, stdout=watchdog_log, stderr=subprocess.STDOUT,
        start_new_session=True, cwd=ROOT)
    write(ART / "watchdog-start.json", {"pod_id": pod_id, "pid": process.pid,
                                         "started_at_utc": datetime.now(timezone.utc).isoformat()})
    print(json.dumps({"pod_id": pod_id, "name": name,
                      "watchdog_pid": process.pid,
                      "hard_stop_at_utc": marker["hard_stop_at_utc"]}))


def watchdog(pod_id: str, deadline: str) -> None:
    end = datetime.fromisoformat(deadline)
    while datetime.now(timezone.utc) < end:
        time.sleep(min(10, max(1, (end - datetime.now(timezone.utc)).total_seconds())))
    for attempt in range(10):
        try:
            stopped = cli("pod", "stop", pod_id)
            print(json.dumps({"at_utc": datetime.now(timezone.utc).isoformat(),
                              "attempt": attempt + 1, "stop": stopped}), flush=True)
            return
        except Exception as exc:
            print(json.dumps({"attempt": attempt + 1,
                              "error": f"{type(exc).__name__}: {exc}"}), flush=True)
            time.sleep(10)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("phase", choices=("create", "watchdog"))
    parser.add_argument("--pod-id")
    parser.add_argument("--deadline")
    args = parser.parse_args()
    if args.phase == "create":
        create()
    else:
        if not args.pod_id or not args.deadline:
            parser.error("watchdog requires pod and deadline")
        watchdog(args.pod_id, args.deadline)


if __name__ == "__main__":
    main()
