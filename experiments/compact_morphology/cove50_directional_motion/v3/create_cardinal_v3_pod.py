"""One-shot A40 create with a detached provider-stop watchdog."""

from __future__ import annotations

import argparse
from datetime import datetime, timedelta, timezone
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time
import uuid

ROOT = Path(__file__).resolve().parents[4]
OUT = ROOT / "artifacts/ape-low-level-training-20261015/cardinal-v3"
METHOD = OUT / "method.json"
MANIFEST = OUT / "bundle-manifest.json"
BUNDLE = OUT / "worker-source.tar.gz"
KEY = Path.home() / ".runpod/ssh/runpodctl-ssh-key"


def cli(*parts: str, timeout: int = 45):
    result = subprocess.run(["runpodctl", *parts], text=True, capture_output=True,
                            timeout=timeout, check=False)
    if result.returncode:
        raise RuntimeError(f"runpodctl {parts[:2]} exited {result.returncode}: {result.stderr[:400]}")
    return json.loads(result.stdout)


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def durable(path: Path, value: dict, *, exclusive: bool = False) -> None:
    target = path if exclusive else path.with_name(path.name + ".incoming")
    with target.open("x" if exclusive else "w") as output:
        json.dump(value, output, sort_keys=True, indent=2)
        output.write("\n")
        output.flush(); os.fsync(output.fileno())
    if not exclusive:
        target.replace(path)


def matches(name: str) -> list[dict]:
    return [pod for pod in cli("pod", "list", "--all", timeout=60)
            if pod.get("name") == name]


def watchdog(name: str, deadline_text: str) -> None:
    deadline = datetime.fromisoformat(deadline_text)
    log = OUT / "watchdog-events.jsonl"
    def event(**fields):
        with log.open("a") as stream:
            stream.write(json.dumps({"at_utc": datetime.now(timezone.utc).isoformat(), **fields}) + "\n")
            stream.flush(); os.fsync(stream.fileno())
    durable(OUT / "watchdog-armed.json", {"name": name,
             "armed_at_utc": datetime.now(timezone.utc).isoformat(),
             "hard_stop_at_utc": deadline_text}, exclusive=True)
    event(action="armed", name=name, deadline=deadline_text)
    while datetime.now(timezone.utc) < deadline:
        time.sleep(min(15, max(1, (deadline - datetime.now(timezone.utc)).total_seconds())))
    for attempt in range(30):
        try:
            found = matches(name)
            running = [pod for pod in found if pod.get("runtimeStatus") != "stopped"]
            for pod in running:
                event(action="stop_requested", pod_id=pod["id"], attempt=attempt + 1)
                cli("pod", "stop", pod["id"], timeout=60)
            if found and not running and all(p.get("runtimeStatus") == "stopped" for p in found):
                event(action="provider_stopped_confirmed", pod_ids=[p["id"] for p in found])
                return
            if not found:
                event(action="no_matching_pod", attempt=attempt + 1)
        except Exception as exc:
            event(action="provider_query_or_stop_failed", reason=str(exc), attempt=attempt + 1)
        time.sleep(10)
    event(action="deadline_watchdog_exhausted_requires_manual_check")


def create() -> None:
    method = json.loads(METHOD.read_text())
    manifest = json.loads(MANIFEST.read_text())
    if (method["schema"] != "robot-learning-lab.cove50-cardinal-v3.method.v1"
            or manifest["method_sha256"] != sha(METHOD)
            or manifest["bundle_sha256"] != sha(BUNDLE)
            or (OUT / "pod-create.json").exists()
            or not KEY.is_file() or not KEY.with_suffix(".pub").is_file()):
        raise RuntimeError("Cardinal one-shot source or key differs")
    user = cli("user")
    active = cli("pod", "list")
    if user["clientBalance"] < 3 or active:
        raise RuntimeError("Insufficient balance or another active pod")
    requested = datetime.now(timezone.utc)
    name = "cove50-cardinal-v3-a40-" + uuid.uuid4().hex[:10]
    deadline = requested + timedelta(seconds=method["spend"]["pod_wall_limit_s"])
    marker = {"schema": "robot-learning-lab.cardinal-v3.create.v1",
              "name": name, "requested_at_utc": requested.isoformat(),
              "hard_stop_at_utc": deadline.isoformat(),
              "method_sha256": sha(METHOD), "bundle_sha256": sha(BUNDLE),
              "status": "attempt_consumed_before_provider_create"}
    durable(OUT / "pod-create.json", marker, exclusive=True)
    if sys.platform != "darwin" or not shutil.which("caffeinate"):
        raise RuntimeError("Detached host watchdog requires macOS caffeinate")
    watchdog_log = (OUT / "watchdog.log").open("a")
    process = subprocess.Popen(["caffeinate", "-ims", sys.executable, str(Path(__file__)), "watchdog",
                                "--name", name, "--deadline", deadline.isoformat()],
                               stdin=subprocess.DEVNULL, stdout=watchdog_log,
                               stderr=subprocess.STDOUT, start_new_session=True, cwd=ROOT)
    armed = OUT / "watchdog-armed.json"
    for _ in range(50):
        if process.poll() is not None:
            raise RuntimeError("Detached watchdog exited before provider create")
        if armed.is_file() and json.loads(armed.read_text()).get("name") == name:
            break
        time.sleep(.1)
    else:
        raise RuntimeError("Detached watchdog did not arm")
    durable(OUT / "watchdog-start.json", {"name": name, "pid": process.pid,
                                         "started_at_utc": datetime.now(timezone.utc).isoformat()})
    env = {"MUJOCO_GL": "egl", "NVIDIA_DRIVER_CAPABILITIES": "compute,utility,graphics",
           "PUBLIC_KEY": KEY.with_suffix(".pub").read_text().strip()}
    try:
        pod = cli("pod", "create", "--image",
                  "runpod/pytorch:1.0.3-cu1281-torch291-ubuntu2404",
                  "--gpu-id", "NVIDIA A40", "--cloud-type", "SECURE",
                  "--name", name, "--ports", "22/tcp", "--volume-in-gb", "20",
                  "--container-disk-in-gb", "30", "--env", json.dumps(env),
                  "--wait", "--wait-timeout", "10m", timeout=720)
        if (pod.get("name") != name or pod.get("gpuCount") != 1
                or pod.get("costPerHr", 100) > method["spend"]["listed_rate_max_usd_h"]):
            raise RuntimeError("Provider pod identity, count or rate differs")
        durable(OUT / "pod-created.json", {**marker, "status": "created",
                                           "pod_id": pod["id"], "provider": pod})
        print(json.dumps({"pod_id": pod["id"], "name": name,
                          "watchdog_pid": process.pid,
                          "hard_stop_at_utc": deadline.isoformat()}))
    except Exception:
        for pod in matches(name):
            if pod.get("runtimeStatus") != "stopped":
                cli("pod", "stop", pod["id"])
        raise


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("phase", choices=("create", "watchdog"))
    parser.add_argument("--name")
    parser.add_argument("--deadline")
    args = parser.parse_args()
    if args.phase == "create":
        create()
    else:
        if not args.name or not args.deadline:
            parser.error("watchdog needs name and deadline")
        watchdog(args.name, args.deadline)


if __name__ == "__main__":
    main()
