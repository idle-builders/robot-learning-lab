"""Bounded restart/stop guard for the V32 seated CUDA capacity diagnostic.

The fixed V31d pod is reused only after the source-frozen V32 plan is checked.
The cap is a local launchd/caffeinate guard, not a provider-side spend cap.
"""

from __future__ import annotations

import argparse
from datetime import datetime, timedelta, timezone
import fcntl
import getpass
import hashlib
import json
import os
from pathlib import Path
import plistlib
import shutil
import subprocess
import sys
import time


ROOT = Path(__file__).resolve().parents[3]
ART = ROOT / "artifacts/ape-cove50-stand-recovery/seated-capacity-v32"
PLAN = ART / "plan.json"
STATE = ART / "guard-state.json"
LOCK = ART / "guard.lock"
POD_ID = "xaw8w8nkpshedi"
IMAGE = "runpod/pytorch:1.0.3-cu1281-torch291-ubuntu2404"
RATE = 0.89
MAX_WALL_S = 2100
LABEL = "com.robot-learning-lab.cove50-seated-capacity-v32"
SERVICE = f"gui/{os.getuid()}/{LABEL}"
PLIST = Path.home() / "Library/LaunchAgents" / f"{LABEL}.plist"


def now() -> datetime:
    return datetime.now(timezone.utc)


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def read(path: Path) -> dict:
    return json.loads(path.read_text())


def write(path: Path, value: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(path.suffix + ".tmp")
    temp.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")
    temp.replace(path)


def edit(fn):
    ART.mkdir(parents=True, exist_ok=True)
    with LOCK.open("a+") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        current = read(STATE) if STATE.exists() else None
        value = fn(current)
        if value is not None:
            write(STATE, value)
        return value


def cli(*args: str, timeout: int = 30) -> dict:
    binary = shutil.which("runpodctl")
    if binary is None:
        raise RuntimeError("runpodctl is unavailable")
    result = subprocess.run([binary, *args], text=True, capture_output=True,
                            check=False, timeout=timeout)
    if result.returncode:
        try:
            code = json.loads(result.stderr).get("code", "unknown")
        except (ValueError, TypeError):
            code = "unknown"
        raise RuntimeError(f"runpodctl {' '.join(args[:2])} failed ({code})")
    return json.loads(result.stdout)


def provider_raw() -> dict:
    # The stop path must work even if provider metadata drifts. The fixed pod
    # ID, rather than mutable image/rate fields, is the stop target.
    return cli("pod", "get", POD_ID)


def provider() -> dict:
    item = provider_raw()
    if (item.get("id") != POD_ID or item.get("imageName") != IMAGE
            or item.get("gpuCount") != 1 or item.get("costPerHr") != RATE):
        raise RuntimeError("Fixed pod identity, image, GPU count or listed rate drifted")
    return item


def stopped(item: dict) -> bool:
    return item.get("desiredStatus") == "EXITED" and item.get("runtimeStatus") == "stopped"


def running(item: dict) -> bool:
    return item.get("desiredStatus") == "RUNNING" and item.get("runtimeStatus") == "running"


def source_contract() -> str:
    plan = read(PLAN)
    if (plan.get("schema") != "robot-learning-lab.cove50-seated-capacity.v32.plan"
            or plan.get("status") != "frozen_before_paid_capacity_diagnostic"
            or plan.get("pod_id") != POD_ID
            or plan.get("image") != IMAGE
            or plan.get("listed_rate_usd_per_hour") != RATE
            or plan.get("max_wall_s") != MAX_WALL_S
            or plan.get("optimizer_updates_max") != 0
            or plan.get("scored_original_visits_max") != 0
            or plan.get("guard_sha256") != sha(Path(__file__))):
        raise RuntimeError("V32 frozen plan or guard identity changed")
    for relative, expected in plan["source_sha256"].items():
        path = (ROOT / relative).resolve()
        if (not path.is_relative_to(ROOT.resolve()) or not path.is_file()
                or sha(path) != expected):
            raise RuntimeError(f"V32 material source changed: {relative}")
    return sha(PLAN)


def alive_caffeinate(pid: object) -> bool:
    if not isinstance(pid, int) or pid <= 0:
        return False
    result = subprocess.run(["ps", "-p", str(pid), "-o", "comm="],
                            text=True, capture_output=True, check=False, timeout=5)
    return result.returncode == 0 and Path(result.stdout.strip()).name == "caffeinate"


def service_live() -> None:
    result = subprocess.run(["launchctl", "print", SERVICE], capture_output=True,
                            check=False, timeout=10)
    if result.returncode:
        raise RuntimeError("V32 launchd guard is not loaded")


def arm() -> None:
    if sys.platform != "darwin":
        raise RuntimeError("V32 guard needs macOS launchd")
    console = subprocess.run(["stat", "-f", "%Su", "/dev/console"], text=True,
                             capture_output=True, check=True, timeout=5).stdout.strip()
    if console != getpass.getuser():
        raise RuntimeError("V32 guard requires the logged-in GUI user")
    plan_sha = source_contract()
    if not stopped(provider()):
        raise RuntimeError("V32 pod must be stopped before arming")
    if cli("user").get("clientBalance", 0) < 1.0:
        raise RuntimeError("Runpod balance below one dollar")
    awake = subprocess.Popen(["/usr/bin/caffeinate", "-dimsu"], stdin=subprocess.DEVNULL,
                             stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
                             start_new_session=True)
    time.sleep(0.1)
    if awake.poll() is not None or not alive_caffeinate(awake.pid):
        raise RuntimeError("System-awake assertion failed")
    armed = now()
    def save(current):
        if current and current.get("status") not in ("stopped", "arm_failed"):
            raise RuntimeError("An earlier V32 guard still owns a transaction")
        return {"schema": "robot-learning-lab.v32-fixed-pod-stop-guard", "pod_id": POD_ID,
                "plan_sha256": plan_sha, "guard_sha256": sha(Path(__file__)),
                "armed_at_utc": armed.isoformat(),
                "hard_deadline_utc": (armed + timedelta(seconds=MAX_WALL_S)).isoformat(),
                "awake_pid": awake.pid, "status": "armed", "stop_requested": False,
                "last_check_at_utc": None, "last_error": None, "stop_attempts": 0}
    try:
        edit(save)
        binary = shutil.which("runpodctl")
        PLIST.parent.mkdir(parents=True, exist_ok=True)
        PLIST.write_bytes(plistlib.dumps({
            "Label": LABEL,
            "ProgramArguments": [sys.executable, str(Path(__file__).resolve()), "check"],
            "EnvironmentVariables": {"PATH": f"{Path(binary).parent}:/usr/bin:/bin:/usr/sbin:/sbin"},
            "RunAtLoad": True, "StartInterval": 10,
            "StandardOutPath": str(ART / "guard.log"),
            "StandardErrorPath": str(ART / "guard.log"),
        }))
        subprocess.run(["launchctl", "bootout", SERVICE], capture_output=True,
                       check=False, timeout=10)
        subprocess.run(["launchctl", "bootstrap", f"gui/{os.getuid()}", str(PLIST)],
                       check=True, timeout=15)
        subprocess.run(["launchctl", "kickstart", "-k", SERVICE],
                       check=True, timeout=15)
        for _ in range(20):
            state = read(STATE)
            if state["last_check_at_utc"] and not state["last_error"]:
                service_live()
                if not alive_caffeinate(awake.pid):
                    raise RuntimeError("System-awake assertion vanished")
                print(json.dumps({"armed": True, "pod_id": POD_ID,
                                  "deadline": state["hard_deadline_utc"],
                                  "plan_sha256": plan_sha}))
                return
            time.sleep(0.5)
        raise RuntimeError("V32 guard did not check provider state")
    except Exception:
        def fail(current):
            if current:
                current["status"] = "arm_failed"
            return current
        edit(fail)
        if alive_caffeinate(awake.pid):
            os.kill(awake.pid, 15)
        raise


def check() -> None:
    if not STATE.exists():
        return
    def inspect(state):
        if state is None:
            return state
        try:
            item = provider_raw()
            # A start request can finish after a stop was confirmed. Continue
            # polling even in the stopped state and stop that late start.
            if state["status"] == "stopped" and stopped(item):
                state["provider_status"] = item.get("runtimeStatus")
                state["last_check_at_utc"] = now().isoformat()
                return state
            bad_host = (item.get("id") != POD_ID or item.get("imageName") != IMAGE
                        or item.get("gpuCount") != 1 or item.get("costPerHr") != RATE)
            deadline = datetime.fromisoformat(state["hard_deadline_utc"])
            try:
                drift = (source_contract() != state["plan_sha256"]
                         or sha(Path(__file__)) != state["guard_sha256"])
            except Exception:
                drift = True
            must_stop = (state["stop_requested"] or state["status"] in ("arm_failed", "stop_pending", "stopped")
                         or now() >= deadline or drift or bad_host
                         or not alive_caffeinate(state.get("awake_pid"))
                         or (state["status"] == "armed" and not stopped(item)))
            if must_stop:
                state["stop_requested"] = True
                state["status"] = "stop_pending"
                state["stop_attempts"] += 1
                if not stopped(item):
                    cli("pod", "stop", POD_ID)
                    item = provider_raw()
                if stopped(item):
                    state["status"] = "stopped"
                    state.setdefault("confirmed_stopped_at_utc", now().isoformat())
                    if alive_caffeinate(state.get("awake_pid")):
                        os.kill(state["awake_pid"], 15)
                        state["awake_pid"] = None
            state["provider_status"] = item.get("runtimeStatus")
            state["last_error"] = None
        except Exception as exc:
            state["last_error"] = f"{type(exc).__name__}: {exc}"
            state["stop_requested"] = True
            state["status"] = "stop_pending"
        state["last_check_at_utc"] = now().isoformat()
        return state
    edit(inspect)


def healthy() -> dict:
    service_live()
    state = read(STATE)
    if (state["status"] not in ("armed", "start_requested", "running")
            or state["stop_requested"] or state["last_error"]
            or not state["last_check_at_utc"]
            or (now() - datetime.fromisoformat(state["last_check_at_utc"])).total_seconds() > 20
            or now() >= datetime.fromisoformat(state["hard_deadline_utc"])
            or not alive_caffeinate(state.get("awake_pid"))
            or source_contract() != state["plan_sha256"]):
        raise RuntimeError("V32 independent stop guard is unhealthy")
    return state


def start() -> None:
    state = healthy()
    if state["status"] != "armed" or not stopped(provider()):
        raise RuntimeError("V32 start requires a stopped fixed pod and armed guard")
    def requested(current):
        if current["status"] != "armed":
            raise RuntimeError("V32 guard changed before provider start")
        current["status"] = "start_requested"
        current["start_requested_at_utc"] = now().isoformat()
        return current
    edit(requested)
    try:
        cli("pod", "start", POD_ID, timeout=60)
        end = now() + timedelta(seconds=180)
        while now() < end:
            state = healthy()
            if running(provider()):
                def ready(current):
                    if current["status"] != "start_requested":
                        raise RuntimeError("V32 guard changed during provider startup")
                    current["status"] = "running"
                    current["ready_at_utc"] = now().isoformat()
                    return current
                edit(ready)
                print(json.dumps({"pod_id": POD_ID, "status": "running",
                                  "deadline": state["hard_deadline_utc"]}))
                return
            time.sleep(3)
        raise TimeoutError("V32 pod start readiness exceeded 180 seconds")
    except Exception:
        stop()
        raise


def verify() -> None:
    state = healthy()
    if state["status"] != "running" or not running(provider()):
        raise RuntimeError("V32 pod is not running under a healthy guard")
    print(json.dumps({"pod_id": POD_ID, "plan_sha256": state["plan_sha256"],
                      "hard_deadline_utc": state["hard_deadline_utc"],
                      "heartbeat_at_utc": state["last_check_at_utc"]}))


def stop() -> None:
    if not STATE.exists():
        return
    def request(current):
        current["stop_requested"] = True
        current["status"] = "stop_pending"
        return current
    edit(request)
    check()
    state = read(STATE)
    if state["status"] != "stopped" or not stopped(provider_raw()):
        raise RuntimeError("V32 provider stop not yet confirmed; launchd continues retrying")
    print(json.dumps({"pod_id": POD_ID,
                      "confirmed_stopped_at_utc": state["confirmed_stopped_at_utc"]}))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("command", choices=("arm", "check", "start", "verify", "stop"))
    command = parser.parse_args().command
    {"arm": arm, "check": check, "start": start, "verify": verify, "stop": stop}[command]()
