"""Independent macOS launchd hard stop for the bounded V30 CUDA pod.

Arm before starting the pod. launchd invokes ``check`` every ten seconds and
after login. The worker does not own this process. A failed stop stays pending
and is retried; successful completion also calls ``stop`` explicitly.
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
import re
import shutil
import subprocess
import sys
import time


ROOT = Path(__file__).resolve().parents[3]
ART = ROOT / "artifacts/ape-cove50-stand-recovery/seated-v30-first24"
PLAN = ART / "plan.json"
STATE = ART / "pod-watchdog-state.json"
LABEL = "com.robot-learning-lab.cove50-seated-v30-pod-watchdog"
SERVICE = f"gui/{os.getuid()}/{LABEL}"
PLIST = Path.home() / "Library/LaunchAgents" / f"{LABEL}.plist"
POD_ID = "lwmfredgubjlg0"
POD_IMAGE = "runpod/pytorch:1.0.3-cu1281-torch291-ubuntu2404"


def utcnow() -> datetime:
    return datetime.now(timezone.utc)


def stamp(value: datetime) -> str:
    return value.isoformat()


def load_json(path: Path) -> dict:
    return json.loads(path.read_text())


def atomic_json(path: Path, value: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(value, indent=2) + "\n")
    tmp.replace(path)


def locked_state(action):
    STATE.parent.mkdir(parents=True, exist_ok=True)
    with STATE.with_suffix(".lock").open("a+") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        current = load_json(STATE) if STATE.exists() else None
        result, new_state = action(current)
        if new_state is not None:
            atomic_json(STATE, new_state)
        return result


def runpodctl(*args: str, timeout: int = 30) -> dict:
    binary = shutil.which("runpodctl")
    if not binary:
        raise RuntimeError("runpodctl is unavailable to the local watchdog")
    done = subprocess.run([binary, *args], capture_output=True, text=True,
                          timeout=timeout, check=False)
    if done.returncode:
        raise RuntimeError(f"runpodctl {' '.join(args[:2])} exited {done.returncode}: "
                           f"{done.stderr.strip()[:300]}")
    return json.loads(done.stdout)


def pod() -> dict:
    result = runpodctl("pod", "get", POD_ID)
    if result.get("id") != POD_ID:
        raise RuntimeError("Runpod returned a different pod")
    return result


def stopped(info: dict) -> bool:
    return (info.get("runtimeStatus") in ("stopped", "terminated")
            and info.get("desiredStatus") in ("EXITED", "TERMINATED"))


def provider_started_at(info: dict) -> datetime | None:
    for key in ("lastStartedAt", "startedAt"):
        value = info.get(key)
        if value:
            try:
                return datetime.fromisoformat(value.replace("Z", "+00:00")).astimezone(timezone.utc)
            except (ValueError, TypeError):
                pass
    raw = info.get("lastStatusChange", "")
    if not re.search(r"^(?:Rented by User|Bid by user):", raw, re.IGNORECASE):
        return None
    match = re.search(r"(?:Mon|Tue|Wed|Thu|Fri|Sat|Sun) [A-Z][a-z]{2} \d{1,2} "
                      r"\d{4} \d{2}:\d{2}:\d{2} GMT[+-]\d{4}", raw)
    if not match:
        return None
    return datetime.strptime(match.group(), "%a %b %d %Y %H:%M:%S GMT%z").astimezone(timezone.utc)


def start_evidence(info: dict) -> dict:
    return {key: info.get(key) for key in ("id", "runtimeStatus", "desiredStatus",
                                            "lastStatusChange", "startedAt",
                                            "lastStartedAt", "costPerHr")}


def awake_assertion_alive(pid: int | None) -> bool:
    if not isinstance(pid, int) or pid <= 0:
        return False
    result = subprocess.run(["ps", "-p", str(pid), "-o", "comm="],
                            capture_output=True, text=True, timeout=5, check=False)
    return result.returncode == 0 and Path(result.stdout.strip()).name == "caffeinate"


def plan_limits() -> tuple[dict, str, float]:
    plan = load_json(PLAN)
    if (plan["max_pod_wall_s"] != 6000 or plan["gpu_listed_rate_usd_per_hour"] != 0.89
            or plan["gpu_listed_spend_cap_usd"] != 1.50
            or plan["max_start_readiness_s"] != 180
            or plan["watchdog_pod_id"] != POD_ID
            or plan["watchdog_pod_image"] != POD_IMAGE
            or plan["source_sha256"].get("watchdog") !=
            hashlib.sha256(Path(__file__).read_bytes()).hexdigest()):
        raise ValueError("V30 reviewed watchdog source or pod/listed-rate limits changed")
    return plan, hashlib.sha256(PLAN.read_bytes()).hexdigest(), min(
        plan["max_pod_wall_s"],
        plan["gpu_listed_spend_cap_usd"] * 3600 / plan["gpu_listed_rate_usd_per_hour"])


def install() -> None:
    if sys.platform != "darwin":
        raise RuntimeError("V30 independent local watchdog requires macOS launchd")
    _, plan_sha, cap_s = plan_limits()
    info = pod()
    if (not stopped(info) or float(info["costPerHr"]) > 0.89
            or info.get("imageName") != POD_IMAGE or info.get("gpuCount") != 1):
        raise RuntimeError("V30 pod must be stopped with frozen image, GPU count and rate")
    if STATE.exists() and load_json(STATE).get("status") not in ("stopped", "disarmed"):
        raise RuntimeError("Existing V30 watchdog is active; inspect it before rearming")
    runpod_binary = shutil.which("runpodctl")
    if not runpod_binary:
        raise RuntimeError("runpodctl not found")
    armed = {"schema": "robot-learning-lab.v30-independent-pod-watchdog",
             "pod_id": POD_ID, "plan_sha256": plan_sha,
             "armed_at_utc": stamp(utcnow()), "start_requested_at_utc": None,
             "provider_started_at_utc": None, "hard_deadline_utc": None,
             "hard_cap_s": cap_s, "status": "armed", "stop_requested": False,
             "awake_pid": None, "last_check_at_utc": None,
             "last_error": None, "stop_attempts": 0}
    atomic_json(STATE, armed)
    PLIST.parent.mkdir(parents=True, exist_ok=True)
    plist = {"Label": LABEL,
             "ProgramArguments": [sys.executable, str(Path(__file__).resolve()), "check"],
             "EnvironmentVariables": {"PATH": f"{Path(runpod_binary).parent}:/usr/bin:/bin:/usr/sbin:/sbin"},
             "RunAtLoad": True, "StartInterval": 10,
             "StandardOutPath": str(ART / "pod-watchdog.log"),
             "StandardErrorPath": str(ART / "pod-watchdog.log")}
    PLIST.write_bytes(plistlib.dumps(plist))
    subprocess.run(["launchctl", "bootout", SERVICE], capture_output=True, check=False)
    subprocess.run(["launchctl", "bootstrap", f"gui/{os.getuid()}", str(PLIST)],
                   check=True, timeout=15)
    subprocess.run(["launchctl", "kickstart", "-k", SERVICE], check=True, timeout=15)
    for _ in range(20):
        status = load_json(STATE)
        if status.get("last_check_at_utc") and status.get("last_error") is None:
            subprocess.run(["launchctl", "print", SERVICE], capture_output=True,
                           check=True, timeout=15)
            print(json.dumps({"armed": True, "pod_id": POD_ID,
                              "plan_sha256": plan_sha, "hard_cap_s": cap_s,
                              "last_check_at_utc": status["last_check_at_utc"]}))
            return
        time.sleep(0.5)
    raise RuntimeError("launchd watchdog did not emit a healthy first check; do not start pod")


def check() -> None:
    if not STATE.exists():
        return
    def evaluate(current):
        if current is None or current["status"] in ("stopped", "disarmed"):
            return None, current
        try:
            try:
                _, plan_sha, _ = plan_limits()
                source_drift = current["plan_sha256"] != plan_sha or current["pod_id"] != POD_ID
            except Exception:
                source_drift = True  # An unreadable or changed plan cannot keep a paid pod alive.
            now = utcnow()
            deadline = (datetime.fromisoformat(current["hard_deadline_utc"])
                        if current["hard_deadline_utc"] else None)
            requested = (datetime.fromisoformat(current["start_requested_at_utc"])
                         if current["start_requested_at_utc"] else None)
            info = pod()
            unauthorized_start = current["status"] == "armed" and not stopped(info)
            readiness_timeout = (current["status"] == "start_requested" and requested is not None
                                 and (now - requested).total_seconds() >= 180)
            awake_alive = awake_assertion_alive(current.get("awake_pid"))
            awake_failed = (current["status"] in ("start_requested", "running")
                            and not awake_alive)
            must_stop = bool(source_drift or unauthorized_start or readiness_timeout or awake_failed
                             or current["status"] == "stop_pending" or current["stop_requested"]
                             or (deadline and now >= deadline))
            if must_stop and not stopped(info):
                runpodctl("pod", "stop", POD_ID, timeout=30)
                info = pod()
            current["last_check_at_utc"] = stamp(utcnow())
            current["last_runtime_status"] = info.get("runtimeStatus")
            current["last_error"] = None
            if must_stop:
                current["stop_attempts"] += 1
                current["status"] = "stopped" if stopped(info) else "stop_pending"
                current["stop_reason"] = ("source_or_launch_drift" if source_drift or
                                          unauthorized_start else
                                          "awake_assertion_lost" if awake_failed else
                                          "requested_or_deadline")
                if stopped(info):
                    current["confirmed_stopped_at_utc"] = stamp(utcnow())
                    if awake_assertion_alive(current.get("awake_pid")):
                        try:
                            os.kill(current["awake_pid"], 15)
                        except ProcessLookupError:
                            pass
                        current["awake_pid"] = None
        except Exception as exc:
            current["last_check_at_utc"] = stamp(utcnow())
            current["last_error"] = f"{type(exc).__name__}: {exc}"
            if current["stop_requested"]:
                current["status"] = "stop_pending"
            print(f"V30 watchdog check failed: {exc}", file=sys.stderr)
        return None, current
    locked_state(evaluate)


def start() -> None:
    _, plan_sha, cap_s = plan_limits()
    status = load_json(STATE)
    if (status["status"] != "armed" or status["plan_sha256"] != plan_sha
            or not status["last_check_at_utc"] or status["last_error"]):
        raise RuntimeError("Healthy, matching independent launchd watchdog must be armed first")
    if (utcnow() - datetime.fromisoformat(status["last_check_at_utc"])).total_seconds() > 30:
        raise RuntimeError("Watchdog heartbeat is stale; do not start paid pod")
    subprocess.run(["launchctl", "print", SERVICE], capture_output=True,
                   check=True, timeout=15)
    console_user = subprocess.run(["stat", "-f", "%Su", "/dev/console"],
                                  capture_output=True, text=True, check=True,
                                  timeout=5).stdout.strip()
    if console_user != getpass.getuser():
        raise RuntimeError("V30 requires the current logged-in console user")
    info = pod()
    if (not stopped(info) or float(info["costPerHr"]) > 0.89
            or info.get("imageName") != POD_IMAGE or info.get("gpuCount") != 1):
        raise RuntimeError("Frozen V30 pod image/rate/GPU or stopped status differs; refuse paid start")
    awake = subprocess.Popen(["/usr/bin/caffeinate", "-dimsu"],
                             stdin=subprocess.DEVNULL, stdout=subprocess.DEVNULL,
                             stderr=subprocess.DEVNULL, start_new_session=True)
    time.sleep(0.1)
    if awake.poll() is not None:
        raise RuntimeError("System-awake assertion did not start; refuse paid pod")
    requested = utcnow()
    def mark_request(current):
        if current["status"] != "armed":
            raise RuntimeError("Watchdog changed during start")
        current["start_requested_at_utc"] = stamp(requested)
        current["hard_deadline_utc"] = stamp(requested + timedelta(seconds=cap_s))
        current["awake_pid"] = awake.pid
        current["status"] = "start_requested"
        return None, current
    locked_state(mark_request)  # Budget timer is live before the paid request.
    try:
        response = runpodctl("pod", "start", POD_ID, timeout=60)
        observations = [{"observed_at_utc": stamp(utcnow()),
                         "source": "pod_start_response", "pod": start_evidence(response)}]
        atomic_json(ART / "pod-start-observations.json",
                    {"start_requested_at_utc": stamp(requested),
                     "observations": observations})
        actual = provider_started_at(response)
        ready = False
        preflight_deadline = requested + timedelta(seconds=180)
        while utcnow() < preflight_deadline:
            info = pod()
            observations.append({"observed_at_utc": stamp(utcnow()),
                                 "source": "pod_get", "pod": start_evidence(info)})
            atomic_json(ART / "pod-start-observations.json",
                        {"start_requested_at_utc": stamp(requested),
                         "observations": observations})
            if stopped(info):
                raise RuntimeError("Provider returned stopped before V30 readiness")
            if utcnow() >= preflight_deadline:
                raise TimeoutError("V30 provider readiness exceeded 180 seconds")
            event = provider_started_at(info)
            if event is not None and requested - timedelta(minutes=1) <= event <= utcnow():
                actual = event
            ready = info.get("desiredStatus") == "RUNNING" and info.get("runtimeStatus") == "running"
            if ready:
                break
            time.sleep(3)
        atomic_json(ART / "pod-start-observations.json",
                    {"start_requested_at_utc": stamp(requested),
                     "observations": observations})
        if not ready:
            raise RuntimeError("Provider pod did not become unambiguously running within 180 seconds")
        if actual is not None and (actual > utcnow() or actual < requested - timedelta(minutes=1)):
            actual = None  # Stale provider text is never a billing clock or start proof.
        def mark_started(current):
            if (current["status"] != "start_requested" or current["stop_requested"]
                    or not awake_assertion_alive(current.get("awake_pid"))
                    or utcnow() >= preflight_deadline):
                raise RuntimeError("V30 watchdog no longer authorizes this pod start")
            final_info = pod()
            if (utcnow() >= preflight_deadline
                    or final_info.get("desiredStatus") != "RUNNING"
                    or final_info.get("runtimeStatus") != "running"):
                raise RuntimeError("V30 pod readiness changed before guarded start was committed")
            current["provider_started_at_utc"] = stamp(actual) if actual else None
            current["start_time_basis"] = "local_start_request_before_paid_api_call"
            current["hard_deadline_utc"] = stamp(requested + timedelta(seconds=cap_s))
            current["status"] = "running"
            return None, current
        locked_state(mark_started)
        print(json.dumps({"pod_id": POD_ID,
                          "local_start_request_at_utc": stamp(requested),
                          "provider_start_event_at_utc": stamp(actual) if actual else None,
                          "hard_deadline_utc": load_json(STATE)["hard_deadline_utc"],
                          "plan_sha256": plan_sha}))
    except Exception:
        request_stop()
        raise


def request_stop() -> None:
    if not STATE.exists():
        raise RuntimeError("No armed V30 watchdog state")
    def mark(current):
        current["stop_requested"] = True
        current["status"] = "stop_pending"
        return None, current
    locked_state(mark)
    check()
    status = load_json(STATE)
    if status["status"] != "stopped":
        raise RuntimeError("Pod stop not confirmed; launchd watchdog will retry")
    if awake_assertion_alive(status.get("awake_pid")):
        try:
            os.kill(status["awake_pid"], 15)
        except ProcessLookupError:
            pass
    print(json.dumps({"pod_id": POD_ID, "status": "stopped",
                      "confirmed_stopped_at_utc": status["confirmed_stopped_at_utc"]}))


def status() -> None:
    state = load_json(STATE)
    print(json.dumps({"watchdog": state, "provider_runtime_status": pod()["runtimeStatus"]}, indent=2))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("command", choices=("install", "check", "start", "stop", "status"))
    args = parser.parse_args()
    {"install": install, "check": check, "start": start,
     "stop": request_stop, "status": status}[args.command]()


if __name__ == "__main__":
    main()
