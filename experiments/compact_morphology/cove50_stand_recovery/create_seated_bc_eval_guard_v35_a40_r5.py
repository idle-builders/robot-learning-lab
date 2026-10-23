"""Local, pre-armed Runpod create/stop transaction for V35 BC evaluation.

This only acquires a bounded host. It does not run a trajectory or optimizer. The cap is local, not a provider-side spending limit.
Run `arm`, inspect its healthy output, then run `create`. The launchd
checker discovers the unique pod name even when the create process dies before
receiving an ID. Never bypass this script with an unguarded pod create.
"""

from __future__ import annotations

import argparse
from datetime import datetime, timedelta, timezone
import fcntl
import getpass
import hashlib
import json
import math
import os
from pathlib import Path
import plistlib
import re
import shutil
import subprocess
import sys
import time
import uuid


ROOT = Path(__file__).resolve().parents[3]
ART = ROOT / "artifacts/ape-cove50-stand-recovery/seated-bc-eval-v35-a40-r5"
PLAN = ART / "plan.json"
MANIFEST = ART / "fresh-pod-creation-manifest.json"
ATTESTATION = ART / "fresh-pod-attestation.json"
STATE = ART / "fresh-pod-create-guard.json"
LOCK = ART / "fresh-pod-create-guard.lock"
BUNDLE = ART / "worker-source.tar.gz"
SOURCE_MANIFEST = ART / "source-manifest.json"
EXECUTABLE_AUDIT = ART / "executable-path-audit.json"
GATE = ROOT / "experiments/compact_morphology/cove50_stand_recovery/checkout_import_gate_v35_a40_r5.py"
RELAY = ROOT / "experiments/compact_morphology/cove50_stand_recovery/relay_seated_bc_guard_v35_a40_r5.py"
LABEL = "com.robot-learning-lab.cove50-seated-bc-eval-v35-a40-r5-create-guard"
SERVICE = f"gui/{os.getuid()}/{LABEL}"
PLIST = Path.home() / "Library/LaunchAgents" / f"{LABEL}.plist"
IMAGE = "runpod/pytorch:1.0.3-cu1281-torch291-ubuntu2404"
GPU = "NVIDIA A40"
POD_ENV = {"MUJOCO_GL": "egl",
           "NVIDIA_DRIVER_CAPABILITIES": "compute,utility,graphics"}
RATE = 0.59
MAX_POD_S = 1800
MAX_SPEND = 0.35
READINESS_S = 300
STOP_RESERVE_S = 60
POST_STOP_SETTLE_S = 600
CAP_S = min(MAX_POD_S, math.floor(MAX_SPEND * 3600 / RATE)) - STOP_RESERVE_S
POD_ID_RE = re.compile(r"^[a-z0-9]{10,32}$")
PLAN_SCHEMA = "robot-learning-lab.cove50-seated-bc-eval.v35-a40-r5.plan"
ATTESTATION_SCHEMA = "robot-learning-lab.cove50-seated-bc-eval.v35-a40-r5.guard-attestation"
MANIFEST_SCHEMA = "robot-learning-lab.cove50-seated-bc-eval.v35-a40-r5.pod-creation"
# Plan is already reviewed; dynamic pod identity belongs only in this manifest/state.
PLAN_SHA256 = "48f42975e6fd320bf2a86249aa19821d816592eba8b63e145162af6f5a84ca00"
CLOSURE_SHA256 = "7c2f52255fd27e42d2d6153b61d05c9eb7b05116a8134b65541f53451cefdfef"


def now() -> datetime:
    return datetime.now(timezone.utc)


def stamp(value: datetime) -> str:
    return value.isoformat()


def read(path: Path) -> dict:
    return json.loads(path.read_text())


def write(path: Path, value: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(path.suffix + ".tmp")
    temp.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")
    temp.replace(path)


def update(action):
    ART.mkdir(parents=True, exist_ok=True)
    with LOCK.open("a+") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        current = read(STATE) if STATE.exists() else None
        result, changed = action(current)
        if changed is not None:
            write(STATE, changed)
        return result


def cli(*args: str, timeout: int = 30):
    binary = shutil.which("runpodctl")
    if binary is None:
        raise RuntimeError("runpodctl is unavailable")
    result = subprocess.run([binary, *args], text=True, capture_output=True,
                            check=False, timeout=timeout)
    if result.returncode:
        # CLI errors may include account data. Do not persist raw stderr/stdout.
        try:
            code = json.loads(result.stderr).get("code", "unknown")
        except (ValueError, TypeError):
            code = "unknown"
        if not isinstance(code, str) or not re.fullmatch(r"[a-z_]{1,40}", code):
            code = "unknown"
        raise RuntimeError(f"runpodctl {' '.join(args[:2])} failed ({code})")
    return json.loads(result.stdout)


def source_sha() -> str:
    return hashlib.sha256(Path(__file__).read_bytes()).hexdigest()


def creation_contract() -> tuple[str, str]:
    """Bind the immutable scientific plan and reviewed local guard/bundle."""
    plan = read(PLAN)
    if (source_file_sha(PLAN) != PLAN_SHA256
            or plan.get("schema") != PLAN_SCHEMA
            or plan.get("max_pod_wall_s") != MAX_POD_S
            or plan.get("gpu_listed_rate_usd_per_hour") != RATE
            or plan.get("gpu_id") != GPU
            or plan.get("cuda_device_name") != GPU
            or plan.get("nvidia_smi_name") != GPU
            or plan.get("gpu_listed_spend_cap_usd") != MAX_SPEND
            or plan.get("max_route_wall_s") != 360
            or plan.get("max_worker_wall_s") != 1200
            or plan.get("stop_reserve_s") != STOP_RESERVE_S
            or plan.get("v31d_installed_closure_content_sha256") != CLOSURE_SHA256
            or plan.get("original_visits_max") != 0
            or plan.get("sentinel_visits_max") != 2
            or plan.get("timing_visits_max") != 1
            or plan.get("timing_seed") != 35003
            or plan.get("timing_route") != "environment_timing"
            or plan.get("timing_control_steps") != 500
            or plan.get("timing_encoded_video_frames") != 250
            or plan.get("worker_evidence_volume") != "/workspace/cove50-seated-bc-eval-v35-a40-r5"
            or plan.get("sentinel_seed") != 35900
            or plan.get("render_preflight_seed") != 35900
            or plan.get("seed_families") != []
            or plan.get("routes") != []
            or plan.get("os_packages") != {"libegl1": "1.7.0-1build1", "libopengl0": "1.7.0-1build1"}
            or plan.get("failed_a40_r3_pod_id") != "imzmr21jwlulvc"
            or plan.get("failed_a40_r4_pod_id") != "nble1kq92kjwrz"
            or plan.get("failed_a40_r4_sentinel_consumed") != 35700
            or plan.get("failed_a40_r4_timing_consumed") != 35002
            or source_file_sha(ROOT / plan["source_paths"]["failed_a40_r4_ledger"])
               != plan["failed_a40_r4_ledger_sha256"]
            or source_file_sha(ROOT / plan["source_paths"]["failed_a40_r4_timing_log"])
               != plan["failed_a40_r4_timing_log_sha256"]
            or source_file_sha(ROOT / plan["source_paths"]["reset_context_static_control"])
               != plan["reset_context_static_control_sha256"]
            or source_file_sha(ROOT / plan["source_paths"]["failed_a40_r3_ledger"])
               != plan["failed_a40_r3_ledger_sha256"]
            or plan.get("evaluator_env") != POD_ENV
            or plan.get("failed_a40_r1_sentinel_consumed") != 35100
            or plan.get("failed_a40_r1_pod_id") != "4bhiy36ik82msf"
            or plan.get("failed_a40_r2_sentinel_consumed") != 35300
            or plan.get("failed_a40_r2_timing_consumed") != 35001
            or plan.get("failed_a40_r2_pod_id") != "a9osk1e2qoaat5"
            or source_file_sha(ROOT / plan["source_paths"]["failed_a40_r1_ledger"])
               != plan["failed_a40_r1_ledger_sha256"]
            or source_file_sha(ROOT / plan["source_paths"]["failed_a40_r2_ledger"])
               != plan["failed_a40_r2_ledger_sha256"]
            or plan.get("timing_max_elapsed_wall_s") != 300
            or plan.get("no_posthoc_replacement") is not True):
        raise RuntimeError("Frozen V35 plan or budget contract changed")
    if set(plan["source_paths"]) != set(plan["source_sha256"]):
        raise RuntimeError("Frozen V35 source key set differs")
    for name, rel in plan["source_paths"].items():
        if not isinstance(rel, str):
            raise RuntimeError(f"Invalid V35 source path: {name}")
        path = (ROOT / rel).resolve()
        if (not path.is_relative_to(ROOT.resolve()) or not path.is_file()
                or source_file_sha(path) != plan["source_sha256"][name]):
            raise RuntimeError(f"Frozen V35 source changed: {name}")
    if not BUNDLE.is_file():
        raise RuntimeError("Frozen V35 worker bundle is missing")
    if not SOURCE_MANIFEST.is_file() or not EXECUTABLE_AUDIT.is_file() or not GATE.is_file() or not RELAY.is_file():
        raise RuntimeError("V35 checkout gate, executable audit, relay or source manifest is missing")
    expected = {
        "schema": MANIFEST_SCHEMA,
        "plan_path": str(PLAN.relative_to(ROOT)),
        "plan_sha256": PLAN_SHA256,
        "guard_source_path": str(Path(__file__).resolve().relative_to(ROOT)),
        "guard_source_sha256": source_sha(),
        "relay_source_path": str(RELAY.relative_to(ROOT)),
        "relay_source_sha256": source_file_sha(RELAY),
        "checkout_gate_path": str(GATE.relative_to(ROOT)),
        "checkout_gate_sha256": source_file_sha(GATE),
        "source_manifest_path": str(SOURCE_MANIFEST.relative_to(ROOT)),
        "source_manifest_sha256": source_file_sha(SOURCE_MANIFEST),
        "executable_audit_path": str(EXECUTABLE_AUDIT.relative_to(ROOT)),
        "executable_audit_sha256": source_file_sha(EXECUTABLE_AUDIT),
        "bundle_path": str(BUNDLE.relative_to(ROOT)),
        "bundle_sha256": source_file_sha(BUNDLE),
        "bootstrap_path": plan["source_paths"]["os_bootstrap"],
        "bootstrap_sha256": plan["source_sha256"]["os_bootstrap"],
        "worker_launch_path": plan["source_paths"]["worker_launch"],
        "worker_launch_sha256": plan["source_sha256"]["worker_launch"],
        "persistent_artifact_path": plan["worker_evidence_volume"],
        "worker_log_paths": [str((ART / f"worker-{stage}.log").relative_to(ROOT))
                             for stage in plan["worker_logged_stages"]],
        "worker_exit_status_paths": [str((ART / f"worker-{stage}-exit-status.json").relative_to(ROOT))
                                     for stage in plan["worker_logged_stages"]],
        "os_egl_loader_path": plan["source_paths"]["os_egl_loader"],
        "os_egl_loader_sha256": plan["source_sha256"]["os_egl_loader"],
        "os_egl_attestation_path": str((ART / "os-egl-loader-attestation.json").relative_to(ROOT)),
        "attestation_schema": ATTESTATION_SCHEMA,
        "image": IMAGE,
        "gpu_id": GPU,
        "environment": POD_ENV,
        "cloud_type": "SECURE",
        "gpu_count": 1,
        "listed_rate_usd_per_hour": RATE,
        "max_pod_wall_s": MAX_POD_S,
        "gpu_listed_spend_cap_usd": MAX_SPEND,
        "effective_cap_s": CAP_S,
        "stop_reserve_s": STOP_RESERVE_S,
        "max_start_readiness_s": READINESS_S,
        "sentinel_visits_max": 2,
        "timing_visits_max": 1,
        "original_visits_max": 0,
        "container_disk_in_gb": 30,
        "volume_in_gb": 10,
        "volume_mount_path": "/workspace",
        "ssh": True,
        "ports": ["22/tcp"],
    }
    if read(MANIFEST) != expected:
        raise RuntimeError("Reviewed V35 creation manifest differs")
    return PLAN_SHA256, source_file_sha(MANIFEST)


def source_file_sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def listed_gpu() -> dict:
    matches = [item for item in cli("gpu", "list", "--include-unavailable")
               if item.get("gpuId") == GPU]
    if (len(matches) != 1 or matches[0].get("securePricePerHr") != RATE
            or matches[0].get("secureCloud") is not True
            or matches[0].get("available") is not True
            or not any(dc.get("stockStatus") in ("Low", "Medium", "High")
                       for dc in matches[0].get("dataCenterAvailability", []))):
        raise RuntimeError("Pinned secure A40 stock or listed rate is unavailable or changed")
    return matches[0]


def matching_pods(name: str) -> list[dict]:
    pods = cli("pod", "list", "--all")
    if not isinstance(pods, list):
        raise RuntimeError("Provider pod list has an unexpected shape")
    return [pod for pod in pods if pod.get("name") == name]


def prior_pod_and_exclusivity() -> None:
    """Do not create r5 while an earlier packet or another pod is running."""
    pods = cli("pod", "list", "--all")
    if not isinstance(pods, list):
        raise RuntimeError("Provider pod list has an unexpected shape")
    for pod_id in ("4bhiy36ik82msf", "a9osk1e2qoaat5", "imzmr21jwlulvc",
                   "nble1kq92kjwrz"):
        previous = [pod for pod in pods if pod.get("id") == pod_id]
        if len(previous) != 1 or not stopped(previous[0]):
            raise RuntimeError(f"Failed A40 pod {pod_id} must remain stopped before r5")
    active = [pod.get("id") for pod in pods if not stopped(pod)]
    if active:
        raise RuntimeError(f"Another provider pod is active before r5 create: {active}")


def one_shot_packet_eligible() -> None:
    """A consumed r5 preflight or first pod create retires this launch packet."""
    ledger = read(ART / "attempt-ledger.json")
    if (ledger.get("sentinel_attempts") != 0
            or ledger.get("render_preflight_attempts") != 0
            or ledger.get("timing_attempts") != 0
            or any(route.get("attempts") != 0 for family in ledger.get("families", [])
                   for route in family.get("routes", {}).values())):
        raise RuntimeError("A40-r5 has a native attempt; a new reviewed packet is required")
    if ATTESTATION.exists():
        raise RuntimeError("A40-r5 already created a pod; a new reviewed packet is required")
    if STATE.exists():
        state = read(STATE)
        if (state.get("create_requested_at_utc") is not None or state.get("pod_id")
                or state.get("pod_ids")):
            raise RuntimeError("A40-r5 already requested a pod; a new reviewed packet is required")


def evidence(info: dict) -> dict:
    return {key: info.get(key) for key in
            ("id", "name", "imageName", "gpuCount", "costPerHr",
             "desiredStatus", "runtimeStatus", "runtimeStatusReason",
             "lastStatusChange")}


def stopped(info: dict) -> bool:
    return (info.get("desiredStatus") in ("EXITED", "TERMINATED")
            and info.get("runtimeStatus") in ("stopped", "terminated"))


def healthy_host(info: dict, name: str) -> bool:
    return (info.get("name") == name and info.get("imageName") == IMAGE
            and info.get("gpuCount") == 1 and info.get("costPerHr") == RATE)


def awake_alive(pid: object) -> bool:
    if not isinstance(pid, int) or pid <= 0:
        return False
    result = subprocess.run(["ps", "-p", str(pid), "-o", "comm="],
                            text=True, capture_output=True, check=False, timeout=5)
    return result.returncode == 0 and Path(result.stdout.strip()).name == "caffeinate"


def process_alive(pid: object) -> bool:
    if not isinstance(pid, int) or pid <= 0:
        return False
    try:
        os.kill(pid, 0)
        return True
    except ProcessLookupError:
        return False


def console_login() -> None:
    user = subprocess.run(["stat", "-f", "%Su", "/dev/console"],
                          text=True, capture_output=True, check=True,
                          timeout=5).stdout.strip()
    if user != getpass.getuser():
        raise RuntimeError("V35 seated BC guard requires the logged-in macOS GUI user")


def service_live() -> None:
    result = subprocess.run(["launchctl", "print", SERVICE], capture_output=True,
                            check=False, timeout=10)
    if result.returncode:
        raise RuntimeError("Fresh create launchd guard is not loaded")


def arm() -> None:
    if sys.platform != "darwin":
        raise RuntimeError("This local guard requires macOS launchd")
    console_login()
    sha, manifest_sha = creation_contract()
    one_shot_packet_eligible()
    listed_gpu()
    prior_pod_and_exclusivity()
    binary = shutil.which("runpodctl")
    if binary is None:
        raise RuntimeError("runpodctl is unavailable")
    name = "cove50-seated-bc-eval-v35-a40-r5-" + uuid.uuid4().hex
    if matching_pods(name):
        raise RuntimeError("Generated pod name already exists")
    awake = subprocess.Popen(["/usr/bin/caffeinate", "-dimsu"],
                             stdin=subprocess.DEVNULL, stdout=subprocess.DEVNULL,
                             stderr=subprocess.DEVNULL, start_new_session=True)
    time.sleep(0.1)
    if awake.poll() is not None or not awake_alive(awake.pid):
        raise RuntimeError("System-awake assertion failed")
    def save(current):
        if current and current.get("status") not in ("stopped",):
            raise RuntimeError("An earlier fresh create guard still owns a transaction")
        return None, {
            "schema": "robot-learning-lab.cove50-seated-bc-eval.v35-a40-r5.guard-state",
            "name": name, "plan_sha256": sha,
            "creation_manifest_sha256": manifest_sha,
            "guard_source_sha256": source_sha(),
            "image": IMAGE, "gpu_id": GPU, "gpu_count": 1,
            "cloud_type": "SECURE", "listed_rate_usd_per_hour": RATE,
            "cap_s": CAP_S, "max_pod_wall_s": MAX_POD_S,
            "gpu_listed_spend_cap_usd": MAX_SPEND,
            "readiness_s": READINESS_S,
            "armed_at_utc": stamp(now()), "create_requested_at_utc": None,
            "hard_deadline_utc": None, "awake_pid": awake.pid,
            "creator_pid": None,
            "pod_id": None, "pod_ids": [], "provider_observations": [],
            "provider_desired_status": None, "provider_runtime_status": None,
            "status": "armed", "stop_requested": False,
            "last_check_at_utc": None, "last_error": None,
            "stop_attempts": 0,
        }
    try:
        update(save)
        PLIST.parent.mkdir(parents=True, exist_ok=True)
        PLIST.write_bytes(plistlib.dumps({
            "Label": LABEL,
            "ProgramArguments": [sys.executable, str(Path(__file__).resolve()), "check"],
            "EnvironmentVariables": {
                "PATH": f"{Path(binary).parent}:/usr/bin:/bin:/usr/sbin:/sbin"},
            "RunAtLoad": True, "StartInterval": 10,
            "StandardOutPath": str(ART / "fresh-pod-create-guard.log"),
            "StandardErrorPath": str(ART / "fresh-pod-create-guard.log"),
        }))
        subprocess.run(["launchctl", "bootout", SERVICE], capture_output=True,
                       check=False, timeout=10)
        subprocess.run(["launchctl", "bootstrap", f"gui/{os.getuid()}", str(PLIST)],
                       check=True, timeout=15)
        subprocess.run(["launchctl", "kickstart", "-k", SERVICE],
                       check=True, timeout=15)
        for _ in range(20):
            state = read(STATE)
            if state.get("last_check_at_utc") and not state.get("last_error"):
                service_live()
                if not awake_alive(awake.pid):
                    raise RuntimeError("System-awake assertion vanished")
                print(json.dumps({"armed": True, "name": name,
                                  "plan_sha256": sha,
                                  "creation_manifest_sha256": manifest_sha,
                                  "cap_s": CAP_S,
                                  "last_check_at_utc": state["last_check_at_utc"]}))
                return
            time.sleep(0.5)
        raise RuntimeError("launchd guard has no healthy first check")
    except Exception:
        # A failed arm must not leave an apparently usable state.
        def failed(current):
            if current and current.get("name") == name:
                current["status"] = "arm_failed"
            return None, current
        update(failed)
        if awake_alive(awake.pid):
            os.kill(awake.pid, 15)
        raise


def check() -> None:
    if not STATE.exists():
        return
    def inspect(state):
        if state is None:
            return None, state
        try:
            current_time = now()
            list_error = None
            try:
                pods = matching_pods(state["name"])
            except Exception as exc:
                pods = []
                list_error = f"{type(exc).__name__}: {exc}"
            known = set(state["pod_ids"])
            for item in pods:
                pod_id = item.get("id")
                if not isinstance(pod_id, str) or not POD_ID_RE.fullmatch(pod_id):
                    raise RuntimeError("Provider returned an invalid pod ID")
                known.add(pod_id)
            # A delayed provider create can complete after a stop was
            # confirmed. Keep discovering this unique name after stop.
            if (state["status"] in ("stopped", "stopped_settling") and list_error is None
                    and pods and all(stopped(item) for item in pods)):
                state["pod_ids"] = sorted(known)
                state["pod_id"] = state["pod_ids"][0] if len(state["pod_ids"]) == 1 else None
                state["provider_observations"] = [evidence(item) for item in pods]
                state["provider_desired_status"] = pods[0].get("desiredStatus") if len(pods) == 1 else None
                state["provider_runtime_status"] = pods[0].get("runtimeStatus") if len(pods) == 1 else None
                if (state["status"] == "stopped_settling"
                        and not process_alive(state.get("creator_pid"))
                        and current_time >= datetime.fromisoformat(
                            state["settle_until_utc"])):
                    state["status"] = "stopped"
                    if awake_alive(state.get("awake_pid")):
                        os.kill(state["awake_pid"], 15)
                        state["awake_pid"] = None
                state["last_error"] = None
                state["last_check_at_utc"] = stamp(now())
                return None, state
            state["pod_ids"] = sorted(known)
            state["pod_id"] = state["pod_ids"][0] if len(state["pod_ids"]) == 1 else None
            state["provider_observations"] = [evidence(item) for item in pods]
            state["provider_desired_status"] = pods[0].get("desiredStatus") if len(pods) == 1 else None
            state["provider_runtime_status"] = pods[0].get("runtimeStatus") if len(pods) == 1 else None
            requested = (datetime.fromisoformat(state["create_requested_at_utc"])
                         if state["create_requested_at_utc"] else None)
            deadline = (datetime.fromisoformat(state["hard_deadline_utc"])
                        if state["hard_deadline_utc"] else None)
            try:
                plan_sha, manifest_sha = creation_contract()
                plan_drift = (state["plan_sha256"] != plan_sha
                              or state["creation_manifest_sha256"] != manifest_sha)
            except Exception:
                plan_drift = True
            bad_host = any(not healthy_host(item, state["name"]) for item in pods)
            must_stop = (
                state["stop_requested"] or state["status"] in (
                    "stop_pending", "arm_failed", "stopped_settling", "stopped")
                or (state["status"] == "armed" and bool(pods))
                or (list_error is not None and requested is not None)
                or (state["status"] == "create_requested"
                    and not process_alive(state.get("creator_pid")))
                or (state["status"] in ("create_requested", "running")
                    and not awake_alive(state.get("awake_pid")))
                or (requested is not None and state["status"] == "create_requested"
                    and (current_time - requested).total_seconds() >= READINESS_S)
                or (deadline is not None and current_time >= deadline)
                or bad_host or plan_drift
                or state.get("guard_source_sha256") != source_sha()
            )
            if must_stop:
                state["stop_requested"] = True
                state["status"] = "stop_pending"
                state["stop_attempts"] += 1
                confirmations = []
                for pod_id in sorted(known):
                    info = cli("pod", "get", pod_id)
                    if not stopped(info):
                        cli("pod", "stop", pod_id)
                        info = cli("pod", "get", pod_id)
                    confirmations.append(evidence(info))
                state["stop_confirmations"] = confirmations
                # An absent pod after a timed-out create remains pending: it may appear
                # after the CLI process has died. Keep discovering by name.
                if list_error is None and known and len(confirmations) == len(known) and all(
                    stopped(item) for item in confirmations
                ):
                    state["status"] = "stopped_settling"
                    state.setdefault("confirmed_stopped_at_utc", stamp(now()))
                    state.setdefault("settle_until_utc", stamp(
                        now() + timedelta(seconds=POST_STOP_SETTLE_S)))
            state["last_error"] = list_error
        except Exception as exc:
            # Persist only the exception class and an operation-safe explanation.
            state["last_error"] = f"{type(exc).__name__}: {exc}"
            if state.get("create_requested_at_utc"):
                state["stop_requested"] = True
                state["status"] = "stop_pending"
        state["last_check_at_utc"] = stamp(now())
        return None, state
    update(inspect)


def create() -> None:
    sha, manifest_sha = creation_contract()
    one_shot_packet_eligible()
    console_login()
    service_live()
    listed_gpu()
    prior_pod_and_exclusivity()
    state = read(STATE)
    if (state["status"] != "armed" or state["plan_sha256"] != sha
            or state["creation_manifest_sha256"] != manifest_sha
            or state.get("guard_source_sha256") != source_sha()
            or not awake_alive(state["awake_pid"])
            or state["last_error"] is not None
            or state["last_check_at_utc"] is None
            or (now() - datetime.fromisoformat(
                state["last_check_at_utc"])).total_seconds() > 20):
        raise RuntimeError("Healthy, matching launchd guard and awake assertion required")
    if matching_pods(state["name"]):
        raise RuntimeError("Unique pod name already exists")
    requested = now()
    def mark(current):
        live_plan_sha, live_manifest_sha = creation_contract()
        service_live()
        if (current["status"] != "armed" or current["plan_sha256"] != sha
                or current["creation_manifest_sha256"] != manifest_sha
                or live_plan_sha != sha or live_manifest_sha != manifest_sha
                or current["guard_source_sha256"] != source_sha()
                or current["last_error"] is not None
                or (now() - datetime.fromisoformat(
                    current["last_check_at_utc"])).total_seconds() > 20
                or not awake_alive(current["awake_pid"])):
            raise RuntimeError("Fresh create guard changed during preflight")
        current["create_requested_at_utc"] = stamp(requested)
        current["hard_deadline_utc"] = stamp(requested + timedelta(seconds=CAP_S))
        current["creator_pid"] = os.getpid()
        current["status"] = "create_requested"
        return None, current
    update(mark)  # The local budget starts before the paid API request.
    try:
        response = cli(
            "pod", "create", "--name", state["name"], "--image", IMAGE,
            "--gpu-id", GPU, "--gpu-count", "1", "--cloud-type", "SECURE",
            "--env", json.dumps(POD_ENV, sort_keys=True, separators=(",", ":")),
            "--container-disk-in-gb", "30", "--volume-in-gb", "10",
            "--volume-mount-path", "/workspace",
            "--ssh=true", "--ports", "22/tcp", timeout=60,
        )
        def record(current):
            current["create_response"] = evidence(response)
            pod_id = response.get("id")
            if isinstance(pod_id, str) and POD_ID_RE.fullmatch(pod_id):
                current["pod_ids"] = sorted(set(current["pod_ids"]) | {pod_id})
                current["pod_id"] = pod_id if len(current["pod_ids"]) == 1 else None
            return None, current
        update(record)
        ready_by = requested + timedelta(seconds=READINESS_S)
        while now() < ready_by:
            check()
            current = read(STATE)
            if current["status"] != "create_requested":
                raise RuntimeError("Create guard requested a stop before readiness")
            pods = matching_pods(state["name"])
            if len(pods) == 1 and healthy_host(pods[0], state["name"]):
                if (pods[0].get("desiredStatus") == "RUNNING"
                        and pods[0].get("runtimeStatus") == "running"):
                    def ready(current):
                        if (current["status"] != "create_requested"
                                or current["stop_requested"] or now() >= ready_by
                                or not awake_alive(current.get("awake_pid"))):
                            raise RuntimeError("Guard no longer authorizes this host")
                        fresh = matching_pods(current["name"])
                        if (now() >= ready_by or len(fresh) != 1
                                or not healthy_host(fresh[0], current["name"])
                                or fresh[0].get("desiredStatus") != "RUNNING"
                                or fresh[0].get("runtimeStatus") != "running"):
                            raise RuntimeError("Provider readiness changed before commit")
                        current["status"] = "running"
                        current["ready_at_utc"] = stamp(now())
                        current["provider_observations"] = [evidence(fresh[0])]
                        current["provider_desired_status"] = fresh[0]["desiredStatus"]
                        current["provider_runtime_status"] = fresh[0]["runtimeStatus"]
                        return evidence(fresh[0]), current
                    update(ready)
                    attest()
                    return
            time.sleep(3)
        raise TimeoutError("Provider readiness exceeded the frozen limit")
    except Exception:
        try:
            request_stop()
        except Exception:
            pass  # The independent launchd checker retains stop_pending.
        raise


def live_attestation() -> dict:
    plan_sha, manifest_sha = creation_contract()
    service_live()
    console_login()
    state = read(STATE)
    checked = datetime.fromisoformat(state["last_check_at_utc"])
    deadline = datetime.fromisoformat(state["hard_deadline_utc"])
    if (state["status"] != "running" or state["stop_requested"]
            or state["plan_sha256"] != plan_sha
            or state["creation_manifest_sha256"] != manifest_sha
            or state["guard_source_sha256"] != source_sha()
            or state["last_error"] is not None
            or (now() - checked).total_seconds() > 20
            or now() >= deadline
            or not awake_alive(state.get("awake_pid"))):
        raise RuntimeError("Fresh pod guard is not healthy and running under its cap")
    pods = matching_pods(state["name"])
    if len(pods) != 1:
        raise RuntimeError("Expected exactly one provider pod with the frozen unique name")
    pod_id = pods[0].get("id")
    if (not isinstance(pod_id, str) or not POD_ID_RE.fullmatch(pod_id)
            or state["pod_ids"] != [pod_id] or state.get("pod_id") != pod_id):
        raise RuntimeError("Provider pod ID differs from the guarded transaction")
    info = cli("pod", "get", pod_id)
    if (not healthy_host(info, state["name"])
            or info.get("desiredStatus") != "RUNNING"
            or info.get("runtimeStatus") != "running"):
        raise RuntimeError("Provider has not confirmed the exact running host")
    return {
        "schema": ATTESTATION_SCHEMA,
        "creation_manifest_sha256": manifest_sha,
        "plan_sha256": plan_sha,
        "guard_source_sha256": source_sha(),
        "guard_service": LABEL,
        "guard_state_path": str(STATE.relative_to(ROOT)),
        "pod_name": state["name"],
        "pod_id": pod_id,
        "local_start_request_at_utc": state["create_requested_at_utc"],
        "hard_deadline_utc": state["hard_deadline_utc"],
        "guard_heartbeat_at_utc": state["last_check_at_utc"],
        "attested_at_utc": stamp(now()),
        "listed_rate_usd_per_hour": RATE,
        "cap_s": CAP_S,
        "gpu_listed_spend_cap_usd": MAX_SPEND,
        "stop_reserve_s": STOP_RESERVE_S,
        "image": IMAGE,
        "gpu_id": GPU,
        "gpu_count": 1,
        "cloud_type": "SECURE",
        "provider_desired_status": info["desiredStatus"],
        "provider_runtime_status": info["runtimeStatus"],
    }


def attest() -> None:
    payload = live_attestation()
    write(ATTESTATION, payload)
    print(json.dumps(payload, sort_keys=True))


def verify() -> None:
    saved = read(ATTESTATION)
    current = live_attestation()
    if saved != current:
        # The latest check may refresh the heartbeat. Stable binding must match,
        # and the saved snapshot must still be recent enough for preflight.
        changing = {"guard_heartbeat_at_utc", "attested_at_utc"}
        if ({key: value for key, value in saved.items() if key not in changing}
                != {key: value for key, value in current.items() if key not in changing}):
            raise RuntimeError("Saved fresh-pod attestation differs from live binding")
    if (now() - datetime.fromisoformat(saved["guard_heartbeat_at_utc"])).total_seconds() > 20:
        raise RuntimeError("Saved fresh-pod attestation heartbeat is stale")
    print(json.dumps(saved, sort_keys=True))


def request_stop() -> None:
    def mark(state):
        if state is None:
            raise RuntimeError("No fresh create transaction exists")
        if state["status"] != "stopped":
            state["stop_requested"] = True
            state["status"] = "stop_pending"
        return None, state
    update(mark)
    check()
    state = read(STATE)
    if state["status"] not in ("stopped", "stopped_settling"):
        raise RuntimeError("Provider stop unconfirmed; launchd will retry discovery and stop")
    print(json.dumps({"name": state["name"], "pod_ids": state["pod_ids"],
                      "confirmed_stopped_at_utc": state["confirmed_stopped_at_utc"],
                      "settle_until_utc": state.get("settle_until_utc")}))


def status() -> None:
    state = read(STATE)
    print(json.dumps({key: value for key, value in state.items()
                      if key not in ("awake_pid",)}, indent=2))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("command", choices=("contract", "arm", "check", "create", "attest",
                                            "verify", "stop", "status"))
    command = parser.parse_args().command
    {"contract": lambda: print(json.dumps({"plan_sha256": creation_contract()[0], "manifest_sha256": creation_contract()[1]})),
     "arm": arm, "check": check, "create": create,
     "attest": attest, "verify": verify,
     "stop": request_stop, "status": status}[command]()


if __name__ == "__main__":
    main()
