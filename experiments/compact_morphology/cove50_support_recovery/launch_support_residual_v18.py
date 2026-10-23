"""One-shot guarded A40 transport for the frozen V18 Stage-A residual pilot."""

from __future__ import annotations

from datetime import datetime, timedelta, timezone
import hashlib
import json
import os
from pathlib import Path
import shlex
import subprocess
import sys
import time
import uuid


ROOT = Path(__file__).resolve().parents[3]
OUT = ROOT / "artifacts/ape-low-level-training-20261014/support-residual-v18"
BASE = ROOT / "artifacts/ape-low-level-training-20261014/locomotion-fast-v4c-a40/worker-source.tar.gz"
OVERLAY = OUT / "support-overlay.tar.gz"
KEY = Path.home() / ".runpod/ssh/runpodctl-ssh-key"
REMOTE_ROOT = "/root/robot-learning-lab"


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def event(action: str, **fields) -> None:
    with (OUT / "transport.jsonl").open("a") as log:
        log.write(json.dumps({"at_utc": datetime.now(timezone.utc).isoformat(),
                              "action": action, **fields}) + "\n")
        log.flush()
        os.fsync(log.fileno())


def cli(*parts: str, timeout_s: int = 60):
    done = subprocess.run(["runpodctl", *parts], text=True, capture_output=True,
                          check=False, timeout=timeout_s)
    if done.returncode:
        raise RuntimeError(f"runpodctl {parts[:2]} failed: {done.stderr[-400:]}")
    return json.loads(done.stdout)


def ssh_base(host: str, port: int) -> list[str]:
    return ["ssh", "-i", str(KEY), "-o", "BatchMode=yes",
            "-o", "StrictHostKeyChecking=accept-new",
            "-o", f"UserKnownHostsFile={OUT / 'worker-known-hosts'}",
            "-o", "ConnectTimeout=15", "-p", str(port), host]


def remote(host: str, port: int, command: str, timeout_s: int = 90) -> str:
    done = subprocess.run([*ssh_base(host, port), "bash -lc " + shlex.quote(command)],
                          text=True, capture_output=True, check=False, timeout=timeout_s)
    if done.returncode:
        raise RuntimeError(f"remote exit {done.returncode}: {done.stdout[-400:]} {done.stderr[-400:]}")
    return done.stdout.strip()


def transfer(host: str, port: int, source: str, destination: str, *, receive=False) -> bool:
    common = ssh_base(host, port)[1:-3]
    command = (["scp", *common, "-P", str(port), f"{host}:{source}", destination]
               if receive else
               ["scp", *common, "-P", str(port), source, f"{host}:{destination}"])
    done = subprocess.run(command, check=False, capture_output=True, timeout=240)
    return done.returncode == 0


def stop(pod_id: str) -> None:
    for attempt in range(10):
        try:
            detail = cli("pod", "get", pod_id)
            if detail.get("runtimeStatus") == "stopped":
                event("provider_stop_confirmed", pod_id=pod_id, attempt=attempt + 1)
                return
            cli("pod", "stop", pod_id)
        except Exception as exc:
            event("provider_stop_retry", pod_id=pod_id, reason=str(exc))
        time.sleep(5)
    raise RuntimeError("Provider stop unconfirmed; hard-stop watchdog remains armed")


def watchdog(name: str, deadline_text: str) -> None:
    deadline = datetime.fromisoformat(deadline_text)
    (OUT / "watchdog-armed.json").write_text(json.dumps({"name": name,
        "deadline": deadline_text, "pid": os.getpid()}) + "\n")
    while datetime.now(timezone.utc) < deadline:
        time.sleep(min(20, max(1, (deadline - datetime.now(timezone.utc)).total_seconds())))
    for attempt in range(30):
        try:
            matches = [pod for pod in cli("pod", "list", "--all")
                       if pod.get("name") == name]
            for pod in matches:
                if pod.get("runtimeStatus") != "stopped":
                    event("watchdog_stop_requested", pod_id=pod["id"], attempt=attempt + 1)
                    cli("pod", "stop", pod["id"])
            if matches and all(pod.get("runtimeStatus") == "stopped" for pod in matches):
                event("watchdog_provider_stop_confirmed", pod_ids=[pod["id"] for pod in matches])
                return
        except Exception as exc:
            event("watchdog_retry", reason=str(exc), attempt=attempt + 1)
        time.sleep(10)
    event("watchdog_exhausted_requires_manual_check")


def launch() -> None:
    method_path = OUT / "method.json"
    method = json.loads(method_path.read_text())
    train = ROOT / "experiments/compact_morphology/cove50_support_recovery/train_support_residual_v18.py"
    if ((OUT / "pod-create.json").exists() or (OUT / "transport.jsonl").exists()
            or method["schema"] != "robot-learning-lab.support-residual-v18.method.v1"
            or sha(BASE) != method["base_bundle_sha256"]
            or sha(OVERLAY) != method["overlay_sha256"]
            or sha(train) != method["source_sha256"][str(train.relative_to(ROOT))]
            or sha(ROOT / "robot/ape-current-design.json") != method["design_pointer_sha256"]
            or not KEY.is_file() or not KEY.with_suffix(".pub").is_file()):
        raise RuntimeError("Frozen V18 method, source, pointer or one-shot contract differs")
    user = cli("user")
    active = cli("pod", "list")
    if user["clientBalance"] < 3 or active:
        raise RuntimeError("Insufficient Runpod balance or another active pod")
    name = "cove50-support-residual-v18-" + uuid.uuid4().hex[:10]
    requested = datetime.now(timezone.utc)
    deadline = requested + timedelta(seconds=method["cost_guard"]["pod_wall_limit_s"])
    create_record = {"name": name, "requested_at_utc": requested.isoformat(),
                     "hard_stop_at_utc": deadline.isoformat(),
                     "method_sha256": sha(method_path), "status": "attempt_consumed"}
    with (OUT / "pod-create.json").open("x") as handle:
        json.dump(create_record, handle, indent=2)
        handle.flush(); os.fsync(handle.fileno())
    watchdog_log = (OUT / "watchdog.log").open("a")
    watcher = subprocess.Popen([sys.executable, str(Path(__file__)), "watchdog",
        name, deadline.isoformat()], cwd=ROOT, stdin=subprocess.DEVNULL,
        stdout=watchdog_log, stderr=subprocess.STDOUT, start_new_session=True)
    for _ in range(60):
        armed = OUT / "watchdog-armed.json"
        if armed.exists() and json.loads(armed.read_text()).get("name") == name:
            break
        if watcher.poll() is not None:
            raise RuntimeError("V18 hard-stop watchdog exited before pod creation")
        time.sleep(.1)
    else:
        raise RuntimeError("V18 hard-stop watchdog did not arm")
    event("watchdog_armed", pid=watcher.pid, name=name, deadline=deadline.isoformat())
    pod_id = None
    host = port = None
    try:
        env = {"MUJOCO_GL": "egl", "NVIDIA_DRIVER_CAPABILITIES": "compute,utility,graphics",
               "PUBLIC_KEY": KEY.with_suffix(".pub").read_text().strip()}
        pod = cli("pod", "create", "--image", "runpod/pytorch:1.0.3-cu1281-torch291-ubuntu2404",
                  "--gpu-id", "NVIDIA A40", "--cloud-type", "SECURE", "--name", name,
                  "--ports", "22/tcp", "--volume-in-gb", "20", "--container-disk-in-gb", "30",
                  "--env", json.dumps(env), "--wait", "--wait-timeout", "10m", timeout_s=720)
        pod_id = pod["id"]
        if pod.get("name") != name or pod.get("gpuCount") != 1 or pod.get("costPerHr", 100) > method["cost_guard"]["listed_rate_max_usd_h"]:
            raise RuntimeError("Wrong pod identity, GPU count or listed rate")
        create_record.update({"status": "created", "pod_id": pod_id, "provider": pod})
        (OUT / "pod-create.json").write_text(json.dumps(create_record, indent=2) + "\n")
        event("pod_created", pod_id=pod_id, listed_rate_usd_h=pod["costPerHr"])
        detail = cli("pod", "get", pod_id)
        ssh = detail.get("ssh", {})
        if not ssh.get("ip") or not ssh.get("port"):
            raise RuntimeError("Pod has no SSH endpoint")
        host, port = "root@" + ssh["ip"], int(ssh["port"])
        if remote(host, port, "findmnt -n -o TARGET --target /workspace") != "/workspace":
            raise RuntimeError("Persistent workspace mount missing")
        remote(host, port, f"test ! -e {REMOTE_ROOT} && mkdir -p {REMOTE_ROOT}")
        for local, target, expected in (
            (BASE, "/root/v18-base.tar.gz", method["base_bundle_sha256"]),
            (OVERLAY, "/root/v18-overlay.tar.gz", method["overlay_sha256"]),
        ):
            if not transfer(host, port, str(local), target):
                raise RuntimeError(f"Upload failed: {local}")
            if remote(host, port, f"sha256sum {target} | cut -d' ' -f1") != expected:
                raise RuntimeError(f"Remote archive SHA mismatch: {local}")
        remote(host, port, f"tar xzf /root/v18-base.tar.gz -C {REMOTE_ROOT} && tar xzf /root/v18-overlay.tar.gz -C {REMOTE_ROOT}", 180)
        for relative, expected in method["source_sha256"].items():
            actual = remote(host, port, f"sha256sum {REMOTE_ROOT}/{relative} | cut -d' ' -f1")
            if actual != expected:
                raise RuntimeError(f"Remote source mismatch: {relative}")
        if remote(host, port, "nvidia-smi --query-gpu=name --format=csv,noheader") != "NVIDIA A40":
            raise RuntimeError("Wrong runtime GPU")
        event("source_verified", pod_id=pod_id)
        bootstrap = f"{REMOTE_ROOT}/experiments/compact_morphology/cove50_low_level/bootstrap_locomotion_fast_v4.sh"
        remote(host, port, f"timeout 1200 bash {bootstrap} > /workspace/v18-bootstrap.log 2>&1", 1260)
        transfer(host, port, "/workspace/v18-bootstrap.log", str(OUT / "bootstrap.log"), receive=True)
        event("runtime_ready", pod_id=pod_id)
        env_prefix = f"PYTHONPATH={REMOTE_ROOT}/experiments/compact_morphology/mjlab_ape/src:{REMOTE_ROOT} MUJOCO_GL=egl"
        script = f"{REMOTE_ROOT}/experiments/compact_morphology/cove50_support_recovery/train_support_residual_v18.py"
        body = f"{REMOTE_ROOT}/artifacts/ape-cove50-transfer-20261008/cove50-direct-training.xml"
        pose_dir = f"{REMOTE_ROOT}/artifacts/ape-cove50-support-recovery-20261009"
        command = ("mkdir -p /workspace/robot-learning-lab && cd /workspace/robot-learning-lab && "
                   f"{env_prefix} timeout 3000 /root/cove50-v4-venv/bin/python -u {script} "
                   f"--body-xml {body} --right-pose-json {pose_dir}/right-low-swing-endpoint.json "
                   f"--left-pose-json {pose_dir}/left-low-swing-endpoint.json "
                   "--lesson load_transfer --reference-residual --residual-pilot "
                   "--domain-variation --worlds 512 --updates 32 --residual-init-std 0.05 "
                   "--seed 88300 --run-name support-residual-v18 "
                   "> /workspace/v18-train.log 2>&1")
        event("fit_started", pod_id=pod_id)
        remote(host, port, command, 3120)
        event("fit_command_completed", pod_id=pod_id)
        if not transfer(host, port, "/workspace/v18-train.log", str(OUT / "train.log"), receive=True):
            raise RuntimeError("Training log retrieval failed")
        remote(host, port, "cd /workspace/robot-learning-lab && tar czf /workspace/v18-fit-evidence.tar.gz logs", 120)
        if not transfer(host, port, "/workspace/v18-fit-evidence.tar.gz", str(OUT / "fit-evidence.tar.gz"), receive=True):
            raise RuntimeError("Training checkpoint retrieval failed")
        event("fit_evidence_retrieved", archive_sha256=sha(OUT / "fit-evidence.tar.gz"))
    except Exception as exc:
        event("launch_failed", reason=f"{type(exc).__name__}: {exc}")
        if pod_id is None:
            # A timed-out create may still have made a billable pod. The
            # detached deadline watcher is the backstop; stop it now too.
            try:
                for found in cli("pod", "list", "--all"):
                    if found.get("name") == name and found.get("runtimeStatus") != "stopped":
                        pod_id = found["id"]
                        event("late_create_recovered", pod_id=pod_id)
                        break
            except Exception as lookup_exc:
                event("late_create_lookup_failed", reason=str(lookup_exc))
        if host is not None:
            transfer(host, port, "/workspace/v18-bootstrap.log", str(OUT / "failure-bootstrap.log"), receive=True)
            transfer(host, port, "/workspace/v18-train.log", str(OUT / "failure-train.log"), receive=True)
        raise
    finally:
        if pod_id is not None:
            stop(pod_id)


if __name__ == "__main__":
    if len(sys.argv) == 4 and sys.argv[1] == "watchdog":
        watchdog(sys.argv[2], sys.argv[3])
    elif len(sys.argv) == 1:
        launch()
    else:
        raise SystemExit("Usage: launch_support_residual_v18.py [watchdog name deadline]")
