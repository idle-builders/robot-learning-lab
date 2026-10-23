"""Restart the stopped V4 pod once, run the short candidate, retrieve, stop."""

from __future__ import annotations

from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import subprocess
import tarfile
import time

from launch_locomotion_fast_v4_sentinel import receive, remote, runpod, send


ROOT = Path(__file__).resolve().parents[3]
OUT = ROOT / "artifacts/ape-low-level-training-20261014/locomotion-fast-v4"
WORKER = ROOT / "experiments/compact_morphology/cove50_low_level/run_locomotion_fast_v4_candidate.py"
PACKET = OUT / "candidate-packet.json"
POD = OUT / "pod-created.json"
REMOTE_ROOT = "/root/robot-learning-lab"
REMOTE_OUT = "/workspace/robot-learning-lab/ape-low-level-training-20261014/locomotion-fast-v4/candidate"


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def event(name: str, **data) -> None:
    with (OUT / "candidate-transport.jsonl").open("a") as log:
        log.write(json.dumps({"at_utc": datetime.now(timezone.utc).isoformat(),
                              "event": name, **data}) + "\n")


def stop(pod_id: str) -> None:
    for attempt in range(8):
        try:
            pod = runpod("pod", "get", pod_id)
            if pod.get("runtimeStatus") == "stopped":
                event("provider_stop_confirmed", attempt=attempt + 1)
                return
            runpod("pod", "stop", pod_id)
        except Exception as exc:
            event("provider_stop_retry", reason=str(exc), attempt=attempt + 1)
        time.sleep(5)
    raise RuntimeError("Provider stop unconfirmed; original wakeful watchdog remains armed")


def retrieve(host: str, port: int, key: Path) -> bool:
    archive = OUT / "candidate-evidence.tar.gz"
    try:
        remote(host, port, key,
               f"test -d {REMOTE_OUT} && tar czf /workspace/v4-candidate-evidence.tar.gz "
               "-C /workspace/robot-learning-lab/ape-low-level-training-20261014/locomotion-fast-v4 candidate",
               180)
        if not receive(host, port, key, "/workspace/v4-candidate-evidence.tar.gz", archive):
            return False
        with tarfile.open(archive, "r:gz") as tar:
            for member in tar.getmembers():
                if (not member.name.startswith("candidate/")
                        or not (member.isdir() or member.isfile())
                        or Path(member.name).is_absolute()
                        or ".." in Path(member.name).parts):
                    raise RuntimeError("Unexpected V4 candidate archive member")
            tar.extractall(OUT)
        return True
    except Exception as exc:
        event("retrieval_failed", reason=str(exc))
        return False


def main() -> None:
    packet = json.loads(PACKET.read_text())
    created = json.loads(POD.read_text())
    if ((OUT / "candidate-transport.jsonl").exists()
            or packet["schema"] != "robot-learning-lab.locomotion-fast-v4b.packet.v1"
            or packet["worker_sha256"] != sha(WORKER)
            or packet["method_sha256"] != sha(OUT / "method.json")
            or packet["sentinel_result_sha256"] != sha(OUT / "sentinel-result.json")
            or packet["pod_id"] != created["pod_id"]):
        raise RuntimeError("V4b source, sentinel or one-shot packet differs")
    deadline = datetime.fromisoformat(created["hard_stop_at_utc"])
    watchdog = json.loads((OUT / "watchdog-start.json").read_text())
    try:
        os.kill(watchdog["pid"], 0)
    except OSError as exc:
        raise RuntimeError("Wakeful V4 pod watchdog is not alive") from exc
    if (deadline - datetime.now(timezone.utc)).total_seconds() < 2400:
        raise RuntimeError("Insufficient guarded time for V4b candidate")
    if runpod("user")["clientBalance"] < 3 or runpod("pod", "list"):
        raise RuntimeError("Balance or active-pod preflight differs")
    pod_id = packet["pod_id"]
    before = runpod("pod", "get", pod_id)
    if before.get("runtimeStatus") != "stopped" or before.get("name") != created["name"]:
        raise RuntimeError("Exact stopped V4 pod unavailable")
    event("start_requested", pod_id=pod_id, packet_sha256=sha(PACKET))
    host = port = None
    key = Path.home() / ".runpod/ssh/runpodctl-ssh-key"
    try:
        runpod("pod", "start", pod_id)
        for _ in range(90):
            live = runpod("pod", "get", pod_id)
            ssh = live.get("ssh", {})
            if (live.get("runtimeStatus") == "running"
                    and live.get("costPerHr", 100) <= packet["listed_rate_max_usd_h"]
                    and ssh.get("ip") and ssh.get("port")):
                host, port = "root@" + ssh["ip"], int(ssh["port"])
                break
            time.sleep(5)
        if host is None:
            raise RuntimeError("V4 pod did not return SSH after restart")
        remote(host, port, key,
               f"test -f {REMOTE_ROOT}/artifacts/ape-low-level-training-20261014/locomotion-fast-v4/method.json "
               f"&& test -x /root/cove50-v4-venv/bin/python && test ! -e {REMOTE_OUT}", 60)
        remote_method = remote(host, port, key,
            f"sha256sum {REMOTE_ROOT}/artifacts/ape-low-level-training-20261014/locomotion-fast-v4/method.json | cut -d' ' -f1", 60)
        if remote_method != packet["method_sha256"]:
            raise RuntimeError("Restarted pod V4 method changed")
        send(host, port, key, WORKER,
             f"{REMOTE_ROOT}/experiments/compact_morphology/cove50_low_level/run_locomotion_fast_v4_candidate.py")
        remote_worker = remote(host, port, key,
            f"sha256sum {REMOTE_ROOT}/experiments/compact_morphology/cove50_low_level/run_locomotion_fast_v4_candidate.py | cut -d' ' -f1", 60)
        if remote_worker != packet["worker_sha256"]:
            raise RuntimeError("Remote V4b worker source mismatch")
        event("source_verified", host=host, port=port)
        command = (f"PYTHONPATH={REMOTE_ROOT}/experiments/compact_morphology/mjlab_ape/src:{REMOTE_ROOT} "
                   "timeout 1800 /root/cove50-v4-venv/bin/python -u "
                   f"{REMOTE_ROOT}/experiments/compact_morphology/cove50_low_level/run_locomotion_fast_v4_candidate.py "
                   "> /workspace/v4-candidate-wrapper.log 2>&1")
        remote(host, port, key, command, 1850)
        if not retrieve(host, port, key):
            raise RuntimeError("V4b candidate evidence retrieval failed")
        result = json.loads((OUT / "candidate/result.json").read_text())
        if (result["method_sha256"] != packet["method_sha256"]
                or result["optimizer_updates"] != 24
                or result["native_visits"] != 16):
            raise RuntimeError("V4b candidate result identity differs")
        for name, expected in result["files"].items():
            if sha(OUT / "candidate" / name) != expected:
                raise RuntimeError(f"V4b retrieved source mismatch: {name}")
        event("candidate_verified", result_sha256=sha(OUT / "candidate/result.json"))
    except Exception as exc:
        event("candidate_failed", reason=f"{type(exc).__name__}: {exc}")
        if host is not None:
            retrieve(host, port, key)
            receive(host, port, key, "/workspace/v4-candidate-wrapper.log",
                    OUT / "candidate-failure-wrapper.log")
        raise
    finally:
        stop(pod_id)


if __name__ == "__main__":
    main()
