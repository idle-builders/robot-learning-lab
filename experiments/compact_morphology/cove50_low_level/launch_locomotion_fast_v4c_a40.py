"""One-shot A40 transport for the V4c short stop-policy development pilot."""

from __future__ import annotations

from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import subprocess
import tarfile
import time

from launch_locomotion_fast_v4_sentinel import receive, remote, runpod, send


ROOT = Path(__file__).resolve().parents[3]
OUT = ROOT / "artifacts/ape-low-level-training-20261014/locomotion-fast-v4c-a40"
WORKER = ROOT / "experiments/compact_morphology/cove50_low_level/run_locomotion_fast_v4_candidate.py"
REMOTE_ROOT = "/root/robot-learning-lab"
REMOTE_OUT = "/workspace/robot-learning-lab/ape-low-level-training-20261014/locomotion-fast-v4/candidate"


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def event(name: str, **data) -> None:
    with (OUT / "transport.jsonl").open("a") as log:
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
    raise RuntimeError("A40 provider stop unconfirmed; independent watchdog remains armed")


def retrieve(host: str, port: int, key: Path) -> bool:
    archive = OUT / "evidence.tar.gz"
    try:
        remote(host, port, key,
               f"test -d {REMOTE_OUT} && tar czf /workspace/v4c-a40-evidence.tar.gz "
               "-C /workspace/robot-learning-lab/ape-low-level-training-20261014/locomotion-fast-v4 candidate", 180)
        if not receive(host, port, key, "/workspace/v4c-a40-evidence.tar.gz", archive):
            return False
        with tarfile.open(archive, "r:gz") as tar:
            for member in tar.getmembers():
                if (not (member.name == "candidate" and member.isdir())
                        and not member.name.startswith("candidate/")):
                    raise RuntimeError("Unexpected V4c archive member")
                if (not (member.isdir() or member.isfile())
                        or Path(member.name).is_absolute()
                        or ".." in Path(member.name).parts):
                    raise RuntimeError("Unexpected V4c archive member")
            tar.extractall(OUT)
        return True
    except Exception as exc:
        event("retrieval_failed", reason=str(exc))
        return False


def main() -> None:
    packet = json.loads((OUT / "packet.json").read_text())
    created = json.loads((OUT / "pod-created.json").read_text())
    if ((OUT / "transport.jsonl").exists()
            or packet["schema"] != "robot-learning-lab.locomotion-fast-v4c-a40.packet.v1"
            or packet["candidate_worker_sha256"] != sha(WORKER)
            or packet["source_method_sha256"] != sha(OUT / "method.json")
            or packet["source_bundle_sha256"] != sha(OUT / "worker-source.tar.gz")
            or created["method_sha256"] != packet["source_method_sha256"]
            or created["bundle_sha256"] != packet["source_bundle_sha256"]):
        raise RuntimeError("A40 packet or one-shot source identity differs")
    pod_id = created["pod_id"]
    host = port = None
    key = Path.home() / ".runpod/ssh/runpodctl-ssh-key"
    event("start", pod_id=pod_id, packet_sha256=sha(OUT / "packet.json"))
    try:
        pod = runpod("pod", "get", pod_id)
        ssh = pod.get("ssh", {})
        if (pod.get("name") != created["name"]
                or pod.get("runtimeStatus") != "running"
                or pod.get("costPerHr", 100) > packet["listed_rate_usd_h_max"]
                or not ssh.get("ip") or not ssh.get("port")):
            raise RuntimeError("A40 provider identity/rate/SSH differs")
        host, port = "root@" + ssh["ip"], int(ssh["port"])
        remote(host, port, key, f"test ! -e {REMOTE_ROOT} && mkdir -p {REMOTE_ROOT}", 60)
        send(host, port, key, OUT / "worker-source.tar.gz", "/root/v4c-worker-source.tar.gz")
        digest = remote(host, port, key,
                        "sha256sum /root/v4c-worker-source.tar.gz | cut -d' ' -f1", 45)
        if digest != packet["source_bundle_sha256"]:
            raise RuntimeError("Remote A40 source bundle differs")
        remote(host, port, key, f"tar xzf /root/v4c-worker-source.tar.gz -C {REMOTE_ROOT}", 120)
        source = remote(host, port, key,
            f"PYTHONPATH={REMOTE_ROOT}/experiments/compact_morphology/mjlab_ape/src:{REMOTE_ROOT} "
            f"python3.12 {REMOTE_ROOT}/experiments/compact_morphology/cove50_low_level/check_locomotion_fast_v4.py", 90)
        if json.loads(source)["method_sha256"] != packet["source_method_sha256"]:
            raise RuntimeError("A40 source checker differs")
        if remote(host, port, key, "findmnt -n -o TARGET --target /workspace", 45) != "/workspace":
            raise RuntimeError("A40 persistent workspace mount missing")
        bootstrap = f"{REMOTE_ROOT}/experiments/compact_morphology/cove50_low_level/bootstrap_locomotion_fast_v4.sh"
        remote(host, port, key,
            f"PYTHONPATH={REMOTE_ROOT}/experiments/compact_morphology/mjlab_ape/src:{REMOTE_ROOT} "
            f"timeout 1200 bash {bootstrap} > /workspace/v4c-bootstrap.log 2>&1", 1250)
        receive(host, port, key, "/workspace/v4c-bootstrap.log", OUT / "bootstrap.log")
        gpu = remote(host, port, key, "nvidia-smi --query-gpu=name --format=csv,noheader", 45)
        if gpu != packet["gpu"]:
            raise RuntimeError(f"A40 GPU differs: {gpu}")
        send(host, port, key, WORKER,
             f"{REMOTE_ROOT}/experiments/compact_morphology/cove50_low_level/run_locomotion_fast_v4_candidate.py")
        digest = remote(host, port, key,
            f"sha256sum {REMOTE_ROOT}/experiments/compact_morphology/cove50_low_level/run_locomotion_fast_v4_candidate.py | cut -d' ' -f1", 45)
        if digest != packet["candidate_worker_sha256"]:
            raise RuntimeError("Remote A40 candidate worker differs")
        event("runtime_ready", gpu=gpu)
        remote(host, port, key,
            f"PYTHONPATH={REMOTE_ROOT}/experiments/compact_morphology/mjlab_ape/src:{REMOTE_ROOT} "
            "timeout 1800 /root/cove50-v4-venv/bin/python -u "
            f"{REMOTE_ROOT}/experiments/compact_morphology/cove50_low_level/run_locomotion_fast_v4_candidate.py "
            "> /workspace/v4c-worker.log 2>&1", 1850)
        receive(host, port, key, "/workspace/v4c-worker.log", OUT / "worker.log")
        if not retrieve(host, port, key):
            raise RuntimeError("A40 candidate evidence retrieval failed")
        result = json.loads((OUT / "candidate/result.json").read_text())
        if (result["method_sha256"] != packet["source_method_sha256"]
                or result["optimizer_updates"] != 24 or result["native_visits"] != 16):
            raise RuntimeError("A40 candidate result identity differs")
        for name, expected in result["files"].items():
            if sha(OUT / "candidate" / name) != expected:
                raise RuntimeError(f"A40 candidate file mismatch: {name}")
        event("candidate_verified", result_sha256=sha(OUT / "candidate/result.json"))
    except Exception as exc:
        event("candidate_failed", reason=f"{type(exc).__name__}: {exc}")
        if host is not None:
            retrieve(host, port, key)
            receive(host, port, key, "/workspace/v4c-bootstrap.log", OUT / "failure-bootstrap.log")
            receive(host, port, key, "/workspace/v4c-worker.log", OUT / "failure-worker.log")
        raise
    finally:
        stop(pod_id)


if __name__ == "__main__":
    main()
