"""One-shot transport for the V4 unscored CUDA stop/update sentinel."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import re
import shlex
import subprocess


ROOT = Path(__file__).resolve().parents[3]
OUT = ROOT / "artifacts/ape-low-level-training-20261014/locomotion-fast-v4"
METHOD = OUT / "method.json"
MANIFEST = OUT / "bundle-manifest.json"
BUNDLE = OUT / "worker-source.tar.gz"
REMOTE_ROOT = "/root/robot-learning-lab"
REMOTE_EVIDENCE = "/workspace/robot-learning-lab/ape-low-level-training-20261014/locomotion-fast-v4/sentinel"


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def runpod(*args: str):
    result = subprocess.run(["runpodctl", *args], text=True, capture_output=True,
                            timeout=60, check=False)
    if result.returncode:
        raise RuntimeError(f"runpodctl {args[:2]} exited {result.returncode}: {result.stderr[:350]}")
    return json.loads(result.stdout)


def ssh_base(host: str, port: int, key: Path) -> list[str]:
    return ["ssh", "-i", str(key), "-o", "BatchMode=yes",
            "-o", "StrictHostKeyChecking=accept-new",
            "-o", f"UserKnownHostsFile={OUT / 'worker-known-hosts'}",
            "-o", "ConnectTimeout=15", "-p", str(port), host]


def remote(host: str, port: int, key: Path, command: str, timeout_s: int) -> str:
    result = subprocess.run([*ssh_base(host, port, key), "bash -lc " + shlex.quote(command)],
                            text=True, capture_output=True, timeout=timeout_s, check=False)
    if result.returncode:
        raise RuntimeError(f"SSH command exited {result.returncode}: {result.stdout[-300:]} {result.stderr[-300:]}")
    return result.stdout.strip()


def send(host: str, port: int, key: Path, source: Path, destination: str) -> None:
    base = ssh_base(host, port, key)
    command = ["scp", *base[1:-3], "-P", str(port), str(source), f"{host}:{destination}"]
    subprocess.run(command, stdout=subprocess.DEVNULL, check=True, timeout=180)


def receive(host: str, port: int, key: Path, source: str, destination: Path) -> bool:
    base = ssh_base(host, port, key)
    temporary = destination.with_name(destination.name + ".incoming")
    result = subprocess.run(["scp", *base[1:-3], "-P", str(port),
                             f"{host}:{source}", str(temporary)],
                            stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
                            timeout=180, check=False)
    if result.returncode:
        return False
    temporary.replace(destination)
    return True


def event(name: str, **fields) -> None:
    with (OUT / "sentinel-transport.jsonl").open("a") as stream:
        stream.write(json.dumps({"at_utc": datetime.now(timezone.utc).isoformat(),
                                 "event": name, **fields}) + "\n")


def confirmed_stop(pod_id: str) -> None:
    for attempt in range(8):
        try:
            detail = runpod("pod", "get", pod_id)
            if detail.get("runtimeStatus") == "stopped":
                event("provider_stop_confirmed", attempt=attempt + 1)
                return
            runpod("pod", "stop", pod_id)
        except Exception as exc:
            event("provider_stop_retry", attempt=attempt + 1, reason=str(exc))
        import time
        time.sleep(5)
    raise RuntimeError("Provider stop unconfirmed; detached hard-stop watchdog remains active")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--host", required=True)
    parser.add_argument("--port", type=int, required=True)
    parser.add_argument("--key", type=Path, required=True)
    parser.add_argument("--pod-id", required=True)
    args = parser.parse_args()
    if (not re.fullmatch(r"root@[A-Za-z0-9.-]+", args.host)
            or not 1 <= args.port <= 65535 or not args.key.is_file()
            or (OUT / "sentinel-transport.jsonl").exists()):
        raise RuntimeError("V4 sentinel is one-shot and needs an exact SSH identity")
    method = json.loads(METHOD.read_text())
    manifest = json.loads(MANIFEST.read_text())
    created = json.loads((OUT / "pod-created.json").read_text())
    if (manifest["method_sha256"] != sha(METHOD)
            or manifest["bundle_sha256"] != sha(BUNDLE)
            or created["pod_id"] != args.pod_id
            or created["method_sha256"] != sha(METHOD)
            or created["bundle_sha256"] != sha(BUNDLE)):
        raise RuntimeError("V4 source or provider create identity differs")
    host, port, key = args.host, args.port, args.key
    try:
        pod = runpod("pod", "get", args.pod_id)
        if (pod.get("name") != created["name"]
                or pod.get("runtimeStatus") != "running"
                or pod.get("costPerHr", 100) > method["spend"]["listed_rate_max_usd_h"]):
            raise RuntimeError("Provider runtime identity or rate differs")
        event("start", pod_id=args.pod_id, method_sha256=sha(METHOD),
              bundle_sha256=sha(BUNDLE))
        remote(host, port, key, f"test ! -e {REMOTE_ROOT} && mkdir -p {REMOTE_ROOT}", 60)
        send(host, port, key, BUNDLE, "/root/v4-worker-source.tar.gz")
        actual = remote(host, port, key,
                        "sha256sum /root/v4-worker-source.tar.gz | cut -d' ' -f1", 45)
        if actual != sha(BUNDLE):
            raise RuntimeError("Remote source bundle SHA mismatch")
        remote(host, port, key,
               f"tar xzf /root/v4-worker-source.tar.gz -C {REMOTE_ROOT}", 120)
        check = f"{REMOTE_ROOT}/experiments/compact_morphology/cove50_low_level/check_locomotion_fast_v4.py"
        env = f"PYTHONPATH={REMOTE_ROOT}/experiments/compact_morphology/mjlab_ape/src:{REMOTE_ROOT}"
        source = remote(host, port, key, f"{env} python3.12 {check}", 90)
        if json.loads(source)["method_sha256"] != sha(METHOD):
            raise RuntimeError("Remote V4 source checker differs")
        event("source_verified", source_check=json.loads(source))
        mount = remote(host, port, key, "findmnt -n -o TARGET --target /workspace", 45)
        if mount != "/workspace":
            raise RuntimeError("V4 persistent /workspace mount is missing")
        bootstrap = f"{REMOTE_ROOT}/experiments/compact_morphology/cove50_low_level/bootstrap_locomotion_fast_v4.sh"
        remote(host, port, key,
               f"{env} timeout 1200 bash {bootstrap} > /workspace/v4-bootstrap.log 2>&1", 1250)
        receive(host, port, key, "/workspace/v4-bootstrap.log", OUT / "sentinel-bootstrap.log")
        gpu = remote(host, port, key, "nvidia-smi --query-gpu=name --format=csv,noheader", 45)
        if gpu != "NVIDIA GeForce RTX 4090":
            raise RuntimeError(f"Wrong V4 GPU: {gpu}")
        event("runtime_ready", gpu=gpu)
        worker = f"{REMOTE_ROOT}/experiments/compact_morphology/cove50_low_level/run_locomotion_fast_v4_sentinel.py"
        remote(host, port, key,
               f"{env} timeout 1500 /root/cove50-v4-venv/bin/python -u {worker} > /workspace/v4-sentinel.log 2>&1",
               1550)
        receive(host, port, key, "/workspace/v4-sentinel.log", OUT / "sentinel-worker.log")
        for name in ("result.json", "stop-visit.json", "stop-visit.npz",
                     "stop-visit.log", "one-update.log", "one-update-model_844.pt",
                     "source-runtime-check.log"):
            if not receive(host, port, key, f"{REMOTE_EVIDENCE}/{name}", OUT / f"sentinel-{name}"):
                raise RuntimeError(f"Missing V4 sentinel source: {name}")
        result = json.loads((OUT / "sentinel-result.json").read_text())
        if (result["method_sha256"] != sha(METHOD)
                or result["stop_visit_report_sha256"] != sha(OUT / "sentinel-stop-visit.json")
                or result["stop_visit_trace_sha256"] != sha(OUT / "sentinel-stop-visit.npz")
                or result["disposable_checkpoint_sha256"] != sha(OUT / "sentinel-one-update-model_844.pt")
                or result["candidate_optimizer_updates"] != 0):
            raise RuntimeError("Retrieved V4 CUDA sentinel source mismatch")
        event("sentinel_verified", result_sha256=sha(OUT / "sentinel-result.json"))
    except Exception as exc:
        event("sentinel_failed", reason=f"{type(exc).__name__}: {exc}")
        for remote_name, local in (("/workspace/v4-bootstrap.log", "failure-bootstrap.log"),
                                   ("/workspace/v4-sentinel.log", "failure-worker.log"),
                                   *((f"{REMOTE_EVIDENCE}/{name}", f"failure-{name}")
                                     for name in ("source-runtime-check.log", "stop-visit.log",
                                                  "stop-visit.json", "stop-visit.npz",
                                                  "one-update.log", "one-update-model_844.pt",
                                                  "result.json"))):
            try:
                receive(host, port, key, remote_name, OUT / local)
            except Exception:
                pass
        raise
    finally:
        confirmed_stop(args.pod_id)


if __name__ == "__main__":
    main()
