"""Run one frozen cardinal CUDA sentinel and capped learner on one A40 pod."""

from __future__ import annotations

from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import shlex
import subprocess
import tarfile
import time

ROOT = Path(__file__).resolve().parents[3]
OUT = ROOT / "artifacts/ape-low-level-training-20261014/cardinal-v1"
REMOTE_ROOT = "/root/robot-learning-lab"
REMOTE_OUT = "/workspace/robot-learning-lab/ape-low-level-training-20261014/cardinal-v1"
KEY = Path.home() / ".runpod/ssh/runpodctl-ssh-key"


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def runpod(*parts: str):
    result = subprocess.run(["runpodctl", *parts], text=True, capture_output=True,
                            timeout=60, check=False)
    if result.returncode:
        raise RuntimeError(f"runpodctl {parts[:2]} exited {result.returncode}: {result.stderr[:350]}")
    return json.loads(result.stdout)


def ssh_args(host: str, port: int) -> list[str]:
    return ["-i", str(KEY), "-o", "BatchMode=yes",
            "-o", "StrictHostKeyChecking=accept-new",
            "-o", f"UserKnownHostsFile={OUT / 'worker-known-hosts'}",
            "-o", "ConnectTimeout=15", "-p", str(port), host]


def remote(host: str, port: int, command: str, seconds: int) -> str:
    result = subprocess.run(["ssh", *ssh_args(host, port)[:-1], host,
                             "bash -lc " + shlex.quote(command)],
                            text=True, capture_output=True, timeout=seconds, check=False)
    if result.returncode:
        raise RuntimeError(f"SSH exited {result.returncode}: {result.stdout[-350:]} {result.stderr[-350:]}")
    return result.stdout.strip()


def copy(host: str, port: int, source: str, dest: str, *, inbound: bool = False) -> bool:
    args = ssh_args(host, port)
    # ssh -p and scp -P differ; exclude host and port option from the shared args.
    options = args[:-3] + ["-P", str(port)]
    command = ["scp", *options, f"{host}:{source}" if inbound else source,
               dest if inbound else f"{host}:{dest}"]
    result = subprocess.run(command, stdout=subprocess.DEVNULL,
                            stderr=subprocess.DEVNULL, timeout=300, check=False)
    return result.returncode == 0


def event(name: str, **fields) -> None:
    with (OUT / "transport.jsonl").open("a") as log:
        log.write(json.dumps({"at_utc": datetime.now(timezone.utc).isoformat(),
                              "event": name, **fields}) + "\n")


def retrieve(host: str, port: int, phase: str) -> bool:
    archive = OUT / f"{phase}-evidence.tar.gz"
    try:
        remote(host, port,
               f"test -d {REMOTE_OUT}/{phase} && tar czf /workspace/cardinal-{phase}.tar.gz "
               f"-C {REMOTE_OUT} {phase}", 180)
        if not copy(host, port, f"/workspace/cardinal-{phase}.tar.gz",
                    str(archive) + ".incoming", inbound=True):
            return False
        Path(str(archive) + ".incoming").replace(archive)
        with tarfile.open(archive, "r:gz") as tar:
            for member in tar.getmembers():
                path = Path(member.name)
                if (not (member.name == phase and member.isdir())
                        and not member.name.startswith(phase + "/")):
                    raise RuntimeError("Unexpected cardinal archive member")
                if (not (member.isdir() or member.isfile())
                        or path.is_absolute() or ".." in path.parts):
                    raise RuntimeError("Unsafe cardinal archive member")
            # The host uses Python 3.9; all members were checked above before
            # using its older extractall API.
            tar.extractall(OUT)
        return True
    except Exception as exc:
        event("retrieval_failed", phase=phase, reason=str(exc))
        return False


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
    raise RuntimeError("Provider stop unconfirmed; detached watchdog remains armed")


def main() -> None:
    created = json.loads((OUT / "pod-created.json").read_text())
    manifest = json.loads((OUT / "bundle-manifest.json").read_text())
    method = json.loads((OUT / "method.json").read_text())
    if ((OUT / "transport.jsonl").exists() or not KEY.is_file()
            or created["method_sha256"] != sha(OUT / "method.json")
            or created["bundle_sha256"] != sha(OUT / "worker-source.tar.gz")
            or manifest["bundle_sha256"] != created["bundle_sha256"]):
        raise RuntimeError("Frozen cardinal source or one-shot transport differs")
    pod_id = created["pod_id"]
    host = None
    port = None
    event("start", pod_id=pod_id, method_sha256=sha(OUT / "method.json"))
    try:
        pod = runpod("pod", "get", pod_id)
        ssh = pod.get("ssh", {})
        if (pod.get("name") != created["name"]
                or pod.get("runtimeStatus") != "running"
                or pod.get("costPerHr", 100) > method["spend"]["listed_rate_max_usd_h"]
                or not ssh.get("ip") or not ssh.get("port")):
            raise RuntimeError("Provider pod identity, price or SSH differs")
        host, port = "root@" + ssh["ip"], int(ssh["port"])
        remote(host, port, f"test ! -e {REMOTE_ROOT} && mkdir -p {REMOTE_ROOT}", 60)
        if not copy(host, port, str(OUT / "worker-source.tar.gz"), "/root/cardinal-source.tar.gz"):
            raise RuntimeError("Cardinal source upload failed")
        digest = remote(host, port, "sha256sum /root/cardinal-source.tar.gz | cut -d' ' -f1", 45)
        if digest != manifest["bundle_sha256"]:
            raise RuntimeError("Remote source bundle hash differs")
        remote(host, port, f"tar xzf /root/cardinal-source.tar.gz -C {REMOTE_ROOT}", 120)
        # Runpod exposes graphics in the container but SSH login shells do not
        # reliably inherit the pod's configured environment variables.
        env = ("MUJOCO_GL=egl NVIDIA_DRIVER_CAPABILITIES=compute,utility,graphics "
               f"PYTHONPATH={REMOTE_ROOT}/experiments/compact_morphology/mjlab_ape/src:{REMOTE_ROOT}")
        script = f"{REMOTE_ROOT}/experiments/compact_morphology/cove50_directional_motion"
        check = remote(host, port, f"{env} python3.12 {script}/check_cardinal_v1.py", 90)
        if json.loads(check)["method_sha256"] != sha(OUT / "method.json"):
            raise RuntimeError("Remote cardinal source check differs")
        if remote(host, port, "findmnt -n -o TARGET --target /workspace", 45) != "/workspace":
            raise RuntimeError("Persistent /workspace mount missing")
        event("source_verified", bundle_sha256=digest)
        remote(host, port,
               f"{env} timeout 1200 bash {script}/bootstrap_cardinal_v1.sh "
               "> /workspace/cardinal-bootstrap.log 2>&1", 1250)
        copy(host, port, "/workspace/cardinal-bootstrap.log", str(OUT / "bootstrap.log"), inbound=True)
        gpu = remote(host, port, "nvidia-smi --query-gpu=name --format=csv,noheader", 45)
        if gpu != method["spend"]["gpu"]:
            raise RuntimeError(f"Wrong cardinal GPU: {gpu}")
        event("runtime_ready", gpu=gpu)
        remote(host, port,
               f"{env} timeout 1500 /root/cove50-cardinal-v1-venv/bin/python -u "
               f"{script}/run_cardinal_v1_sentinel.py > /workspace/cardinal-sentinel.log 2>&1", 1550)
        copy(host, port, "/workspace/cardinal-sentinel.log", str(OUT / "sentinel-worker.log"), inbound=True)
        if not retrieve(host, port, "sentinel"):
            raise RuntimeError("Cardinal sentinel retrieval failed")
        sentinel = json.loads((OUT / "sentinel/result.json").read_text())
        if (sentinel["method_sha256"] != sha(OUT / "method.json")
                or sentinel["worlds_observing_live_stop"] < 1
                or sentinel["stop_events_with_nonzero_command"]
                or sentinel["disposable_optimizer_updates"] != 1):
            raise RuntimeError("Cardinal CUDA sentinel failed")
        for name, expected in sentinel["files"].items():
            if sha(OUT / "sentinel" / name) != expected:
                raise RuntimeError(f"Cardinal sentinel source differs: {name}")
        event("sentinel_verified", result_sha256=sha(OUT / "sentinel/result.json"),
              world_steps_per_s=sentinel["native_world_steps_per_second"])
        remote(host, port,
               f"{env} timeout 4200 /root/cove50-cardinal-v1-venv/bin/python -u "
               f"{script}/run_cardinal_v1_candidate.py > /workspace/cardinal-candidate.log 2>&1", 4250)
        copy(host, port, "/workspace/cardinal-candidate.log", str(OUT / "candidate-worker.log"), inbound=True)
        if not retrieve(host, port, "candidate"):
            raise RuntimeError("Cardinal candidate retrieval failed")
        result = json.loads((OUT / "candidate/result.json").read_text())
        if (result["method_sha256"] != sha(OUT / "method.json")
                or result["optimizer_updates"] != 40 or result["native_visits"] != 16):
            raise RuntimeError("Cardinal candidate method or count differs")
        for name, expected in result["files"].items():
            if sha(OUT / "candidate" / name) != expected:
                raise RuntimeError(f"Cardinal candidate file differs: {name}")
        event("candidate_verified", result_sha256=sha(OUT / "candidate/result.json"))
    except Exception as exc:
        event("failed", reason=f"{type(exc).__name__}: {exc}")
        if host is not None and port is not None:
            retrieve(host, port, "sentinel")
            retrieve(host, port, "candidate")
            for source, name in (("/workspace/cardinal-bootstrap.log", "failure-bootstrap.log"),
                                 ("/workspace/cardinal-sentinel.log", "failure-sentinel.log"),
                                 ("/workspace/cardinal-candidate.log", "failure-candidate.log")):
                copy(host, port, source, str(OUT / name), inbound=True)
        raise
    finally:
        stop(pod_id)


if __name__ == "__main__":
    main()
