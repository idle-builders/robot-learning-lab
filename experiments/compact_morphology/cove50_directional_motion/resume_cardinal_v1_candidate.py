"""Resume the verified cardinal CUDA sentinel from persistent storage after host extraction failure."""

from __future__ import annotations

from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import time

from launch_cardinal_v1 import (ROOT, OUT, REMOTE_ROOT, REMOTE_OUT,
                                copy, remote, retrieve, runpod, stop)

POD_ID = "gfifwtuzxjuzpv"
KEY = Path.home() / ".runpod/ssh/runpodctl-ssh-key"


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def event(name: str, **fields) -> None:
    with (OUT / "resume-transport.jsonl").open("a") as log:
        log.write(json.dumps({"at_utc": datetime.now(timezone.utc).isoformat(),
                              "event": name, **fields}) + "\n")


def main() -> None:
    if (OUT / "resume-transport.jsonl").exists() or not KEY.is_file():
        raise RuntimeError("One-shot cardinal resume already consumed")
    method = json.loads((OUT / "method.json").read_text())
    manifest = json.loads((OUT / "bundle-manifest.json").read_text())
    sentinel = json.loads((OUT / "sentinel/result.json").read_text())
    created = json.loads((OUT / "pod-created.json").read_text())
    if (created["pod_id"] != POD_ID or created["method_sha256"] != sha(OUT / "method.json")
            or manifest["bundle_sha256"] != sha(OUT / "worker-source.tar.gz")
            or sentinel["method_sha256"] != sha(OUT / "method.json")
            or sentinel["worlds_observing_live_stop"] != 1024
            or sentinel["stop_events_with_nonzero_command"]
            or sentinel["decoded_video_frames"] != 101
            or sentinel["disposable_optimizer_updates"] != 1
            or sentinel["candidate_optimizer_updates"] != 0):
        raise RuntimeError("Cardinal method, bundle or source-checked sentinel differs")
    for name, expected in sentinel["files"].items():
        if sha(OUT / "sentinel" / name) != expected:
            raise RuntimeError(f"Local CUDA sentinel hash differs: {name}")
    pod = runpod("pod", "get", POD_ID)
    ssh = pod.get("ssh", {})
    if (pod.get("name") != created["name"] or pod.get("runtimeStatus") != "running"
            or pod.get("costPerHr", 100) > method["spend"]["listed_rate_max_usd_h"]
            or not ssh.get("ip") or not ssh.get("port")):
        raise RuntimeError("Resumed provider pod identity or SSH differs")
    host, port = "root@" + ssh["ip"], int(ssh["port"])
    event("start", pod_id=POD_ID, local_sentinel_sha256=sha(OUT / "sentinel/result.json"),
          host_transport_sha256=sha(Path(__file__).with_name("launch_cardinal_v1.py")))
    try:
        source = remote(host, port, f"sha256sum {REMOTE_OUT}/sentinel/result.json | cut -d' ' -f1", 45)
        if source != sha(OUT / "sentinel/result.json"):
            raise RuntimeError("Persistent remote sentinel differs from local source")
        if remote(host, port, "findmnt -n -o TARGET --target /workspace", 45) != "/workspace":
            raise RuntimeError("Persistent workspace missing after pod resume")
        remote(host, port, f"test ! -e {REMOTE_ROOT} && mkdir -p {REMOTE_ROOT}", 60)
        if not copy(host, port, str(OUT / "worker-source.tar.gz"), "/root/cardinal-source.tar.gz"):
            raise RuntimeError("Resumed source upload failed")
        digest = remote(host, port, "sha256sum /root/cardinal-source.tar.gz | cut -d' ' -f1", 45)
        if digest != manifest["bundle_sha256"]:
            raise RuntimeError("Resumed source bundle hash differs")
        remote(host, port, f"tar xzf /root/cardinal-source.tar.gz -C {REMOTE_ROOT}", 120)
        env = ("MUJOCO_GL=egl NVIDIA_DRIVER_CAPABILITIES=compute,utility,graphics "
               f"PYTHONPATH={REMOTE_ROOT}/experiments/compact_morphology/mjlab_ape/src:{REMOTE_ROOT}")
        script = f"{REMOTE_ROOT}/experiments/compact_morphology/cove50_directional_motion"
        checked = json.loads(remote(host, port, f"{env} python3.12 {script}/check_cardinal_v1.py", 90))
        if checked["method_sha256"] != sha(OUT / "method.json"):
            raise RuntimeError("Resumed source checker differs")
        event("source_verified", bundle_sha256=digest)
        remote(host, port,
               f"{env} timeout 1200 bash {script}/bootstrap_cardinal_v1.sh "
               "> /workspace/cardinal-resume-bootstrap.log 2>&1", 1250)
        copy(host, port, "/workspace/cardinal-resume-bootstrap.log",
             str(OUT / "resume-bootstrap.log"), inbound=True)
        gpu = remote(host, port, "nvidia-smi --query-gpu=name --format=csv,noheader", 45)
        if gpu != method["spend"]["gpu"]:
            raise RuntimeError(f"Resumed GPU differs: {gpu}")
        event("runtime_ready", gpu=gpu)
        remote(host, port,
               f"{env} timeout 4200 /root/cove50-cardinal-v1-venv/bin/python -u "
               f"{script}/run_cardinal_v1_candidate.py > /workspace/cardinal-resume-candidate.log 2>&1", 4250)
        copy(host, port, "/workspace/cardinal-resume-candidate.log",
             str(OUT / "resume-candidate-worker.log"), inbound=True)
        if not retrieve(host, port, "candidate"):
            raise RuntimeError("Resumed candidate retrieval failed")
        result = json.loads((OUT / "candidate/result.json").read_text())
        if (result["method_sha256"] != sha(OUT / "method.json")
                or result["sentinel_sha256"] != sha(OUT / "sentinel/result.json")
                or result["optimizer_updates"] != 40 or result["native_visits"] != 16):
            raise RuntimeError("Resumed candidate method or count differs")
        for name, expected in result["files"].items():
            if sha(OUT / "candidate" / name) != expected:
                raise RuntimeError(f"Resumed candidate file differs: {name}")
        event("candidate_verified", result_sha256=sha(OUT / "candidate/result.json"))
    except Exception as exc:
        event("failed", reason=f"{type(exc).__name__}: {exc}")
        retrieve(host, port, "candidate")
        for source, name in (("/workspace/cardinal-resume-bootstrap.log", "failure-resume-bootstrap.log"),
                             ("/workspace/cardinal-resume-candidate.log", "failure-resume-candidate.log")):
            copy(host, port, source, str(OUT / name), inbound=True)
        raise
    finally:
        stop(POD_ID)


if __name__ == "__main__":
    main()
