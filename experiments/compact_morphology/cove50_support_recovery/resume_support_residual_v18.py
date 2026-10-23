"""One-shot EGL loader repair and fit on the stopped V18 A40 pod."""

from __future__ import annotations

import json
from pathlib import Path
import subprocess
import time

from launch_support_residual_v18 import ROOT, OUT, cli, event, remote, sha, stop, transfer


def main() -> None:
    method = json.loads((OUT / "method.json").read_text())
    repair = json.loads((OUT / "runtime-repair.json").read_text())
    created = json.loads((OUT / "pod-create.json").read_text())
    marker = OUT / "resume-attempt.json"
    trainer = ROOT / "experiments/compact_morphology/cove50_support_recovery/train_support_residual_v18.py"
    if (marker.exists() or repair["pod_id"] != created["pod_id"]
            or repair["unchanged_training_method_sha256"] != sha(OUT / "method.json")
            or repair["failure_log_sha256"] != sha(OUT / "failure-train.log")
            or method["source_sha256"][str(trainer.relative_to(ROOT))] != sha(trainer)):
        raise RuntimeError("V18 repair source, method or one-shot identity differs")
    pod_id = created["pod_id"]
    if (cli("pod", "get", pod_id).get("runtimeStatus") != "stopped"
            or cli("pod", "list") or cli("user")["clientBalance"] < 3):
        raise RuntimeError("Expected stopped V18 pod, no active pod and funded account")
    marker.write_text(json.dumps({"pod_id": pod_id, "repair_sha256": sha(OUT / "runtime-repair.json"),
                                  "status": "attempt_consumed"}, indent=2) + "\n")
    host = port = None
    started = False
    try:
        detail = cli("pod", "start", pod_id, timeout_s=90)
        started = True
        event("repair_pod_start_requested", pod_id=pod_id,
              returned_status=detail.get("runtimeStatus"))
        for attempt in range(60):
            detail = cli("pod", "get", pod_id)
            ssh = detail.get("ssh", {})
            if (detail.get("runtimeStatus") == "running" and ssh.get("ip")
                    and ssh.get("port")):
                host, port = "root@" + ssh["ip"], int(ssh["port"])
                try:
                    remote(host, port, "test -f /root/robot-learning-lab/experiments/compact_morphology/cove50_support_recovery/train_support_residual_v18.py && test -x /root/cove50-v4-venv/bin/python", 30)
                    break
                except Exception:
                    pass
            time.sleep(5)
        else:
            raise RuntimeError("Stopped pod did not regain its exact trainer and venv")
        if detail.get("name") != created["name"] or detail.get("costPerHr", 100) > method["cost_guard"]["listed_rate_max_usd_h"]:
            raise RuntimeError("V18 provider identity or rate changed")
        actual = remote(host, port, "sha256sum /root/robot-learning-lab/experiments/compact_morphology/cove50_support_recovery/train_support_residual_v18.py | cut -d' ' -f1")
        if actual != method["source_sha256"][str(trainer.relative_to(ROOT))]:
            raise RuntimeError("Resumed trainer SHA differs from frozen source")
        event("repair_source_verified", pod_id=pod_id)
        remote(host, port, "export DEBIAN_FRONTEND=noninteractive; timeout 300 apt-get update -qq && timeout 300 apt-get install -y --no-install-recommends libegl1=1.7.0-1build1 libopengl0=1.7.0-1build1 > /workspace/v18-egl-repair.log 2>&1", 660)
        if not transfer(host, port, "/workspace/v18-egl-repair.log", str(OUT / "egl-repair.log"), receive=True):
            raise RuntimeError("EGL repair log retrieval failed")
        probe = remote(host, port,
            "MUJOCO_GL=egl /root/cove50-v4-venv/bin/python -c \"import ctypes; ctypes.CDLL('libEGL.so.1'); import mujoco; from OpenGL import EGL; assert EGL.eglQueryString is not None; print(mujoco.__version__)\"", 90)
        event("egl_loader_verified", mujoco_version=probe)
        root = "/root/robot-learning-lab"
        env = f"PYTHONPATH={root}/experiments/compact_morphology/mjlab_ape/src:{root} MUJOCO_GL=egl"
        script = f"{root}/experiments/compact_morphology/cove50_support_recovery/train_support_residual_v18.py"
        pose = f"{root}/artifacts/ape-cove50-support-recovery-20261009"
        command = ("mkdir -p /workspace/robot-learning-lab && cd /workspace/robot-learning-lab && "
                   f"{env} timeout 3000 /root/cove50-v4-venv/bin/python -u {script} "
                   f"--body-xml {root}/artifacts/ape-cove50-transfer-20261008/cove50-direct-training.xml "
                   f"--right-pose-json {pose}/right-low-swing-endpoint.json "
                   f"--left-pose-json {pose}/left-low-swing-endpoint.json "
                   "--lesson load_transfer --reference-residual --residual-pilot "
                   "--domain-variation --worlds 512 --updates 32 --residual-init-std 0.05 "
                   "--seed 88300 --run-name support-residual-v18 "
                   "> /workspace/v18-train.log 2>&1")
        event("fit_restarted", pod_id=pod_id)
        remote(host, port, command, 3120)
        event("fit_command_completed", pod_id=pod_id)
        if not transfer(host, port, "/workspace/v18-train.log", str(OUT / "train.log"), receive=True):
            raise RuntimeError("Training log retrieval failed")
        remote(host, port,
               "cd /workspace/robot-learning-lab && tar czf /workspace/v18-fit-evidence.tar.gz logs", 120)
        if not transfer(host, port, "/workspace/v18-fit-evidence.tar.gz", str(OUT / "fit-evidence.tar.gz"), receive=True):
            raise RuntimeError("Checkpoint retrieval failed")
        event("fit_evidence_retrieved", archive_sha256=sha(OUT / "fit-evidence.tar.gz"))
    except Exception as exc:
        event("repair_or_fit_failed", reason=f"{type(exc).__name__}: {exc}")
        if host is not None:
            transfer(host, port, "/workspace/v18-egl-repair.log", str(OUT / "failure-egl-repair.log"), receive=True)
            transfer(host, port, "/workspace/v18-train.log", str(OUT / "failure-train-after-repair.log"), receive=True)
        raise
    finally:
        if started:
            stop(pod_id)


if __name__ == "__main__":
    main()
