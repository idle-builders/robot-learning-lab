"""Run one bounded native V18 evaluation, retain originals, then stop the pod."""

from __future__ import annotations

from datetime import datetime, timedelta, timezone
import hashlib
import json
import os
from pathlib import Path
import shlex
import subprocess
import sys
import tarfile
import time
import uuid

import launch_support_residual_v18b as infra


ROOT = Path(__file__).resolve().parents[3]
SOURCE = ROOT / "artifacts/ape-low-level-training-20261014/support-residual-v18"
OUT = SOURCE / "cuda-eval-v5"
METHOD = SOURCE / "evaluation-method-v5.json"
CHECKPOINT = SOURCE / "model_31.pt"
BASE = ROOT / "artifacts/ape-low-level-training-20261014/locomotion-fast-v4c-a40/worker-source.tar.gz"
TRAIN_OVERLAY = SOURCE / "support-overlay.tar.gz"
REMOTE_ROOT = "/root/robot-learning-lab"
infra.OUT = OUT


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def make_overlay() -> Path:
    overlay = OUT / "eval-overlay.tar.gz"
    members = [
        ROOT / "experiments/compact_morphology/cove50_support_recovery/evaluate_support_residual_v18.py",
        ROOT / "experiments/compact_morphology/cove50_support_recovery/run_support_residual_v18_cuda_eval.py",
        METHOD, CHECKPOINT,
    ]
    with tarfile.open(overlay, "w:gz") as archive:
        for member in members:
            archive.add(member, arcname=str(member.relative_to(ROOT)))
    return overlay


def launch() -> None:
    OUT.mkdir(exist_ok=False)
    method = json.loads(METHOD.read_text())
    runner = ROOT / "experiments/compact_morphology/cove50_support_recovery/run_support_residual_v18_cuda_eval.py"
    evaluator = runner.with_name("evaluate_support_residual_v18.py")
    if (method["schema"] != "robot-learning-lab.support-residual-v18.evaluation-method.v5"
            or method["runner_sha256"] != sha(runner)
            or method["evaluator_sha256"] != sha(evaluator)
            or method["launcher_sha256"] != sha(Path(__file__))
            or method["checkpoint_sha256"] != sha(CHECKPOINT)
            or len(method["fresh_families"]) != 8):
        raise RuntimeError("Frozen V18 evaluation bundle differs")
    overlay = make_overlay()
    (OUT / "transfer-manifest.json").write_text(json.dumps({
        "method_sha256": sha(METHOD), "checkpoint_sha256": sha(CHECKPOINT),
        "base_sha256": sha(BASE), "train_overlay_sha256": sha(TRAIN_OVERLAY),
        "eval_overlay_sha256": sha(overlay),
    }, indent=2) + "\n")
    user, active = infra.cli("user"), infra.cli("pod", "list")
    if user["clientBalance"] < 3 or active:
        raise RuntimeError("Insufficient Runpod balance or another active pod")
    name = "cove50-support-v18-eval-" + uuid.uuid4().hex[:10]
    requested = datetime.now(timezone.utc)
    deadline = requested + timedelta(seconds=2400)
    with (OUT / "pod-create.json").open("x") as record:
        json.dump({"name": name, "requested_at_utc": requested.isoformat(),
                   "hard_stop_at_utc": deadline.isoformat(),
                   "method_sha256": sha(METHOD), "status": "attempt_consumed"}, record, indent=2)
        record.flush(); os.fsync(record.fileno())
    with (OUT / "watchdog.log").open("a") as log:
        watcher = subprocess.Popen([sys.executable, str(Path(__file__)), "watchdog",
            name, deadline.isoformat()], cwd=ROOT, stdin=subprocess.DEVNULL,
            stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
    for _ in range(60):
        armed = OUT / "watchdog-retry-armed.json"
        if armed.exists() and json.loads(armed.read_text()).get("name") == name:
            break
        if watcher.poll() is not None:
            raise RuntimeError("Evaluation watchdog exited before pod creation")
        time.sleep(.1)
    else:
        raise RuntimeError("Evaluation watchdog did not arm")
    infra.event("watchdog_armed", pid=watcher.pid, name=name)
    pod_id = None
    host = port = None
    try:
        env = {"MUJOCO_GL": "egl", "NVIDIA_DRIVER_CAPABILITIES": "compute,utility,graphics",
               "PUBLIC_KEY": infra.KEY.with_suffix(".pub").read_text().strip()}
        pod = infra.cli("pod", "create", "--image", "runpod/pytorch:1.0.3-cu1281-torch291-ubuntu2404",
              "--gpu-id", "NVIDIA A40", "--cloud-type", "SECURE", "--name", name,
              "--ports", "22/tcp", "--volume-in-gb", "20", "--container-disk-in-gb", "30",
              "--env", json.dumps(env), "--wait", "--wait-timeout", "10m", timeout_s=720)
        pod_id = pod["id"]
        if pod.get("name") != name or pod.get("gpuCount") != 1 or pod.get("costPerHr", 100) > .8:
            raise RuntimeError("Evaluation pod identity/rate differs")
        (OUT / "pod-create.json").write_text(json.dumps(pod, indent=2) + "\n")
        infra.event("pod_created", pod_id=pod_id, listed_rate_usd_h=pod["costPerHr"])
        detail = infra.cli("pod", "get", pod_id)
        ssh = detail.get("ssh", {})
        if not ssh.get("ip") or not ssh.get("port"):
            raise RuntimeError("Evaluation pod has no SSH endpoint")
        host, port = "root@" + ssh["ip"], int(ssh["port"])
        if infra.remote(host, port, "findmnt -n -o TARGET --target /workspace") != "/workspace":
            raise RuntimeError("Persistent workspace mount missing")
        infra.remote(host, port, f"test ! -e {REMOTE_ROOT} && mkdir -p {REMOTE_ROOT}")
        for local, target in ((BASE, "/root/v18-base.tar.gz"),
                              (TRAIN_OVERLAY, "/root/v18-train-overlay.tar.gz"),
                              (overlay, "/root/v18-eval-overlay.tar.gz")):
            if not infra.transfer(host, port, str(local), target):
                raise RuntimeError(f"Upload failed: {local}")
            if infra.remote(host, port, f"sha256sum {target} | cut -d' ' -f1") != sha(local):
                raise RuntimeError(f"Remote archive SHA mismatch: {local}")
        infra.remote(host, port, f"tar xzf /root/v18-base.tar.gz -C {REMOTE_ROOT} && "
            f"tar xzf /root/v18-train-overlay.tar.gz -C {REMOTE_ROOT} && "
            f"tar xzf /root/v18-eval-overlay.tar.gz -C {REMOTE_ROOT}", 180)
        if infra.remote(host, port, "nvidia-smi --query-gpu=name --format=csv,noheader") != "NVIDIA A40":
            raise RuntimeError("Wrong evaluation GPU")
        infra.remote(host, port, "export DEBIAN_FRONTEND=noninteractive; "
            "timeout 300 apt-get update -qq && timeout 300 apt-get install -y "
            "--no-install-recommends libegl1=1.7.0-1build1 libopengl0=1.7.0-1build1 "
            "> /workspace/v18-eval-egl.log 2>&1", 660)
        bootstrap = f"{REMOTE_ROOT}/experiments/compact_morphology/cove50_low_level/bootstrap_locomotion_fast_v4.sh"
        infra.remote(host, port, f"timeout 1200 bash {bootstrap} > /workspace/v18-eval-bootstrap.log 2>&1", 1260)
        infra.event("runtime_ready", pod_id=pod_id)
        smoke_evaluator = f"{REMOTE_ROOT}/experiments/compact_morphology/cove50_support_recovery/evaluate_support_residual_v18.py"
        for controller in ("reference", "ppo"):
            smoke = (f"cd {REMOTE_ROOT} && PYTHONPATH={REMOTE_ROOT}/experiments/compact_morphology/mjlab_ape/src:{REMOTE_ROOT} "
                f"MUJOCO_GL=egl timeout 240 /root/cove50-v4-venv/bin/python -u {smoke_evaluator} "
                "--body-xml artifacts/ape-cove50-transfer-20261008/cove50-direct-training.xml "
                "--right-pose-json artifacts/ape-cove50-support-recovery-20261009/right-low-swing-endpoint.json "
                "--left-pose-json artifacts/ape-cove50-support-recovery-20261009/left-low-swing-endpoint.json "
                "--lesson load_transfer --reference-residual --domain-variation --preflight-only "
                f"--preflight-reward-controller {controller} --preflight-side left "
                f"--seed {method['opened_smoke_seed']} --eval-fixed-lag-steps 2 "
                "--eval-device cuda:0 --run-name support-v18-opened-cuda-smoke "
                f"--preflight-out /workspace/v18-smoke-{controller}.json "
                f"--eval-video /workspace/v18-smoke-{controller}.mp4 "
                + (f"--eval-checkpoint {CHECKPOINT.relative_to(ROOT)} " if controller == "ppo" else "")
                + f"> /workspace/v18-smoke-{controller}.log 2>&1")
            infra.remote(host, port, smoke, 260)
            observed = json.loads(infra.remote(host, port,
                "python3 -c " + shlex.quote(
                    f"import json; a=json.load(open('/workspace/v18-smoke-{controller}.json'))['reward_episode_preflight']; "
                    "print(json.dumps({k:a[k] for k in ('steps','eval_device','eval_fixed_lag_steps','checkpoint_iteration')}))")))
            if (observed["steps"] != 400 or observed["eval_device"] != "cuda:0"
                    or observed["eval_fixed_lag_steps"] != 2
                    or observed["checkpoint_iteration"] != (31 if controller == "ppo" else None)):
                raise RuntimeError(f"Opened CUDA {controller} smoke failed identity")
            video_check = infra.remote(host, port, "/root/cove50-v4-venv/bin/python -c " + shlex.quote(
                f"import imageio.v2 as i; a=i.get_reader('/workspace/v18-smoke-{controller}.mp4'); "
                "f=a.get_data(0); assert f.shape == (480,640,3) and int(f.max()) > int(f.min()); "
                "print(f.shape, int(f.max())-int(f.min()))"))
            infra.event("opened_cuda_smoke_passed", controller=controller, **observed)
            infra.event("opened_cuda_video_decoded", controller=controller, result=video_check)
        command = (f"cd {REMOTE_ROOT} && PYTHONPATH={REMOTE_ROOT}/experiments/compact_morphology/mjlab_ape/src:{REMOTE_ROOT} "
            f"MUJOCO_GL=egl timeout 1500 /root/cove50-v4-venv/bin/python -u {runner.relative_to(ROOT)} "
            f"--method {METHOD.relative_to(ROOT)} --checkpoint {CHECKPOINT.relative_to(ROOT)} "
            "--out /workspace/v18-eval > /workspace/v18-eval-run.log 2>&1")
        infra.event("evaluation_started", pod_id=pod_id)
        infra.remote(host, port, command, 1560)
        infra.event("evaluation_completed", pod_id=pod_id)
        infra.remote(host, port, "tar czf /workspace/v18-eval-evidence.tar.gz -C /workspace v18-eval "
            "v18-eval-run.log v18-eval-bootstrap.log v18-eval-egl.log "
            "v18-smoke-reference.json v18-smoke-reference.npz v18-smoke-reference.log "
            "v18-smoke-reference.mp4 v18-smoke-ppo.json v18-smoke-ppo.npz "
            "v18-smoke-ppo.log v18-smoke-ppo.mp4", 240)
        if not infra.transfer(host, port, "/workspace/v18-eval-evidence.tar.gz",
                              str(OUT / "native-evidence.tar.gz"), receive=True):
            raise RuntimeError("Native evaluation evidence retrieval failed")
        infra.event("evidence_retrieved", archive_sha256=sha(OUT / "native-evidence.tar.gz"))
    except Exception as exc:
        infra.event("launch_failed", reason=f"{type(exc).__name__}: {exc}")
        if host is not None:
            for remote_file, local in (("/workspace/v18-eval-run.log", "failure-eval.log"),
                                        ("/workspace/v18-eval-bootstrap.log", "failure-bootstrap.log"),
                                        ("/workspace/v18-smoke-reference.log", "failure-smoke-reference.log"),
                                        ("/workspace/v18-smoke-ppo.log", "failure-smoke-ppo.log")):
                infra.transfer(host, port, remote_file, str(OUT / local), receive=True)
            try:
                infra.remote(host, port, "tar czf /workspace/v18-eval-partial.tar.gz -C /workspace v18-eval", 120)
                infra.transfer(host, port, "/workspace/v18-eval-partial.tar.gz",
                               str(OUT / "partial-evidence.tar.gz"), receive=True)
            except Exception:
                pass
        raise
    finally:
        if pod_id is None:
            for found in infra.cli("pod", "list", "--all"):
                if found.get("name") == name and found.get("runtimeStatus") != "stopped":
                    pod_id = found["id"]
                    break
        if pod_id is not None:
            infra.stop(pod_id)


if __name__ == "__main__":
    if len(sys.argv) == 4 and sys.argv[1] == "watchdog":
        infra.watchdog(sys.argv[2], sys.argv[3])
    elif len(sys.argv) == 1:
        launch()
    else:
        raise SystemExit("Usage: launch_support_residual_v18_cuda_eval.py [watchdog name deadline]")
