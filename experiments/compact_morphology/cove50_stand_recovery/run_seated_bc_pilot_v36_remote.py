"""Run the four-family V36 development pilot on an already-created A40 pod.

Pod creation and a separate time-limited provider stop are operator steps.
This script never creates or starts a pod. It retrieves each original before
starting another and leaves failures visible instead of substituting a seed.
"""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import shlex
import subprocess


ROOT = Path(__file__).resolve().parents[3]
ART = ROOT / "artifacts/ape-cove50-stand-recovery/seated-bc-pilot-v36"
PLAN = ART / "plan.json"
EVAL = ROOT / "experiments/compact_morphology/cove50_stand_recovery/evaluate_seated_bc_pilot_v36.py"
BUNDLE = ROOT / "artifacts/ape-cove50-stand-recovery/seated-bc-eval-v35-a40-r5/worker-source.tar.gz"
REMOTE_ROOT = "/root/robot-learning-lab"
REMOTE_EVAL = REMOTE_ROOT + "/experiments/compact_morphology/cove50_stand_recovery/evaluate_seated_bc_pilot_v36.py"
REMOTE_OUT = "/workspace/robot-learning-lab/seated-bc-pilot-v36"
PYTHONPATH = REMOTE_ROOT + "/experiments/compact_morphology/mjlab_ape/src:" + REMOTE_ROOT


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def log(event: str, **fields) -> None:
    ART.mkdir(parents=True, exist_ok=True)
    with (ART / "transport.jsonl").open("a") as output:
        output.write(json.dumps({"at_utc": datetime.now(timezone.utc).isoformat(),
                                 "event": event, **fields}, sort_keys=True) + "\n")


def run(argv: list[str], *, timeout: int = 60, capture: bool = True) -> subprocess.CompletedProcess:
    result = subprocess.run(argv, check=False, text=True, capture_output=capture,
                            timeout=timeout)
    if result.returncode:
        raise RuntimeError(f"Command {argv[0]} exited {result.returncode}")
    return result


def ssh_args(key: Path, known: Path, port: int) -> list[str]:
    return ["-i", str(key), "-p", str(port), "-o", "BatchMode=yes",
            "-o", "StrictHostKeyChecking=accept-new", "-o", f"UserKnownHostsFile={known}",
            "-o", "ConnectTimeout=15"]


def remote(host: str, port: int, key: Path, known: Path, command: str,
           timeout: int = 90) -> str:
    result = run(["ssh", *ssh_args(key, known, port), host,
                  "bash -lc " + shlex.quote(command)], timeout=timeout)
    return result.stdout.strip()


def send(host: str, port: int, key: Path, known: Path,
         local: Path, remote_path: str) -> None:
    args = ssh_args(key, known, port)
    args[args.index("-p")] = "-P"
    run(["scp", *args, str(local), f"{host}:{remote_path}"], timeout=180)


def receive(host: str, port: int, key: Path, known: Path,
            remote_path: str, local: Path) -> bool:
    if local.exists():
        raise FileExistsError(f"Original local output exists: {local}")
    args = ssh_args(key, known, port)
    args[args.index("-p")] = "-P"
    incoming = local.with_name(local.name + ".incoming")
    result = subprocess.run(["scp", *args, f"{host}:{remote_path}", str(incoming)],
                            capture_output=True, text=True, timeout=180)
    if result.returncode:
        return False
    incoming.replace(local)
    return True


def source_contract(plan: dict) -> None:
    paths = ((Path(__file__), "runner_sha256"),
             (EVAL, "evaluator_sha256"), (BUNDLE, "a40_environment_bundle_sha256"),
             (ROOT / plan["body"], "body_sha256"),
             (ROOT / plan["bc_checkpoint"], "bc_checkpoint_sha256"),
             (ROOT / plan["baseline_checkpoint"], "baseline_checkpoint_sha256"))
    for path, name in paths:
        if not path.is_file() or sha(path) != plan[name]:
            raise RuntimeError(f"V36 frozen source changed: {name}")
    if (plan["development_seeds"] != [36101, 36102, 36103, 36104]
            or plan["routes_per_seed"] != ["bc", "model25", "scripted"]
            or plan["gpu"] != "NVIDIA A40"):
        raise RuntimeError("V36 development contract changed")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--host", required=True, help="root@SSH host from Runpod")
    parser.add_argument("--port", required=True, type=int)
    parser.add_argument("--key", required=True, type=Path)
    parser.add_argument("--pod-id", required=True)
    args = parser.parse_args()
    if (not args.host.startswith("root@") or not 1 <= args.port <= 65535
            or not args.key.is_file() or (ART / "transport.jsonl").exists()):
        raise RuntimeError("V36 is one-shot and needs the exact active pod SSH identity")
    plan = json.loads(PLAN.read_text())
    source_contract(plan)
    known = ART / "worker-known-hosts"
    log("start", pod_id=args.pod_id, plan_sha256=sha(PLAN), evaluator_sha256=sha(EVAL))
    host, port, key = args.host, args.port, args.key
    remote(host, port, key, known, f"mkdir -p {REMOTE_ROOT} {REMOTE_OUT}")
    send(host, port, key, known, BUNDLE, "/root/v36-base-source.tar.gz")
    remote(host, port, key, known,
           f"test $(sha256sum /root/v36-base-source.tar.gz | cut -d' ' -f1) = {sha(BUNDLE)} && "
           f"tar xzf /root/v36-base-source.tar.gz -C {REMOTE_ROOT}", timeout=180)
    send(host, port, key, known, EVAL, REMOTE_EVAL)
    send(host, port, key, known, PLAN, REMOTE_OUT + "/plan.json")
    actual = remote(host, port, key, known,
                    f"sha256sum {REMOTE_EVAL} | cut -d' ' -f1")
    if actual != sha(EVAL):
        raise RuntimeError("Remote V36 evaluator differs")
    log("source_verified", remote_evaluator_sha256=actual)
    bootstrap = REMOTE_ROOT + "/experiments/compact_morphology/cove50_stand_recovery/bootstrap_seated_runtime_v35_a40_r5.sh"
    remote(host, port, key, known,
           f"MUJOCO_GL=egl NVIDIA_DRIVER_CAPABILITIES=compute,utility,graphics "
           f"timeout 780 bash {bootstrap} > {REMOTE_OUT}/bootstrap.log 2>&1",
           timeout=850)
    receive(host, port, key, known, REMOTE_OUT + "/bootstrap.log", ART / "bootstrap.log")
    gpu = remote(host, port, key, known,
                 "nvidia-smi --query-gpu=name --format=csv,noheader")
    if gpu != "NVIDIA A40":
        raise RuntimeError(f"Wrong GPU: {gpu}")
    log("runtime_ready", gpu=gpu, bootstrap_sha256=sha(ART / "bootstrap.log"))
    for seed in plan["development_seeds"]:
        for route in plan["routes_per_seed"]:
            stem = f"seed{seed}-{route}"
            marker = ART / f"{stem}.attempt.json"
            if marker.exists():
                raise RuntimeError(f"Original already attempted: {stem}")
            marker.write_text(json.dumps({"seed": seed, "route": route,
                                          "pod_id": args.pod_id,
                                          "attempted_at_utc": datetime.now(timezone.utc).isoformat()}) + "\n")
            log("original_start", seed=seed, route=route)
            checkpoint = plan["bc_checkpoint"] if route == "bc" else plan["baseline_checkpoint"]
            prefix = f"{REMOTE_OUT}/{stem}"
            command = (f"test ! -e {prefix}.json && test ! -e {prefix}.npz && "
                       f"test ! -e {prefix}.mp4 && "
                       f"PYTHONPATH={shlex.quote(PYTHONPATH)} MUJOCO_GL=egl "
                       f"NVIDIA_DRIVER_CAPABILITIES=compute,utility,graphics "
                       f"timeout 240 /root/cove50-v31b-venv/bin/python -u {REMOTE_EVAL} "
                       f"{REMOTE_ROOT}/{checkpoint} --out {prefix}.json "
                       f"--native-trajectory-out {prefix}.npz --video {prefix}.mp4 "
                       f"--intervention-route {route} --seed {seed} --device cuda:0 "
                       f"--cove50-body-xml {REMOTE_ROOT}/{plan['body']} --mode cycle "
                       f"--start-squat --recover-from-start --seated-rise-stage "
                       f"--randomized-physics > {prefix}.log 2>&1")
            failure = None
            try:
                remote(host, port, key, known, command, timeout=265)
            except Exception as exc:
                failure = f"{type(exc).__name__}: {exc}"
            received = {}
            for suffix in ("json", "npz", "mp4", "log"):
                path = ART / f"{stem}.{suffix}"
                received[suffix] = receive(host, port, key, known,
                                           f"{prefix}.{suffix}", path)
            if failure or not all(received.values()):
                log("original_failed", seed=seed, route=route,
                    reason=failure, files=received)
                raise RuntimeError(f"V36 original failed: {stem}; preserve and inspect")
            report = json.loads((ART / f"{stem}.json").read_text())
            if (report.get("seed") != seed or report.get("intervention_route") != route
                    or report.get("evaluator_source_sha256") != plan["evaluator_sha256"]
                    or report.get("body_sha256") != plan["body_sha256"]):
                log("original_source_fault", seed=seed, route=route)
                raise RuntimeError(f"V36 source fault: {stem}")
            log("original_complete", seed=seed, route=route,
                steps=len(report["timeseries"]), ended_early=report["ended_early"],
                report_sha256=sha(ART / f"{stem}.json"),
                trace_sha256=sha(ART / f"{stem}.npz"),
                video_sha256=sha(ART / f"{stem}.mp4"))
    log("all_development_originals_complete", families=len(plan["development_seeds"]))


if __name__ == "__main__":
    main()
