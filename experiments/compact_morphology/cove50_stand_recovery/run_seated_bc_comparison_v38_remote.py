"""Run V37 versus V35 BC on four fresh matched A40 development starts.

The provider create and independent hard-stop timer are separate operations.
Each original is copied before the next; a failure consumes its route/start.
"""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import shlex

from experiments.compact_morphology.cove50_stand_recovery import run_seated_bc_pilot_v36_remote as transport


ROOT = Path(__file__).resolve().parents[3]
ART = ROOT / "artifacts/ape-cove50-stand-recovery/seated-bc-comparison-v38"
REMOTE_ROOT = "/root/robot-learning-lab"
REMOTE_OUT = "/workspace/robot-learning-lab/seated-bc-comparison-v38"
PYTHONPATH = REMOTE_ROOT + "/experiments/compact_morphology/mjlab_ape/src:" + REMOTE_ROOT
EVAL35 = ROOT / "experiments/compact_morphology/cove50_stand_recovery/evaluate_seated_bc_pilot_v36.py"
EVAL37 = ROOT / "experiments/compact_morphology/cove50_stand_recovery/evaluate_seated_bc_comparison_v38.py"
BUNDLE = ROOT / "artifacts/ape-cove50-stand-recovery/seated-bc-eval-v35-a40-r5/worker-source.tar.gz"


def log(event: str, **fields) -> None:
    with (ART / "transport.jsonl").open("a") as output:
        output.write(json.dumps({"at_utc": datetime.now(timezone.utc).isoformat(),
                                 "event": event, **fields}, sort_keys=True) + "\n")


def source_contract(plan: dict) -> None:
    paths = ((Path(__file__), "runner_sha256"),
             (Path(transport.__file__), "transport_helper_sha256"),
             (EVAL35, "v35_evaluator_sha256"),
             (EVAL37, "v37_evaluator_sha256"), (BUNDLE, "a40_bundle_sha256"),
             (ROOT / plan["body"], "body_sha256"),
             (ROOT / plan["v35_checkpoint"], "v35_checkpoint_sha256"),
             (ROOT / plan["v37_checkpoint"], "v37_checkpoint_sha256"),
             (ROOT / plan["script_checkpoint"], "script_checkpoint_sha256"),
             (ROOT / plan["v37_fit_report"], "v37_fit_report_sha256"))
    for path, name in paths:
        if not path.is_file() or transport.sha(path) != plan[name]:
            raise RuntimeError(f"V38 frozen source changed: {name}")
    if (plan["status"] != "frozen_before_paid_development" or
            plan["development_seeds"] != [36105, 36106, 36107, 36108] or
            plan["routes_per_seed"] != ["bc37", "bc35", "scripted"] or
            plan["gpu"] != "NVIDIA A40"):
        raise RuntimeError("V38 method differs from freeze")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--host", required=True)
    parser.add_argument("--port", type=int, required=True)
    parser.add_argument("--key", type=Path, required=True)
    parser.add_argument("--pod-id", required=True)
    args = parser.parse_args()
    if (not args.host.startswith("root@") or not args.key.is_file() or
            not 1 <= args.port <= 65535 or (ART / "transport.jsonl").exists()):
        raise RuntimeError("V38 is one-shot and needs exact live pod identity")
    plan = json.loads((ART / "plan.json").read_text())
    source_contract(plan)
    host, port, key = args.host, args.port, args.key
    known = ART / "worker-known-hosts"
    remote = transport.remote
    send = transport.send
    receive = transport.receive
    log("start", pod_id=args.pod_id, plan_sha256=transport.sha(ART / "plan.json"))
    remote(host, port, key, known,
           f"mkdir -p {REMOTE_ROOT} {REMOTE_OUT} "
           f"{REMOTE_ROOT}/artifacts/ape-cove50-stand-recovery/seated-bc-v37")
    send(host, port, key, known, BUNDLE, "/root/v38-base-source.tar.gz")
    remote(host, port, key, known,
           f"test $(sha256sum /root/v38-base-source.tar.gz | cut -d' ' -f1) = {transport.sha(BUNDLE)} && "
           f"tar xzf /root/v38-base-source.tar.gz -C {REMOTE_ROOT}", timeout=180)
    remote35 = f"{REMOTE_ROOT}/{EVAL35.relative_to(ROOT)}"
    remote37 = f"{REMOTE_ROOT}/{EVAL37.relative_to(ROOT)}"
    send(host, port, key, known, EVAL35, remote35)
    send(host, port, key, known, EVAL37, remote37)
    send(host, port, key, known, ROOT / plan["v37_checkpoint"],
         f"{REMOTE_ROOT}/{plan['v37_checkpoint']}")
    send(host, port, key, known, ART / "plan.json", REMOTE_OUT + "/plan.json")
    actual = remote(host, port, key, known,
                    f"sha256sum {remote35} {remote37} "
                    f"{REMOTE_ROOT}/{plan['v37_checkpoint']}")
    for expected in (plan["v35_evaluator_sha256"], plan["v37_evaluator_sha256"],
                     plan["v37_checkpoint_sha256"]):
        if expected not in actual:
            raise RuntimeError("Remote source or V37 checkpoint differs")
    log("source_verified", remote_shas=actual.splitlines())
    bootstrap = (f"{REMOTE_ROOT}/experiments/compact_morphology/cove50_stand_recovery/"
                 "bootstrap_seated_runtime_v35_a40_r5.sh")
    remote(host, port, key, known,
           f"MUJOCO_GL=egl NVIDIA_DRIVER_CAPABILITIES=compute,utility,graphics "
           f"timeout 780 bash {bootstrap} > {REMOTE_OUT}/bootstrap.log 2>&1", timeout=850)
    receive(host, port, key, known, REMOTE_OUT + "/bootstrap.log", ART / "bootstrap.log")
    gpu = remote(host, port, key, known, "nvidia-smi --query-gpu=name --format=csv,noheader")
    if gpu != "NVIDIA A40":
        raise RuntimeError(f"Wrong GPU: {gpu}")
    log("runtime_ready", gpu=gpu, bootstrap_sha256=transport.sha(ART / "bootstrap.log"))
    for seed in plan["development_seeds"]:
        for route in plan["routes_per_seed"]:
            stem = f"seed{seed}-{route}"
            marker = ART / f"{stem}.attempt.json"
            if marker.exists():
                raise RuntimeError(f"Original already attempted: {stem}")
            marker.write_text(json.dumps({"seed": seed, "route": route,
                                          "pod_id": args.pod_id,
                                          "at_utc": datetime.now(timezone.utc).isoformat()}) + "\n")
            log("original_start", seed=seed, route=route)
            evaluator = remote37 if route == "bc37" else remote35
            eval_route = "scripted" if route == "scripted" else "bc"
            checkpoint = (plan["v37_checkpoint"] if route == "bc37" else
                          plan["v35_checkpoint"] if route == "bc35" else
                          plan["script_checkpoint"])
            prefix = f"{REMOTE_OUT}/{stem}"
            command = (f"test ! -e {prefix}.json && test ! -e {prefix}.npz && "
                       f"test ! -e {prefix}.mp4 && "
                       f"PYTHONPATH={shlex.quote(PYTHONPATH)} MUJOCO_GL=egl "
                       f"NVIDIA_DRIVER_CAPABILITIES=compute,utility,graphics "
                       f"timeout 240 /root/cove50-v31b-venv/bin/python -u {evaluator} "
                       f"{REMOTE_ROOT}/{checkpoint} --out {prefix}.json "
                       f"--native-trajectory-out {prefix}.npz --video {prefix}.mp4 "
                       f"--intervention-route {eval_route} --seed {seed} --device cuda:0 "
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
                received[suffix] = receive(host, port, key, known,
                                           f"{prefix}.{suffix}", ART / f"{stem}.{suffix}")
            if failure or not all(received.values()):
                log("original_failed", seed=seed, route=route,
                    reason=failure, files=received)
                raise RuntimeError(f"V38 original failed: {stem}; preserve and inspect")
            report = json.loads((ART / f"{stem}.json").read_text())
            evaluator_sha = (plan["v37_evaluator_sha256"] if route == "bc37"
                             else plan["v35_evaluator_sha256"])
            checkpoint_sha = (plan["v37_checkpoint_sha256"] if route == "bc37"
                              else plan["v35_checkpoint_sha256"] if route == "bc35"
                              else plan["script_checkpoint_sha256"])
            if (report.get("seed") != seed or report.get("intervention_route") != eval_route
                    or report.get("evaluator_source_sha256") != evaluator_sha
                    or report.get("checkpoint_sha256") != checkpoint_sha
                    or report.get("body_sha256") != plan["body_sha256"]):
                log("original_source_fault", seed=seed, route=route)
                raise RuntimeError(f"V38 source fault: {stem}")
            log("original_complete", seed=seed, route=route,
                steps=len(report["timeseries"]), ended_early=report["ended_early"],
                report_sha256=transport.sha(ART / f"{stem}.json"),
                trace_sha256=transport.sha(ART / f"{stem}.npz"),
                video_sha256=transport.sha(ART / f"{stem}.mp4"))
    log("all_development_originals_complete", families=len(plan["development_seeds"]))


if __name__ == "__main__":
    main()
