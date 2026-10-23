"""Run frozen opened-seed CUDA critic/action-switch diagnostic, without fitting."""

from __future__ import annotations

import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import subprocess
import sys
import time


ROOT = Path(__file__).resolve().parents[3]
ART = ROOT / "artifacts/ape-cove50-stand-recovery/seated-intervention-v29"
V25 = ROOT / "artifacts/ape-cove50-stand-recovery/seated-rise-ppo-pilot-v25"
V26 = ROOT / "artifacts/ape-cove50-stand-recovery/seated-reference-cuda-v26"
BODY = ROOT / "artifacts/ape-cove50-transfer-20261008/cove50-direct-training.xml"
CHECKPOINT = V25 / "checkpoints/model_25.pt"
EVALUATOR = ROOT / "experiments/compact_morphology/cove50_stand_recovery/evaluate_seated_intervention_v29.py"
SCORER = ROOT / "experiments/compact_morphology/cove50_stand_recovery/score_seated_intervention_v29.py"
V26_SCORER = ROOT / "experiments/compact_morphology/cove50_stand_recovery/score_seated_reference_cuda_v26.py"


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def verify_sources(plan: dict) -> None:
    source = plan["source_sha256"]
    paths = {
        "body": BODY,
        "checkpoint": CHECKPOINT,
        "v25_plan": V25 / "plan.json",
        "v26_plan": V26 / "plan.json",
        "v26_report": V26 / "report.json",
        "trainer": ROOT / "experiments/compact_morphology/cove50_policy_transfer/train_posture_arm.py",
        "task": ROOT / "experiments/compact_morphology/mjlab_ape/src/ape_walk/task.py",
        "frame50": ROOT / "experiments/compact_morphology/mjlab_ape/src/ape_walk/physics/frame50.py",
        "evaluator": EVALUATOR,
        "scorer": SCORER,
        "v26_scorer": V26_SCORER,
        "runner": Path(__file__),
    }
    for name, path in paths.items():
        if sha(path) != source[name]:
            raise ValueError(f"V29 source drift: {name}")
    import mjlab_microduck.tasks.mdp as upstream_mdp
    if sha(Path(upstream_mdp.__file__)) != source["upstream_mdp"]:
        raise ValueError("V29 upstream MDP source drift")
    if not sys.version.startswith(plan["runtime_versions"]["python"]):
        raise ValueError("V29 Python runtime drift")
    for package, expected in plan["runtime_versions"].items():
        if package != "python" and importlib.metadata.version(package) != expected:
            raise ValueError(f"V29 runtime drift: {package}")
    if plan["opened_development_seeds"] != [2231, 2232, 2233, 2234, 2235, 2236]:
        raise ValueError("V29 may use only the six opened V25 development starts")
    if plan["routes"] != ["model25", "scripted", "script-first-0p2", "actor-first-0p1", "actor-first-0p2"]:
        raise ValueError("V29 route contract changed")


def run(seed: int, route: str, timeout_s: int) -> dict:
    stem = ART / f"seed{seed}-{route}"
    report_path, native_path = stem.with_suffix(".json"), stem.with_suffix(".npz")
    if report_path.exists() and native_path.exists():
        report = json.loads(report_path.read_text())
        if (report.get("device") == "cuda:0" and report.get("seed") == seed
                and report.get("intervention_route") == route
                and 0 < report.get("duration_s", 0) <= 10.0
                and (not report.get("ended_early") or report.get("end_termination_terms"))):
            return report
        raise ValueError(f"{seed} {route}: existing source is incomplete or mismatched")
    if report_path.exists() or native_path.exists():
        raise ValueError(f"{seed} {route}: incomplete prior visit needs inspection")
    command = [
        sys.executable, str(EVALUATOR), str(CHECKPOINT),
        "--out", str(report_path), "--native-trajectory-out", str(native_path),
        "--intervention-route", route, "--seed", str(seed), "--device", "cuda:0",
        "--cove50-body-xml", str(BODY), "--mode", "cycle",
        "--start-squat", "--recover-from-start", "--seated-rise-stage", "--randomized-physics",
    ]
    env = os.environ.copy()
    env["PYTHONPATH"] = f"{ROOT / 'experiments/compact_morphology/mjlab_ape/src'}:{ROOT}"
    with stem.with_suffix(".log").open("w") as log:
        completed = subprocess.run(command, cwd=ROOT, env=env, stdout=log,
                                   stderr=subprocess.STDOUT, timeout=timeout_s, check=False)
    if completed.returncode:
        raise RuntimeError(f"{seed} {route}: native evaluator exited {completed.returncode}")
    report = json.loads(report_path.read_text())
    if (report["device"] != "cuda:0" or not 0 < report["duration_s"] <= 10.0
            or (report["ended_early"] and not report["end_termination_terms"])):
        raise ValueError(f"{seed} {route}: invalid native visit")
    return report


def main() -> None:
    plan_path = ART / "plan.json"
    plan = json.loads(plan_path.read_text())
    verify_sources(plan)
    ART.mkdir(parents=True, exist_ok=True)
    started = time.monotonic()
    for seed in plan["opened_development_seeds"]:
        for route in plan["routes"]:
            remaining = plan["max_worker_wall_s"] - (time.monotonic() - started)
            if remaining < 1:
                raise TimeoutError("V29 worker wall cap reached")
            report = run(seed, route, min(plan["max_route_wall_s"], int(remaining)))
            print(json.dumps({
                "seed": seed, "route": route, "duration_s": report["duration_s"],
                "initial_state_sha256": report["initial_state_sha256"],
                "report_sha256": sha(ART / f"seed{seed}-{route}.json"),
                "native_sha256": sha(ART / f"seed{seed}-{route}.npz"),
            }), flush=True)
            if route == "scripted" or route in plan["routes"][2:]:
                # Source-score each family before its first switch, then each
                # intervention immediately; stop on source/physics/action drift.
                baseline_env = os.environ.copy()
                baseline_env["PYTHONPATH"] = str(ROOT)
                check_args = (["--baseline-seed", str(seed)] if route == "scripted" else
                              ["--partial-seed", str(seed), "--partial-route", route])
                completed = subprocess.run(
                    [sys.executable, str(SCORER), *check_args],
                    cwd=ROOT, env=baseline_env, capture_output=True, text=True,
                    timeout=min(120, max(1, int(plan["max_worker_wall_s"] -
                                            (time.monotonic() - started)))),
                    check=False,
                )
                if completed.returncode:
                    raise RuntimeError(
                        f"{seed} {route}: immediate source/physical parity failed: "
                        f"{completed.stdout[-2000:]} {completed.stderr[-2000:]}"
                    )
                print(completed.stdout.strip(), flush=True)


if __name__ == "__main__":
    main()
