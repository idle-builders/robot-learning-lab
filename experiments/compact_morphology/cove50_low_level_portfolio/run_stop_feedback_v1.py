"""Run frozen native baseline/residual pairs on disjoint whole starts."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys

import numpy as np


ROOT = Path(__file__).resolve().parents[3]
BASE = ROOT / "artifacts/ape-low-level-training-20261015/modular-stop-feedback-prep"
BASELINE = Path(__file__).with_name("evaluate_portfolio_breadth_v2.py")
CANDIDATE = Path(__file__).with_name("evaluate_stop_feedback_v1.py")


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--score-freeze", type=Path, default=BASE / "candidate-score-freeze.json")
    ap.add_argument("--training-report", type=Path, required=True)
    ap.add_argument("--residual-actor", type=Path, required=True)
    ap.add_argument("--start", type=int, default=0)
    ap.add_argument("--limit", type=int, default=16)
    args = ap.parse_args()
    freeze = json.loads(args.score_freeze.read_text())
    if freeze.get("schema") != "robot-learning-lab.modular-stop-feedback-score-freeze.v1":
        raise ValueError("candidate score freeze schema changed")
    sources = {"score_plan": BASE / "fresh-score-plan.json", "baseline": BASELINE,
               "candidate": CANDIDATE, "runner": Path(__file__),
               "trainer": Path(__file__).with_name("train_stop_feedback_v1.py"),
               "training_env": Path(__file__).with_name("stop_feedback_env.py"),
               "training_rail": Path(__file__).with_name("stop_feedback_rail.py"),
               "training_reward": Path(__file__).with_name("stop_feedback_reward.py"),
               "batched_router": Path(__file__).with_name("batched_controller.py")}
    if any(digest(path) != freeze["sha256"][name] for name, path in sources.items()):
        raise ValueError("frozen score source changed")
    plan = json.loads(sources["score_plan"].read_text())
    method_path = ROOT / plan["source_method"]
    if digest(method_path) != plan["source_method_sha256"]:
        raise ValueError("frozen V2 source method changed")
    method = json.loads(method_path.read_text())
    assets = method["assets"]
    for name, expected in method["source_sha256"].items():
        if name in method["sources"] and digest(ROOT / method["sources"][name]) != expected:
            raise ValueError(f"source changed: {name}")
    training = json.loads(args.training_report.read_text())
    if (training.get("status") != "completed" or
            args.residual_actor.resolve() not in
            {Path(path).resolve() for path in training.get("checkpoint_paths", [])}):
        raise ValueError("candidate checkpoint lacks a completed source training report")
    train_names = {"trainer": "train_stop_feedback_v1.py",
                   "training_env": "stop_feedback_env.py",
                   "training_rail": "stop_feedback_rail.py",
                   "training_reward": "stop_feedback_reward.py",
                   "batched_router": "batched_controller.py"}
    if any(training.get("training_source_sha256", {}).get(filename) != freeze["sha256"][name]
           for name, filename in train_names.items()):
        raise ValueError("completed candidate was fitted under a different frozen source")
    if not 0 <= args.start < len(plan["cases"]) or not 1 <= args.limit <= len(plan["cases"]):
        ap.error("invalid case slice")
    out = BASE / "fresh-score-native"
    out.mkdir(parents=True, exist_ok=True)
    run_manifest = {"schema": "robot-learning-lab.modular-stop-feedback-score-run.v1",
                    "training_report_sha256": digest(args.training_report),
                    "candidate_checkpoint_sha256": digest(args.residual_actor),
                    "score_freeze_sha256": digest(args.score_freeze)}
    run_manifest_path = out / "score-run-manifest.json"
    if run_manifest_path.exists():
        if json.loads(run_manifest_path.read_text()) != run_manifest:
            raise ValueError("score run source manifest changed")
    else:
        run_manifest_path.write_text(json.dumps(run_manifest, indent=2) + "\n")
    env = os.environ.copy()
    env.update({"MUJOCO_GL": "egl", "OMP_NUM_THREADS": "1", "OPENBLAS_NUM_THREADS": "1",
                "PYTHONPATH": "experiments/compact_morphology/mjlab_ape/src"})
    for case in plan["cases"][args.start:args.start + args.limit]:
        common = ["--body", assets["body"], "--gait-actor", assets["gait"],
                  "--stop-actor", assets["stop"], "--squat-actor", assets["squat"],
                  "--arm-actor", assets["arms"], "--seed", str(case["seed"]),
                  "--rail-v", str(case["rail_v"]), "--delay-ms", str(case["delay_ms"]),
                  "--walk-velocity", str(case["speed_m_s"]), "--walk-s", str(case["walk_s"]),
                  "--stop-s", str(case["stop_s"]), "--squat-start-s", str(case["squat_start_s"]),
                  "--squat-s", str(case["squat_s"]), "--arm-start-s", str(case["arm_start_s"]),
                  "--arm-s", str(case["arm_s"]), "--sole-friction", str(case["sole_friction"]),
                  "--device", "cuda:0", "--native-evaluation"]
        for label, script, extra in (
            ("baseline", BASELINE, []),
            ("candidate", CANDIDATE,
             ["--residual-actor", str(args.residual_actor),
              "--training-report", str(args.training_report),
              "--score-plan", str(sources["score_plan"]),
              "--residual-gain", str(training["residual_gain"])]),
        ):
            stem = out / f"seed-{case['seed']}-{label}"
            paths = {ext: stem.with_suffix(ext) for ext in (".json", ".npz", ".mp4", ".log")}
            if any(path.exists() for path in paths.values()):
                if not all(path.exists() for path in paths.values()):
                    raise FileExistsError(f"Incomplete original visit needs an audit: {stem}")
                prior = json.loads(paths[".json"].read_text())
                if prior.get("seed") != case["seed"] or prior.get("steps") != 1200:
                    raise ValueError(f"Existing original visit needs an audit: {stem}")
                print(json.dumps({"seed": case["seed"], "route": label,
                                  "status": "existing_original_preserved"}), flush=True)
                continue
            cmd = [sys.executable, str(script), *common, *extra,
                   "--out", str(paths[".json"]), "--trace", str(paths[".npz"]),
                   "--video", str(paths[".mp4"])]
            with paths[".log"].open("w") as log:
                done = subprocess.run(cmd, cwd=ROOT, env=env, stdout=log,
                                      stderr=subprocess.STDOUT, check=False)
            print(json.dumps({"seed": case["seed"], "route": label,
                              "exit_code": done.returncode,
                              "report_exists": paths[".json"].exists()}), flush=True)
            if done.returncode:
                raise RuntimeError(f"Native visit failed; inspect {paths['.log']}")
        pre_stop_steps = round(case["walk_s"] / 0.02)
        traces = {}
        for label in ("baseline", "candidate"):
            path = out / f"seed-{case['seed']}-{label}.npz"
            with np.load(path, allow_pickle=False) as source:
                traces[label] = {key: source[key][:pre_stop_steps].copy() for key in
                                 ("pre_qpos", "pre_qvel", "applied_action")}
                traces[label]["initial_qpos"] = source["initial_qpos"].copy()
                traces[label]["initial_qvel"] = source["initial_qvel"].copy()
        gaps = {key: float(np.max(np.abs(traces["baseline"][key] - traces["candidate"][key])))
                for key in traces["baseline"]}
        check = {"schema": "robot-learning-lab.modular-stop-feedback-pair-prefix.v1",
                 "seed": case["seed"], "pre_stop_steps": pre_stop_steps,
                 "max_abs_gaps": gaps,
                 "pass": (max(gaps["initial_qpos"], gaps["initial_qvel"]) <= 1e-4
                 and max(gaps["pre_qpos"], gaps["pre_qvel"], gaps["applied_action"]) <= 1e-5)}
        check_path = out / f"seed-{case['seed']}-prefix-check.json"
        if check_path.exists():
            if json.loads(check_path.read_text()) != check:
                raise ValueError(f"Existing prefix check differs: {check_path}")
        else:
            check_path.write_text(json.dumps(check, indent=2) + "\n")
        if not check["pass"]:
            raise ValueError(f"Matched pre-stop prefix failed for seed {case['seed']}: {gaps}")


if __name__ == "__main__":
    main()
