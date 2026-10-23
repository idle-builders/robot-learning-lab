"""Collect opened-derived baseline and forced-teacher native RGB trajectories."""

from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(ROOT))
from experiments.compact_morphology.cove50_policy_transfer.run_squat_learned_reach_matrix import gate
from experiments.compact_morphology.cove50_policy_transfer.reach_image_goal import (
    run_v16_opened_matrix as v16,
)

BASE = ROOT / "artifacts/ape-cove50-reach-goal-channels-20261010"
OUT = BASE / "v18"
DEVELOPMENT = OUT / "development-variance-plan.json"
EXECUTION = OUT / "development-execution-plan.json"
BASELINE = OUT / "development-baseline"
TEACHER = OUT / "development-teacher"
TEACHER_EVALUATOR = Path(__file__).with_name("evaluate_teacher_probe_v18.py")
PROBES = {
    "v16_v10_height_locked": ("v16_v10_height_locked", 0.0, 0.0),
    "v10_locked_plus_x2mm": ("v16_v10_locked", 2.0, 0.0),
    "v14_locked_minus_y2mm": ("v16_locked", 0.0, -2.0),
    "v16_live": ("v16_live", 0.0, 0.0),
    "v16_bias_y_locked": ("v16_bias_y_locked", 0.0, 0.0),
    "v16_specialist_locked": ("v16_specialist_locked", 0.0, 0.0),
}


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def frozen_plan() -> dict:
    development = json.loads(DEVELOPMENT.read_text())
    if (development["case_count"] != 48 or len(development["cases"]) != 48 or
            development["role"] != "opened_development_not_disjoint_fresh"):
        raise ValueError("Expected frozen 48-case opened-derived plan")
    if sha(BASE / "v16/fresh-reserved/matrix.json") != development["v16_fresh_matrix_sha256"]:
        raise ValueError("Sealed v16 score changed")
    cases = development["cases"]
    if ({row["assigned_teacher_probe"] for row in cases if row["assigned_teacher_probe"]}
            != set(PROBES) or
            sum(row["assigned_teacher_probe"] is not None for row in cases) != 24):
        raise ValueError("Teacher probes or failures changed")
    return {
        "schema": "robot-learning-lab.cove50-v18-variance-execution-plan.v1",
        "role": "opened_development_native_data_collection_only",
        "development_plan_sha256": sha(DEVELOPMENT),
        "baseline_source_sha256": {name: sha(path) for name, path in v16.SOURCES.items()},
        "teacher_evaluator_sha256": sha(TEACHER_EVALUATOR),
        "runner_sha256": sha(Path(__file__)),
        "probe_contract": {name: {"forced_mode": mode, "extra_body_xy_mm": [x, y]}
                           for name, (mode, x, y) in PROBES.items()},
        "baseline_case_count": 48,
        "teacher_case_count": 24,
        "qualification": "Seventy-two planned native visits are opened-derived development data. Baseline uses observable v16 routing; teacher mode is experimenter-forced and may add a bounded prescribed goal shift. Neither is a new disjoint score.",
    }


def freeze(plan: dict) -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    expected = json.dumps(plan, indent=2).encode() + b"\n"
    if EXECUTION.exists():
        if EXECUTION.read_bytes() != expected:
            raise ValueError("Frozen v18 execution plan differs from current source")
    else:
        EXECUTION.write_bytes(expected)


def baseline_case(case: dict, plan: dict) -> dict:
    # Reuse the exact v16 evaluator, hash checks and full native gate. New
    # seeds are development cases, despite the runner's old origin shortcut.
    v16.DEST = BASELINE
    run_case = {**case, "origin": "v16_fresh_reserved"}
    row = v16.run_case(run_case, {"source_sha256": plan["baseline_source_sha256"]})
    row["origin"] = "v18_opened_derived_development"
    row["origin_seed"] = case["origin_seed"]
    row["role"] = case["role"]
    return row


def teacher_case(case: dict, plan: dict) -> dict:
    seed = case["seed"]
    probe = case["assigned_teacher_probe"]
    mode, bias_x, bias_y = PROBES[probe]
    result, trace, log = (TEACHER / f"seed-{seed}.{suffix}"
                          for suffix in ("json", "npz", "log"))
    sources = v16.SOURCES
    command = [sys.executable, str(TEACHER_EVALUATOR),
               "--body-xml", str(sources["body"]),
               "--calibration-model", str(sources["calibration"]),
               "--squat-checkpoint", str(sources["squat"]),
               "--reach-checkpoint", str(sources["actor"]),
               "--alternate-reach-checkpoint", str(sources["alternate_actor"]),
               "--specialist-reach-checkpoint", str(sources["specialist_actor"]),
               "--force-selection-mode", mode,
               "--extra-goal-bias-body-x-mm", str(bias_x),
               "--extra-goal-bias-body-y-mm", str(bias_y),
               "--reach-controller", "learned-residual",
               "--reach-reference", str(sources["reference"]),
               "--geometry", str(sources["geometry"]),
               "--entry", "continuous", "--seed", str(seed),
               "--rail-v", str(case["rail_v"]), "--delay-ms", str(case["delay_ms"]),
               "--toy-offset-x-m", str(case["toy_offset_x_m"]),
               "--toy-offset-y-m", str(case["toy_offset_y_m"]),
               "--pitch-offset-rad", ".105", "--roll-clearance-rad", "-.25",
               "--terminal-roll-offset-rad", "-.08", "--terminal-roll-mode", "entry-error-x",
               "--reach-duration-s", "9", "--gaze-head-yaw-rad", str(case["gaze_head_yaw_rad"]),
               "--gaze-start-s", "0", "--gaze-ramp-s", "7", "--capture-stride-steps", "5",
               "--max-goal-age-s", "9", "--out", str(result), "--trace", str(trace)]
    if not result.exists():
        env = os.environ.copy()
        env["PYTHONPATH"] = os.pathsep.join(
            [str(ROOT), str(ROOT / "experiments/compact_morphology/mjlab_ape/src"),
             "/workspace/microduck_rl/src"]
            + ([env["PYTHONPATH"]] if env.get("PYTHONPATH") else []))
        with log.open("w") as stream:
            code = subprocess.run(command, cwd=ROOT, env=env, stdout=stream,
                                  stderr=subprocess.STDOUT, check=False).returncode
        if code:
            raise RuntimeError(f"Teacher evaluator failed for seed {seed}: {log}")
    report = json.loads(result.read_text())
    hashes = plan["baseline_source_sha256"]
    if (report["evaluator_sha256"] != plan["teacher_evaluator_sha256"] or
            report["body_sha256"] != hashes["body"] or
            report["selected_pointer_sha256"] != hashes["pointer"] or
            report["reach_checkpoint_sha256"] != hashes["actor"] or
            report["alternate_reach_checkpoint_sha256"] != hashes["alternate_actor"] or
            report["specialist_reach_checkpoint_sha256"] != hashes["specialist_actor"] or
            report["adaptive_variant"] != "v18_opened_derived_teacher_probe" or
            report["selection_mode"] != mode or
            report["extra_goal_bias_body_xy_mm"] != [bias_x, bias_y]):
        raise ValueError(f"V18 teacher identity mismatch: {result}")
    faults = gate(report)
    entry = next((event for event in report["image_goal_capture_events"]
                  if event["time_s"] == 7.5), None)
    if not entry or not entry["used_for_control"] or entry["orange_pixels"] < 10:
        faults.append("no valid RGB goal at entry")
    if entry and (entry.get("xy_error_truth_mm_scoring_only") is None or
                  entry["xy_error_truth_mm_scoring_only"] > 10):
        faults.append("RGB error above 10 mm")
    if report["image_goal_missing_at_s"] is not None:
        faults.append("image goal missing during reach")
    if mode == "v16_live":
        if report["image_goal_lock_after_entry"] or report["image_goal_used_update_count"] < 2:
            faults.append("live RGB updates missing")
    elif not report["image_goal_lock_after_entry"] or report["image_goal_used_update_count"] != 1:
        faults.append("entry RGB goal not locked")
    return {
        "seed": seed, "origin": "v18_opened_derived_development",
        "origin_seed": case["origin_seed"], "role": case["role"],
        "teacher_probe": probe, "selection_mode": mode,
        "goal_bias_body_xy_mm": [bias_x, bias_y - (2 if mode == "v16_bias_y_locked" else 0)],
        "rail_v": case["rail_v"], "delay_ms": case["delay_ms"],
        "toy_offset_b_m": [case["toy_offset_x_m"], case["toy_offset_y_m"]],
        "pass": not faults,
        "contact_s": report["phases"]["reach"]["continuous_finger_contact_s"],
        "entry_goal_error_mm_scoring_only": None if entry is None else entry.get("xy_error_truth_mm_scoring_only"),
        "reach_both_feet_fraction": report["phases"]["reach"]["both_feet_fraction"],
        "reach_rated_leg_fraction": report["phases"]["reach"]["rated_leg_fraction"],
        "faults": faults,
        "result": str(result.relative_to(ROOT)),
        "trace": str(trace.relative_to(ROOT)),
        "log": str(log.relative_to(ROOT)),
    }


def run_stage(stage: str, cases: list[dict], plan: dict, workers: int) -> None:
    dest = BASELINE if stage == "baseline" else TEACHER
    dest.mkdir(parents=True, exist_ok=True)
    executor = baseline_case if stage == "baseline" else teacher_case
    rows = {}
    with ThreadPoolExecutor(max_workers=workers) as pool:
        futures = {pool.submit(executor, case, plan): case for case in cases}
        for future in as_completed(futures):
            case = futures[future]
            row = future.result()
            rows[case["seed"]] = row
            ordered = [rows[case["seed"]] for case in cases if case["seed"] in rows]
            matrix = {
                "schema": f"robot-learning-lab.cove50-v18-variance-{stage}-native.v1",
                "role": "opened_derived_development_not_fresh_score",
                "execution_plan_sha256": sha(EXECUTION),
                "complete": len(ordered) == len(cases),
                "case_count": len(ordered), "pass_count": sum(row["pass"] for row in ordered),
                "cases": ordered,
            }
            (dest / "matrix.json").write_text(json.dumps(matrix, indent=2) + "\n")
            print(json.dumps({"stage": stage, **row}), flush=True)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--freeze-only", action="store_true")
    parser.add_argument("--stage", choices=("baseline", "teacher", "both"), default="both")
    parser.add_argument("--workers", type=int, default=4)
    args = parser.parse_args()
    if not 1 <= args.workers <= 6:
        raise ValueError("Use one to six independent native evaluators")
    plan = frozen_plan()
    freeze(plan)
    if args.freeze_only:
        print(f"frozen {EXECUTION} sha256={sha(EXECUTION)}")
        return
    cases = json.loads(DEVELOPMENT.read_text())["cases"]
    if args.stage in ("baseline", "both"):
        run_stage("baseline", cases, plan, args.workers)
    if args.stage in ("teacher", "both"):
        teacher = [case for case in cases if case["assigned_teacher_probe"] is not None]
        run_stage("teacher", teacher, plan, args.workers)


if __name__ == "__main__":
    main()
