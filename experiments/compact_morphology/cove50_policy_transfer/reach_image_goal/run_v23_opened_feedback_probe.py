"""Check whether the saved observed-RGB PI teacher can rescue opened V21 misses."""

from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(ROOT))
from experiments.compact_morphology.cove50_policy_transfer.run_squat_learned_reach_matrix import gate


BASE = ROOT / "artifacts/ape-cove50-reach-goal-channels-20261010"
V21 = BASE / "v21"
OUT = BASE / "v23"
PLAN = OUT / "opened-feedback-plan.json"
DEST = OUT / "opened-feedback"
SCORED = V21 / "fresh-reserved/matrix.json"
FRESH_PLAN = V21 / "fresh-challenge-plan.json"
EVALUATOR = Path(__file__).with_name("evaluate_calibrated_goal.py")
SEEDS = (10604, 10601, 10615, 10613)
SOURCES = {
    "body": ROOT / "artifacts/ape-cove50-transfer-20261008/cove50-direct-training.xml",
    "squat": ROOT / "artifacts/ape-cove50-squat-recovery-20261009/squat-teacher-bc-blendmask.pt",
    "teacher": ROOT / "artifacts/ape-cove50-side-reach-20261009/reach-feedback-v1/controller.pt",
    "reference": ROOT / "artifacts/ape-cove50-hand-candidate-mjlab-20261010/reach-entry/development-selected.npz",
    "geometry": ROOT / "artifacts/ape-cove50-side-reach-20261009/geometry-center.json",
    "calibration": BASE / "rgb-calibration-model.npz",
    "evaluator": EVALUATOR,
    "pointer": ROOT / "robot/ape-current-design.json",
}


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def capture(report: dict) -> dict:
    return next(event for event in report["image_goal_capture_events"]
                if event["time_s"] == 7.5)


def frozen_plan() -> dict:
    scored = json.loads(SCORED.read_text())
    fresh = json.loads(FRESH_PLAN.read_text())
    if (not scored["complete"] or scored["case_count"] != 24 or
            scored["pass_count"] != 22 or scored["promotion_gate_pass"] or
            scored["plan_sha256"] != sha(FRESH_PLAN)):
        raise ValueError("Need sealed failed V21 fresh score")
    rows = {row["seed"]: row for row in scored["cases"]}
    if [rows[seed]["pass"] for seed in SEEDS] != [False, True, False, True]:
        raise ValueError("Failure/control roles changed")
    cases = {case["seed"]: case for case in fresh["cases"]}
    pointer = json.loads(SOURCES["pointer"].read_text())
    if pointer["status"] != "rejected_no_selected_mechanical_design":
        raise ValueError("Historical rejected-body diagnostic only")
    return {
        "schema": "robot-learning-lab.cove50-v23-opened-feedback-plan.v1",
        "role": "postscore_opened_saved_feedback_diagnostic_not_automatic_or_fresh",
        "body_role": "rejected_historical_direct_frame_sandbox",
        "v21_fresh_plan_sha256": sha(FRESH_PLAN),
        "v21_fresh_matrix_sha256": sha(SCORED),
        "source_sha256": {name: sha(path) for name, path in SOURCES.items()},
        "runner_sha256": sha(Path(__file__)),
        "paired_pre_reach_tolerance": {"qpos_max_abs": 5e-5,
                                        "action_max_abs": 1e-3,
                                        "rgb_goal_max_abs_m": 5e-4},
        "cases": [{**cases[seed], "baseline_result": rows[seed]["result"],
                   "baseline_pass": rows[seed]["pass"],
                   "baseline_contact_s": rows[seed]["contact_s"]}
                  for seed in SEEDS],
        "qualification": "The experimenter chooses a saved observed-RGB feedback controller on two opened failures and two passing controls. It can diagnose actor versus observation limitations but cannot alter the sealed V21 score, qualify an automatic router, or qualify a new body.",
    }


def freeze(plan: dict) -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    expected = (json.dumps(plan, indent=2) + "\n").encode()
    if PLAN.exists():
        if PLAN.read_bytes() != expected:
            raise ValueError("Frozen V23 feedback plan differs")
    else:
        PLAN.write_bytes(expected)


def run_case(case: dict, plan: dict) -> dict:
    seed = case["seed"]
    result, trace, log = (DEST / f"seed-{seed}.{suffix}"
                          for suffix in ("json", "npz", "log"))
    command = [sys.executable, str(EVALUATOR),
               "--body-xml", str(SOURCES["body"]),
               "--calibration-model", str(SOURCES["calibration"]),
               "--squat-checkpoint", str(SOURCES["squat"]),
               "--reach-checkpoint", str(SOURCES["teacher"]),
               "--reach-controller", "feedback",
               "--reach-reference", str(SOURCES["reference"]),
               "--geometry", str(SOURCES["geometry"]),
               "--entry", "continuous", "--seed", str(seed),
               "--rail-v", str(case["rail_v"]),
               "--delay-ms", str(case["delay_ms"]),
               "--toy-offset-x-m", str(case["toy_offset_x_m"]),
               "--toy-offset-y-m", str(case["toy_offset_y_m"]),
               "--pitch-offset-rad", ".105", "--roll-clearance-rad", "-.25",
               "--terminal-roll-offset-rad", "-.08",
               "--terminal-roll-mode", "entry-error-x",
               "--reach-duration-s", "9",
               "--gaze-head-yaw-rad", str(case["gaze_head_yaw_rad"]),
               "--gaze-start-s", "0", "--gaze-ramp-s", "7",
               "--capture-stride-steps", "5", "--max-goal-age-s", "9",
               "--lock-goal-after-entry", "--out", str(result),
               "--trace", str(trace)]
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
            raise RuntimeError(f"Feedback evaluator failed for {seed}: {log}")
    report = json.loads(result.read_text())
    baseline_path = ROOT / case["baseline_result"]
    baseline = json.loads(baseline_path.read_text())
    with np.load(trace) as new_trace, np.load(baseline_path.with_suffix(".npz")) as old_trace:
        if (new_trace["qpos"].shape != old_trace["qpos"].shape or
                new_trace["action"].shape != old_trace["action"].shape):
            raise ValueError(f"Paired trace shape changed: {seed}")
        qpos_delta = float(np.max(np.abs(new_trace["qpos"][375] - old_trace["qpos"][375])))
        action_delta = float(np.max(np.abs(new_trace["action"][:375] - old_trace["action"][:375])))
    entry = capture(report)
    old_entry = capture(baseline)
    goal_delta = float(np.max(np.abs(np.asarray(entry["estimated_goal_world_xyz_m"][:2]) -
                                    np.asarray(old_entry["estimated_goal_world_xyz_m"][:2]))))
    tolerance = plan["paired_pre_reach_tolerance"]
    if (report["evaluator_sha256"] != plan["source_sha256"]["evaluator"] or
            report["body_sha256"] != plan["source_sha256"]["body"] or
            report["selected_pointer_sha256"] != plan["source_sha256"]["pointer"] or
            report["reach_checkpoint_sha256"] != plan["source_sha256"]["teacher"] or
            report["reach_controller"] != "feedback" or
            report["boundary_state_sha256"]["posture"] != baseline["boundary_state_sha256"]["posture"] or
            report["first_action_sha256"] != baseline["first_action_sha256"] or
            qpos_delta > tolerance["qpos_max_abs"] or
            action_delta > tolerance["action_max_abs"] or
            goal_delta > tolerance["rgb_goal_max_abs_m"]):
        raise ValueError(f"Feedback source or paired entry mismatch: {seed}")
    faults = gate(report)
    if (not entry["used_for_control"] or entry["orange_pixels"] < 10 or
            entry.get("xy_error_truth_mm_scoring_only") is None or
            entry["xy_error_truth_mm_scoring_only"] > 10 or
            report["image_goal_missing_at_s"] is not None or
            not report["image_goal_lock_after_entry"] or
            report["image_goal_used_update_count"] != 1):
        faults.append("invalid locked RGB goal")
    return {
        "seed": seed, "intended_stratum": case["intended_stratum"],
        "baseline_pass": case["baseline_pass"],
        "baseline_contact_s": case["baseline_contact_s"],
        "pass": not faults,
        "contact_s": report["phases"]["reach"]["continuous_finger_contact_s"],
        "entry_goal_error_mm_scoring_only": entry["xy_error_truth_mm_scoring_only"],
        "paired_reach_entry_qpos_max_abs": qpos_delta,
        "paired_pre_reach_action_max_abs": action_delta,
        "paired_rgb_goal_delta_m": goal_delta,
        "faults": faults,
        "result": str(result.relative_to(ROOT)),
        "trace": str(trace.relative_to(ROOT)),
        "log": str(log.relative_to(ROOT)),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--freeze-only", action="store_true")
    parser.add_argument("--workers", type=int, default=4)
    args = parser.parse_args()
    if not 1 <= args.workers <= 4:
        raise ValueError("Use one to four native evaluators")
    plan = frozen_plan()
    freeze(plan)
    if args.freeze_only:
        print(f"frozen {PLAN} sha256={sha(PLAN)}")
        return
    DEST.mkdir(parents=True, exist_ok=True)
    rows = {}
    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        futures = {pool.submit(run_case, case, plan): case for case in plan["cases"]}
        for future in as_completed(futures):
            row = future.result()
            rows[row["seed"]] = row
            ordered = [rows[case["seed"]] for case in plan["cases"]
                       if case["seed"] in rows]
            matrix = {
                "schema": "robot-learning-lab.cove50-v23-opened-feedback-native.v1",
                "role": "postscore_opened_saved_feedback_diagnostic_not_automatic_or_fresh",
                "plan_sha256": sha(PLAN),
                "complete": len(ordered) == len(plan["cases"]),
                "case_count": len(ordered),
                "pass_count": sum(item["pass"] for item in ordered),
                "rescues": [item["seed"] for item in ordered
                            if item["pass"] and not item["baseline_pass"]],
                "regressions": [item["seed"] for item in ordered
                                if not item["pass"] and item["baseline_pass"]],
                "cases": ordered,
            }
            (DEST / "matrix.json").write_text(json.dumps(matrix, indent=2) + "\n")
            print(json.dumps(row), flush=True)


if __name__ == "__main__":
    main()
