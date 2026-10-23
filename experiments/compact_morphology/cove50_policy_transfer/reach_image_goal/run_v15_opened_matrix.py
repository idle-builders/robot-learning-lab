"""Gate the v15 RGB/arm candidate on all thirty now-opened native starts."""

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

BASE = ROOT / "artifacts/ape-cove50-reach-goal-channels-20261010"
V14 = BASE / "v14"
V15 = BASE / "v15"
DEST = V15 / "opened-native"
EVALUATOR = Path(__file__).with_name("evaluate_adaptive_goal_v15.py")
SOURCES = {
    "body": ROOT / "artifacts/ape-cove50-transfer-20261008/cove50-direct-training.xml",
    "squat": ROOT / "artifacts/ape-cove50-squat-recovery-20261009/squat-teacher-bc-blendmask.pt",
    "actor": V15 / "actor-locked.pt",
    "alternate_actor": BASE / "v10/actor.pt",
    "reference": ROOT / "artifacts/ape-cove50-hand-candidate-mjlab-20261010/reach-entry/development-selected.npz",
    "geometry": ROOT / "artifacts/ape-cove50-side-reach-20261009/geometry-center.json",
    "calibration": BASE / "rgb-calibration-model.npz",
    "camera_freeze": ROOT / "artifacts/ape-cove50-reach-viewpoint-20261010/heldout-freeze.json",
    "virtual_rgb_observer": Path(__file__).with_name("virtual_rgb_goal.py"),
    "calibrated_rgb_observer": Path(__file__).with_name("calibrated_virtual_rgb_goal.py"),
    "evaluator": EVALUATOR,
    "pointer": ROOT / "robot/ape-current-design.json",
}


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def frozen_plan() -> dict:
    old_plan = json.loads((V14 / "plan.json").read_text())
    dev_path = V14 / "development/matrix.json"
    fresh_path = V14 / "fresh-reserved/matrix.json"
    dev = json.loads(dev_path.read_text())
    fresh = json.loads(fresh_path.read_text())
    if (dev["case_count"], dev["pass_count"], fresh["case_count"], fresh["pass_count"]) != (10, 10, 20, 16):
        raise ValueError("Expected complete v14 opened and sealed fresh cohorts")
    pointer = json.loads(SOURCES["pointer"].read_text())
    if pointer["status"] != "rejected_no_selected_mechanical_design":
        raise ValueError("Historical control sandbox only")
    cases = ([{"origin": "prior_v14_development", **case} for case in old_plan["opened_development_cases"]] +
             [{"origin": "prior_v14_fresh_now_opened", **case} for case in old_plan["fresh_reserved_cases"]])
    if len(cases) != 30 or len({case["seed"] for case in cases}) != 30:
        raise ValueError("Expected thirty unique opened starts")
    return {
        "schema": "robot-learning-lab.cove50-v15-opened-native-plan.v1",
        "role": "opened_development_gate_only",
        "source_sha256": {name: sha(path) for name, path in SOURCES.items()},
        "v14_development_matrix_sha256": sha(dev_path),
        "v14_fresh_matrix_sha256": sha(fresh_path),
        "actor_training_plan_sha256": sha(V15 / "actor-training-plan.json"),
        "actor_training_report_sha256": sha(V15 / "actor-training-report.json"),
        "cases": cases,
        "fresh_gate": "All thirty opened starts must pass before any new disjoint v15 score",
    }


def run_case(case: dict, plan: dict, old_mode: str) -> dict:
    seed = case["seed"]
    result, trace, log = (DEST / f"seed-{seed}.{suffix}" for suffix in ("json", "npz", "log"))
    command = [sys.executable, str(EVALUATOR),
               "--body-xml", str(SOURCES["body"]),
               "--calibration-model", str(SOURCES["calibration"]),
               "--squat-checkpoint", str(SOURCES["squat"]),
               "--reach-checkpoint", str(SOURCES["actor"]),
               "--alternate-reach-checkpoint", str(SOURCES["alternate_actor"]),
               "--reach-controller", "learned-residual",
               "--reach-reference", str(SOURCES["reference"]),
               "--geometry", str(SOURCES["geometry"]),
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
            raise RuntimeError(f"Evaluator failed for seed {seed}: {log}")
    report = json.loads(result.read_text())
    expected_mode = {"v10_locked": "v15_v10_height_locked",
                     "v14_live": "v15_live", "v14_locked": "v15_locked"}[old_mode]
    if (report["evaluator_sha256"] != plan["source_sha256"]["evaluator"] or
            report["reach_checkpoint_sha256"] != plan["source_sha256"]["actor"] or
            report["alternate_reach_checkpoint_sha256"] != plan["source_sha256"]["alternate_actor"] or
            report["selected_pointer_sha256"] != plan["source_sha256"]["pointer"] or
            report["adaptive_variant"] != "v15_candidate" or
            report["selection_mode"] != expected_mode or
            report["v10_terminal_arm_offset_applied"] != (old_mode == "v10_locked")):
        raise ValueError(f"V15 report identity mismatch: {result}")
    faults = gate(report)
    entry = next((event for event in report["image_goal_capture_events"] if event["time_s"] == 7.5), None)
    if not entry or not entry["used_for_control"] or entry["orange_pixels"] < 10:
        faults.append("no valid RGB goal at entry")
    if entry and (entry.get("xy_error_truth_mm_scoring_only") is None or
                  entry["xy_error_truth_mm_scoring_only"] > 10):
        faults.append("RGB error above 10 mm")
    if report["image_goal_missing_at_s"] is not None:
        faults.append("image goal missing during reach")
    if expected_mode == "v15_live":
        if report["image_goal_lock_after_entry"] or report["image_goal_used_update_count"] < 2:
            faults.append("live RGB updates missing")
    elif not report["image_goal_lock_after_entry"] or report["image_goal_used_update_count"] != 1:
        faults.append("entry RGB goal not locked")
    return {
        "seed": seed, "origin": case["origin"], "selection_mode": expected_mode,
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


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--freeze-only", action="store_true")
    parser.add_argument("--workers", type=int, default=3)
    args = parser.parse_args()
    if not 1 <= args.workers <= 3:
        raise ValueError("Use one to three independent native evaluators")
    plan = frozen_plan()
    DEST.mkdir(parents=True, exist_ok=True)
    plan_path = V15 / "opened-native-plan.json"
    planned_bytes = (json.dumps(plan, indent=2) + "\n").encode()
    if plan_path.exists():
        if plan_path.read_bytes() != planned_bytes:
            raise ValueError("Frozen v15 development plan differs from current sources")
    else:
        plan_path.write_bytes(planned_bytes)
    if args.freeze_only:
        print(f"frozen {plan_path} sha256={sha(plan_path)}")
        return
    previous = {}
    for stage in ("development", "fresh-reserved"):
        old = json.loads((V14 / stage / "matrix.json").read_text())
        previous.update({row["seed"]: row["selection_mode"] for row in old["cases"]})
    rows = {}
    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        futures = {pool.submit(run_case, case, plan, previous[case["seed"]]): case["seed"]
                   for case in plan["cases"]}
        for future in as_completed(futures):
            seed = futures[future]
            rows[seed] = future.result()
            ordered = [rows[case["seed"]] for case in plan["cases"] if case["seed"] in rows]
            matrix = {
                "schema": "robot-learning-lab.cove50-v15-opened-native.v1",
                "role": "opened_development_gate_only",
                "plan_sha256": sha(plan_path),
                "complete": len(ordered) == 30,
                "case_count": len(ordered), "pass_count": sum(row["pass"] for row in ordered),
                "cases": ordered,
            }
            (DEST / "matrix.json").write_text(json.dumps(matrix, indent=2) + "\n")
            print(json.dumps(rows[seed]), flush=True)


if __name__ == "__main__":
    main()
