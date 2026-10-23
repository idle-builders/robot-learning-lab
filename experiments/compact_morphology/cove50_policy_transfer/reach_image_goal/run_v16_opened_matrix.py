"""Gate the v16 observable router on opened native starts, in two stages."""

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
V16 = BASE / "v16"
DEST = V16 / "opened-native"
EVALUATOR = Path(__file__).with_name("evaluate_adaptive_goal_v16.py")
SOURCES = {
    "body": ROOT / "artifacts/ape-cove50-transfer-20261008/cove50-direct-training.xml",
    "squat": ROOT / "artifacts/ape-cove50-squat-recovery-20261009/squat-teacher-bc-blendmask.pt",
    "actor": BASE / "v13/actor.pt",
    "alternate_actor": BASE / "v10/actor.pt",
    "specialist_actor": V15 / "actor-locked.pt",
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
    stage_one_seeds = {case["seed"] for case in old_plan["opened_development_cases"]} | {10303, 10304, 10310, 10318}
    return {
        "schema": "robot-learning-lab.cove50-v16-opened-native-plan.v1",
        "role": "opened_development_gate_only",
        "source_sha256": {name: sha(path) for name, path in SOURCES.items()},
        "v14_development_matrix_sha256": sha(dev_path),
        "v14_fresh_matrix_sha256": sha(fresh_path),
        "specialist_training_plan_sha256": sha(V15 / "actor-training-plan.json"),
        "specialist_training_report_sha256": sha(V15 / "actor-training-report.json"),
        "stage_one_seeds": [case["seed"] for case in cases if case["seed"] in stage_one_seeds],
        "stage_two_seeds": [case["seed"] for case in cases if case["seed"] not in stage_one_seeds],
        "cases": cases,
        "fresh_gate": "All thirty opened starts must pass before any new disjoint v16 score",
    }


def run_case(case: dict, plan: dict) -> dict:
    seed = case["seed"]
    result, trace, log = (DEST / f"seed-{seed}.{suffix}" for suffix in ("json", "npz", "log"))
    command = [sys.executable, str(EVALUATOR),
               "--body-xml", str(SOURCES["body"]),
               "--calibration-model", str(SOURCES["calibration"]),
               "--squat-checkpoint", str(SOURCES["squat"]),
               "--reach-checkpoint", str(SOURCES["actor"]),
               "--alternate-reach-checkpoint", str(SOURCES["alternate_actor"]),
               "--specialist-reach-checkpoint", str(SOURCES["specialist_actor"]),
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
    expected_mode = (selection_for_case(case)
                     if case["origin"] != "v16_fresh_reserved" else report["selection_mode"])
    if expected_mode not in {"v16_v10_height_locked", "v16_v10_locked", "v16_live",
                             "v16_bias_y_locked", "v16_specialist_locked", "v16_locked"}:
        raise ValueError(f"Unexpected V16 selection mode: {result}")
    if (report["evaluator_sha256"] != plan["source_sha256"]["evaluator"] or
            report["reach_checkpoint_sha256"] != plan["source_sha256"]["actor"] or
            report["alternate_reach_checkpoint_sha256"] != plan["source_sha256"]["alternate_actor"] or
            report["specialist_reach_checkpoint_sha256"] != plan["source_sha256"]["specialist_actor"] or
            report["selected_pointer_sha256"] != plan["source_sha256"]["pointer"] or
            report["adaptive_variant"] != "v16_candidate" or
            report["selection_mode"] != expected_mode or
            report["v10_terminal_arm_offset_applied"] != (expected_mode == "v16_v10_height_locked")):
        raise ValueError(f"V16 report identity mismatch: {result}")
    selected_sha = (plan["source_sha256"]["alternate_actor"] if expected_mode.startswith("v16_v10") else
                    plan["source_sha256"]["specialist_actor"] if expected_mode == "v16_specialist_locked" else
                    plan["source_sha256"]["actor"])
    if report["selected_reach_checkpoint_sha256"] != selected_sha:
        raise ValueError(f"V16 selected actor mismatch: {result}")
    faults = gate(report)
    entry = next((event for event in report["image_goal_capture_events"] if event["time_s"] == 7.5), None)
    if not entry or not entry["used_for_control"] or entry["orange_pixels"] < 10:
        faults.append("no valid RGB goal at entry")
    if entry and (entry.get("xy_error_truth_mm_scoring_only") is None or
                  entry["xy_error_truth_mm_scoring_only"] > 10):
        faults.append("RGB error above 10 mm")
    if report["image_goal_missing_at_s"] is not None:
        faults.append("image goal missing during reach")
    if expected_mode == "v16_live":
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


def selection_for_case(case: dict) -> str:
    # Selection is observable at the native 7.5 s entry, so the expected
    # branch is read from the frozen v14 entry capture for this opened seed.
    prior = V14 / ("development" if case["origin"] == "prior_v14_development" else "fresh-reserved") / "matrix.json"
    row = next(row for row in json.loads(prior.read_text())["cases"] if row["seed"] == case["seed"])
    x, y = row["selection_features"]["entry_error_xy_m"]
    return mode_for_xy(x, y)


def mode_for_xy(x: float, y: float) -> str:
    if x < -.0095 and y < -.0025:
        return "v16_v10_height_locked" if x < -.012 else "v16_v10_locked"
    if -.004 <= x < 0.0 and (y < -.004 or (x < -.001 and y < -.002)):
        return "v16_live"
    if -.007 <= x < -.0055 and -.001 <= y < .001:
        return "v16_bias_y_locked"
    if 0.0 <= x < .002 and .001 <= y < .004:
        return "v16_specialist_locked"
    return "v16_locked"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--freeze-only", action="store_true")
    parser.add_argument("--workers", type=int, default=3)
    args = parser.parse_args()
    if not 1 <= args.workers <= 3:
        raise ValueError("Use one to three independent native evaluators")
    plan = frozen_plan()
    DEST.mkdir(parents=True, exist_ok=True)
    plan_path = V16 / "opened-native-plan.json"
    planned_bytes = (json.dumps(plan, indent=2) + "\n").encode()
    if plan_path.exists():
        if plan_path.read_bytes() != planned_bytes:
            raise ValueError("Frozen v16 development plan differs from current sources")
    else:
        plan_path.write_bytes(planned_bytes)
    if args.freeze_only:
        print(f"frozen {plan_path} sha256={sha(plan_path)}")
        return
    rows = {}
    for stage_number, stage_seeds in ((1, plan["stage_one_seeds"]),
                                      (2, plan["stage_two_seeds"])):
        cases = [case for case in plan["cases"] if case["seed"] in stage_seeds]
        with ThreadPoolExecutor(max_workers=args.workers) as pool:
            futures = {pool.submit(run_case, case, plan): case["seed"] for case in cases}
            for future in as_completed(futures):
                seed = futures[future]
                rows[seed] = future.result()
                ordered = [rows[case["seed"]] for case in plan["cases"] if case["seed"] in rows]
                matrix = {
                    "schema": "robot-learning-lab.cove50-v16-opened-native.v1",
                    "role": "opened_development_gate_only",
                    "plan_sha256": sha(plan_path),
                    "stage": stage_number,
                    "complete": len(ordered) == 30,
                    "case_count": len(ordered), "pass_count": sum(row["pass"] for row in ordered),
                    "cases": ordered,
                }
                (DEST / "matrix.json").write_text(json.dumps(matrix, indent=2) + "\n")
                print(json.dumps(rows[seed]), flush=True)
        if not all(rows[seed]["pass"] for seed in stage_seeds):
            matrix["stopped_early"] = True
            matrix["stop_reason"] = (f"Stage {stage_number} opened gate failed; "
                                     "no later stage or fresh seeds opened")
            (DEST / "matrix.json").write_text(json.dumps(matrix, indent=2) + "\n")
            print(f"stage {stage_number} failed; no later stage opened", flush=True)
            break


if __name__ == "__main__":
    main()
