"""Gate the automatic scoped-Y RGB selector on opened native starts."""

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
from experiments.compact_morphology.cove50_policy_transfer.reach_image_goal import (
    run_v16_opened_matrix as v16,
)


BASE = ROOT / "artifacts/ape-cove50-reach-goal-channels-20261010"
V18 = BASE / "v18"
V20 = BASE / "v20"
OUT = BASE / "v21"
PLAN = OUT / "opened-gate-plan.json"
DEST = OUT / "opened-native"
EVALUATOR = Path(__file__).with_name("evaluate_adaptive_goal_v21.py")
ORIGINAL_SEEDS = (10417, 10422, 10433)
REQUIRED_RESCUES = (10510, 10511, 10518, 10522, 10523, 10543, 10544, 10546)


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def mode_for_xy(x: float, y: float) -> str:
    prior = v16.mode_for_xy(x, y)
    if prior == "v16_locked" and (x > .0015 or (x > -.001 and y > .009)):
        return "v21_scoped_y_locked"
    return prior


def frozen_plan() -> dict:
    development_path = V18 / "development-variance-plan.json"
    baseline_path = V18 / "development-baseline/matrix.json"
    screen_path = V20 / "opened-bias-screen/matrix.json"
    development = json.loads(development_path.read_text())
    baseline = json.loads(baseline_path.read_text())
    screen = json.loads(screen_path.read_text())
    if (not baseline["complete"] or baseline["case_count"] != 48 or
            not screen["complete"] or screen["case_count"] != 8 or
            screen["pass_count"] != 8 or screen["paired_regressions"]):
        raise ValueError("Need complete v18 baseline and clean v20 correction screen")
    pointer = json.loads(v16.SOURCES["pointer"].read_text())
    if pointer["status"] != "rejected_no_selected_mechanical_design":
        raise ValueError("Historical rejected-body control sandbox only")
    baseline_rows = {row["seed"]: row for row in baseline["cases"]}
    v16_path = BASE / "v16/fresh-reserved-plan.json"
    v16_cases = {case["seed"]: case for case in json.loads(v16_path.read_text())["cases"]}
    v16_rows = {row["seed"]: row for row in json.loads((BASE / "v16/fresh-reserved/matrix.json").read_text())["cases"]}
    cases = []
    for case in development["cases"]:
        seed = case["seed"]
        row = baseline_rows[seed]
        report_path = ROOT / row["result"]
        report = json.loads(report_path.read_text())
        x, y = report["selection_features"]["entry_error_xy_m"]
        cases.append({"seed": seed, "cohort": "v18_opened_derived", "origin_seed": case["origin_seed"],
                      "rail_v": case["rail_v"], "delay_ms": case["delay_ms"],
                      "toy_offset_x_m": case["toy_offset_x_m"], "toy_offset_y_m": case["toy_offset_y_m"],
                      "gaze_head_yaw_rad": case["gaze_head_yaw_rad"],
                      "baseline_report": row["result"], "baseline_report_sha256": sha(report_path),
                      "baseline_pass": row["pass"], "baseline_contact_s": row["contact_s"],
                      "expected_mode": mode_for_xy(x, y), "baseline_entry_error_xy_m": [x, y]})
    for seed in ORIGINAL_SEEDS:
        case, row = v16_cases[seed], v16_rows[seed]
        report_path = ROOT / row["result"]
        report = json.loads(report_path.read_text())
        x, y = report["selection_features"]["entry_error_xy_m"]
        cases.append({"seed": seed, "cohort": "original_opened_failure", "origin_seed": seed,
                      "rail_v": case["rail_v"], "delay_ms": case["delay_ms"],
                      "toy_offset_x_m": case["toy_offset_x_m"], "toy_offset_y_m": case["toy_offset_y_m"],
                      "gaze_head_yaw_rad": case["gaze_head_yaw_rad"],
                      "baseline_report": row["result"], "baseline_report_sha256": sha(report_path),
                      "baseline_pass": row["pass"], "baseline_contact_s": row["contact_s"],
                      "expected_mode": mode_for_xy(x, y), "baseline_entry_error_xy_m": [x, y]})
    if len(cases) != 51 or len({case["seed"] for case in cases}) != 51:
        raise ValueError("Expected 48 varied and three original opened starts")
    sources = {name: sha(path) for name, path in v16.SOURCES.items()}
    sources["evaluator"] = sha(EVALUATOR)
    return {
        "schema": "robot-learning-lab.cove50-v21-automatic-scoped-y-opened-plan.v1",
        "role": "opened_native_development_gate_not_fresh_score",
        "body_role": "rejected_historical_direct_frame_sandbox",
        "v18_development_plan_sha256": sha(development_path),
        "v18_baseline_matrix_sha256": sha(baseline_path),
        "v20_screen_matrix_sha256": sha(screen_path),
        "v16_fresh_plan_sha256": sha(v16_path),
        "source_sha256": sources,
        "runner_sha256": sha(Path(__file__)),
        "paired_pre_reach_tolerance": {"qpos_max_abs": 5e-5, "action_max_abs": 1e-3},
        "selector": "Preserve v16 branch order; only if it would select v16_locked, choose the saved v13 actor with -2 mm body-Y RGB goal when observed entry error x > 1.5 mm, or x > -1 mm and y > 9 mm.",
        "cases": cases,
        "required_rescue_seeds": list(REQUIRED_RESCUES),
        "advance_gate": "On all 48 opened-derived varied starts: at least 40 pass, zero matched regressions, and all eight prescreened rescue seeds pass. Original opened 10417 and 10422 must also pass; 10433 is a diagnostic. No disjoint set is opened until this gate is sealed.",
        "qualification": "Automatic observed-RGB entry routing on a rejected historical body; opened development only. A passing gate permits disjoint native evaluation but cannot qualify a newly selected body.",
    }


def freeze(plan: dict) -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    expected = (json.dumps(plan, indent=2) + "\n").encode()
    if PLAN.exists():
        if PLAN.read_bytes() != expected:
            raise ValueError("Frozen v21 opened plan differs from sources")
    else:
        PLAN.write_bytes(expected)


def run_case(case: dict, plan: dict) -> dict:
    seed = case["seed"]
    result, trace, log = (DEST / f"seed-{seed}.{suffix}" for suffix in ("json", "npz", "log"))
    sources = v16.SOURCES
    command = [sys.executable, str(EVALUATOR),
               "--body-xml", str(sources["body"]),
               "--calibration-model", str(sources["calibration"]),
               "--squat-checkpoint", str(sources["squat"]),
               "--reach-checkpoint", str(sources["actor"]),
               "--alternate-reach-checkpoint", str(sources["alternate_actor"]),
               "--specialist-reach-checkpoint", str(sources["specialist_actor"]),
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
            raise RuntimeError(f"V21 evaluator failed for seed {seed}: {log}")
    report = json.loads(result.read_text())
    old_path = ROOT / case["baseline_report"]
    old = json.loads(old_path.read_text())
    with np.load(trace) as new_trace, np.load(old_path.with_suffix(".npz")) as old_trace:
        if (new_trace["qpos"].shape != old_trace["qpos"].shape or
                new_trace["action"].shape != old_trace["action"].shape):
            raise ValueError(f"V21 paired trace shape differs: {result}")
        entry_step = 375
        qpos_delta = float(np.max(np.abs(new_trace["qpos"][entry_step] -
                                           old_trace["qpos"][entry_step])))
        action_delta = float(np.max(np.abs(new_trace["action"][:entry_step] -
                                            old_trace["action"][:entry_step])))
    hashes = plan["source_sha256"]
    mode = case["expected_mode"]
    expected_checkpoint = (hashes["alternate_actor"] if mode.startswith("v16_v10") else
                           hashes["specialist_actor"] if mode == "v16_specialist_locked" else
                           hashes["actor"])
    expected_bias = -.002 if mode in ("v16_bias_y_locked", "v21_scoped_y_locked") else 0.0
    x, y = report["selection_features"]["entry_error_xy_m"]
    bx, by = case["baseline_entry_error_xy_m"]
    if (sha(old_path) != case["baseline_report_sha256"] or
            report["evaluator_sha256"] != hashes["evaluator"] or
            report["body_sha256"] != hashes["body"] or
            report["selected_pointer_sha256"] != hashes["pointer"] or
            report["reach_checkpoint_sha256"] != hashes["actor"] or
            report["selected_reach_checkpoint_sha256"] != expected_checkpoint or
            report["adaptive_variant"] != "v21_scoped_y_router_candidate" or
            report["selection_mode"] != mode or
            report["goal_bias_body_xy_m"] != [0.0, expected_bias] or
            mode_for_xy(x, y) != mode or max(abs(x - bx), abs(y - by)) > 5e-5 or
            report["boundary_state_sha256"]["posture"] != old["boundary_state_sha256"]["posture"] or
            report["first_action_sha256"] != old["first_action_sha256"] or
            qpos_delta > plan["paired_pre_reach_tolerance"]["qpos_max_abs"] or
            action_delta > plan["paired_pre_reach_tolerance"]["action_max_abs"]):
        raise ValueError(f"V21 selector or paired-start identity mismatch: {result}")
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
        "seed": seed, "cohort": case["cohort"], "origin_seed": case["origin_seed"],
        "selection_mode": mode, "baseline_selection_mode": old["selection_mode"],
        "pass": not faults, "baseline_pass": case["baseline_pass"],
        "contact_s": report["phases"]["reach"]["continuous_finger_contact_s"],
        "baseline_contact_s": case["baseline_contact_s"],
        "rail_v": case["rail_v"], "delay_ms": case["delay_ms"],
        "toy_offset_b_m": [case["toy_offset_x_m"], case["toy_offset_y_m"]],
        "entry_error_body_xy_m": [x, y],
        "entry_goal_error_mm_scoring_only": None if entry is None else entry.get("xy_error_truth_mm_scoring_only"),
        "paired_reach_entry_qpos_max_abs": qpos_delta,
        "paired_pre_reach_action_max_abs": action_delta,
        "reach_both_feet_fraction": report["phases"]["reach"]["both_feet_fraction"],
        "reach_rated_leg_fraction": report["phases"]["reach"]["rated_leg_fraction"],
        "faults": faults,
        "result": str(result.relative_to(ROOT)), "trace": str(trace.relative_to(ROOT)),
        "log": str(log.relative_to(ROOT)),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--freeze-only", action="store_true")
    parser.add_argument("--workers", type=int, default=6)
    args = parser.parse_args()
    if not 1 <= args.workers <= 6:
        raise ValueError("Use one to six native evaluators")
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
            case = futures[future]
            row = future.result()
            rows[case["seed"]] = row
            ordered = [rows[case["seed"]] for case in plan["cases"] if case["seed"] in rows]
            varied = [item for item in ordered if item["cohort"] == "v18_opened_derived"]
            all_done = len(ordered) == len(plan["cases"])
            no_regression = not any(not item["pass"] and item["baseline_pass"] for item in varied)
            matrix = {
                "schema": "robot-learning-lab.cove50-v21-automatic-scoped-y-opened-native.v1",
                "role": "opened_development_not_disjoint_fresh",
                "plan_sha256": sha(PLAN),
                "complete": all_done,
                "case_count": len(ordered),
                "varied_case_count": len(varied),
                "varied_pass_count": sum(item["pass"] for item in varied),
                "original_pass_count": sum(item["pass"] for item in ordered if item["cohort"] == "original_opened_failure"),
                "scoped_y_selection_count": sum(item["selection_mode"] == "v21_scoped_y_locked" for item in ordered),
                "paired_rescues": [item["seed"] for item in ordered
                                   if item["pass"] and not item["baseline_pass"]],
                "paired_regressions": [item["seed"] for item in ordered
                                       if not item["pass"] and item["baseline_pass"]],
                "advance_gate_pass": (all_done and len(varied) == 48 and
                                      sum(item["pass"] for item in varied) >= 40 and no_regression and
                                      all(rows[seed]["pass"] for seed in plan["required_rescue_seeds"]) and
                                      rows[10417]["pass"] and rows[10422]["pass"]),
                "cases": ordered,
            }
            (DEST / "matrix.json").write_text(json.dumps(matrix, indent=2) + "\n")
            print(json.dumps(row), flush=True)


if __name__ == "__main__":
    main()
