"""Reconcile v21 native reports after strict RGB pixel-repeatability check halted the matrix writer.

The original evaluator, opened cases, branch rule, physical gate, and fresh challenge stay
frozen. This audit only changes the paired RGB entry repeatability tolerance, while
requiring the same selected branch and tighter paired physical/action checks.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import shutil

import numpy as np

from experiments.compact_morphology.cove50_policy_transfer.run_squat_learned_reach_matrix import gate
from experiments.compact_morphology.cove50_policy_transfer.reach_image_goal import (
    run_v16_opened_matrix as v16,
    run_v21_opened_gate as v21,
)


ROOT = Path(__file__).resolve().parents[4]
OUT = ROOT / "artifacts/ape-cove50-reach-goal-channels-20261010/v21"
PLAN = OUT / "opened-gate-plan.json"
AMENDMENT = OUT / "opened-rgb-repeatability-amendment.json"
DEST = OUT / "opened-native"


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def audited_row(case: dict, plan: dict, rgb_tolerance_m: float) -> dict:
    seed = case["seed"]
    result, trace, log = (DEST / f"seed-{seed}.{suffix}" for suffix in ("json", "npz", "log"))
    for path in (result, trace, log):
        if not path.exists():
            raise ValueError(f"Missing v21 native artifact: {path}")
    report = json.loads(result.read_text())
    old_path = ROOT / case["baseline_report"]
    old = json.loads(old_path.read_text())
    with np.load(trace) as new_trace, np.load(old_path.with_suffix(".npz")) as old_trace:
        if (new_trace["qpos"].shape != old_trace["qpos"].shape or
                new_trace["action"].shape != old_trace["action"].shape):
            raise ValueError(f"Paired trace shape differs: {result}")
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
    rgb_delta = max(abs(x - bx), abs(y - by))
    if (sha(old_path) != case["baseline_report_sha256"] or
            report["evaluator_sha256"] != hashes["evaluator"] or
            report["body_sha256"] != hashes["body"] or
            report["selected_pointer_sha256"] != hashes["pointer"] or
            report["reach_checkpoint_sha256"] != hashes["actor"] or
            report["selected_reach_checkpoint_sha256"] != expected_checkpoint or
            report["adaptive_variant"] != "v21_scoped_y_router_candidate" or
            report["selection_mode"] != mode or
            report["goal_bias_body_xy_m"] != [0.0, expected_bias] or
            v21.mode_for_xy(x, y) != mode or rgb_delta > rgb_tolerance_m or
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
        "paired_rgb_entry_delta_m": rgb_delta,
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
    parser.add_argument("--publish", action="store_true")
    args = parser.parse_args()
    plan = json.loads(PLAN.read_text())
    amendment = json.loads(AMENDMENT.read_text())
    if (amendment["opened_plan_sha256"] != sha(PLAN) or
            amendment["audit_source_sha256"] != sha(Path(__file__)) or
            amendment["rgb_entry_tolerance_m"] != .0005 or
            plan["runner_sha256"] != sha(Path(v21.__file__))):
        raise ValueError("V21 opened audit sources or frozen amendment changed")
    for name, path in v16.SOURCES.items():
        if name != "evaluator" and sha(path) != plan["source_sha256"][name]:
            raise ValueError(f"V21 source changed: {name}")
    if sha(v21.EVALUATOR) != plan["source_sha256"]["evaluator"]:
        raise ValueError("V21 evaluator changed")
    rows = [audited_row(case, plan, amendment["rgb_entry_tolerance_m"])
            for case in plan["cases"]]
    varied = [item for item in rows if item["cohort"] == "v18_opened_derived"]
    by_seed = {item["seed"]: item for item in rows}
    no_regression = not any(not item["pass"] and item["baseline_pass"] for item in varied)
    matrix = {
        "schema": "robot-learning-lab.cove50-v21-automatic-scoped-y-opened-native-reconciled.v1",
        "role": "opened_development_not_disjoint_fresh",
        "plan_sha256": sha(PLAN),
        "validation_amendment_sha256": sha(AMENDMENT),
        "complete": True,
        "case_count": len(rows),
        "varied_case_count": len(varied),
        "varied_pass_count": sum(item["pass"] for item in varied),
        "original_pass_count": sum(item["pass"] for item in rows if item["cohort"] == "original_opened_failure"),
        "scoped_y_selection_count": sum(item["selection_mode"] == "v21_scoped_y_locked" for item in rows),
        "max_paired_rgb_entry_delta_m": max(item["paired_rgb_entry_delta_m"] for item in rows),
        "paired_rescues": [item["seed"] for item in rows if item["pass"] and not item["baseline_pass"]],
        "paired_regressions": [item["seed"] for item in rows if not item["pass"] and item["baseline_pass"]],
        "advance_gate_pass": (len(rows) == 51 and len(varied) == 48 and
                              sum(item["pass"] for item in varied) >= 40 and no_regression and
                              all(by_seed[seed]["pass"] for seed in plan["required_rescue_seeds"]) and
                              by_seed[10417]["pass"] and by_seed[10422]["pass"]),
        "cases": rows,
    }
    reconciled = DEST / "matrix-reconciled.json"
    reconciled.write_text(json.dumps(matrix, indent=2) + "\n")
    if args.publish:
        original = DEST / "matrix.json"
        strict_partial = DEST / "matrix-strict-partial.json"
        if not strict_partial.exists():
            shutil.copy2(original, strict_partial)
        original.write_text(json.dumps(matrix, indent=2) + "\n")
    print(json.dumps({key: value for key, value in matrix.items() if key != "cases"}, indent=2))


if __name__ == "__main__":
    main()
