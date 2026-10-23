"""Replay predeclared native reach cases using only the calibrated RGB goal."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(ROOT))
from experiments.compact_morphology.cove50_policy_transfer.run_squat_learned_reach_matrix import gate

OUT = ROOT / "artifacts/ape-cove50-reach-goal-channels-20261010"
HERE = Path(__file__).resolve().parent
SOURCES = {
    "body": ROOT / "artifacts/ape-cove50-transfer-20261008/cove50-direct-training.xml",
    "squat_checkpoint": ROOT / "artifacts/ape-cove50-squat-recovery-20261009/squat-teacher-bc-blendmask.pt",
    "reach_checkpoint": ROOT / "artifacts/ape-cove50-squat-learned-reach-20261010/v7/actor.pt",
    "reach_reference": ROOT / "artifacts/ape-cove50-hand-candidate-mjlab-20261010/reach-entry/development-selected.npz",
    "geometry": ROOT / "artifacts/ape-cove50-side-reach-20261009/geometry-center.json",
    "observer": HERE / "virtual_rgb_goal.py",
    "features": HERE / "rgb_goal_features.py",
    "model_source": HERE / "rgb_calibration_model.py",
    "calibrated_observer": HERE / "calibrated_virtual_rgb_goal.py",
    "evaluator": HERE / "evaluate_calibrated_goal.py",
    "calibration_model": OUT / "rgb-calibration-model.npz",
}


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--stage", choices=("development", "fresh-reserved"), required=True)
    args = parser.parse_args()
    plan_path = OUT / "rgb-calibration-plan.json"
    plan = json.loads(plan_path.read_text())
    freeze = json.loads((OUT / "calibrated-run-freeze.json").read_text())
    pointer = json.loads((ROOT / "robot/ape-current-design.json").read_text())
    if pointer["status"] != "rejected_no_selected_mechanical_design":
        raise ValueError("Historical sandbox requires a rejected-body pointer")
    if sha(plan_path) != freeze["plan_sha256"]:
        raise ValueError("Calibrated run plan changed")
    for key, path in SOURCES.items():
        if sha(path) != freeze["sha256"][key]:
            raise ValueError(f"Calibrated source changed: {path}")
    fit = json.loads((OUT / "rgb-calibration-fit.json").read_text())
    if not fit["offline_gate_pass"] or fit["model_sha256"] != freeze["sha256"]["calibration_model"]:
        raise ValueError("RGB calibration offline gate not frozen and passed")
    if args.stage == "development":
        opened = json.loads((OUT / "plan.json").read_text())
        cases = [next(case for case in opened["cases"] if case["seed"] == seed and
                      case["route_goal_source"] == case["residual_goal_source"] == "image")
                 for seed in plan["native_development_cases"]]
    else:
        development = json.loads((OUT / "calibrated-development/matrix.json").read_text())
        if development["pass_count"] != len(plan["native_development_cases"]):
            raise ValueError("Opened native gate failed; fresh starts remain sealed")
        cases = plan["fresh_reserved_cases"]
    stage_dir = OUT / ("calibrated-development" if args.stage == "development"
                       else "calibrated-fresh-reserved")
    stage_dir.mkdir(parents=True, exist_ok=True)
    rows = []
    for case in cases:
        stem = f"seed-{case['seed']}"
        result = stage_dir / f"{stem}.json"
        trace = stage_dir / f"{stem}.npz"
        log = stage_dir / f"{stem}.log"
        command = [sys.executable, str(SOURCES["evaluator"]),
                   "--body-xml", str(SOURCES["body"]),
                   "--calibration-model", str(SOURCES["calibration_model"]),
                   "--squat-checkpoint", str(SOURCES["squat_checkpoint"]),
                   "--reach-checkpoint", str(SOURCES["reach_checkpoint"]),
                   "--reach-controller", "learned-residual",
                   "--reach-reference", str(SOURCES["reach_reference"]),
                   "--geometry", str(SOURCES["geometry"]),
                   "--entry", "continuous", "--seed", str(case["seed"]),
                   "--rail-v", str(case["rail_v"]), "--delay-ms", str(case["delay_ms"]),
                   "--toy-offset-x-m", str(case["toy_offset_x_m"]),
                   "--toy-offset-y-m", str(case["toy_offset_y_m"]),
                   "--pitch-offset-rad", ".105", "--roll-clearance-rad", "-.25",
                   "--terminal-roll-offset-rad", "-.08", "--terminal-roll-mode", "entry-error-x",
                   "--reach-duration-s", "9", "--gaze-head-yaw-rad", str(case["gaze_head_yaw_rad"]),
                   "--gaze-start-s", "0", "--gaze-ramp-s", "7",
                   "--capture-stride-steps", "5", "--max-goal-age-s", "9",
                   "--lock-goal-after-entry", "--out", str(result), "--trace", str(trace)]
        if not result.exists():
            env = os.environ.copy()
            env["PYTHONPATH"] = str(ROOT) + (os.pathsep + env["PYTHONPATH"] if env.get("PYTHONPATH") else "")
            with log.open("w") as stream:
                code = subprocess.run(command, cwd=ROOT, env=env, stdout=stream,
                                      stderr=subprocess.STDOUT, check=False).returncode
        else:
            code = 0
        if code or not result.exists():
            row = {"seed": case["seed"], "pass": False,
                   "faults": [f"evaluator exit {code}"], "log": str(log.relative_to(ROOT))}
        else:
            report = json.loads(result.read_text())
            if (report["body_sha256"] != freeze["sha256"]["body"] or
                    report["reach_checkpoint_sha256"] != freeze["sha256"]["reach_checkpoint"] or
                    report["evaluator_sha256"] != freeze["sha256"]["evaluator"] or
                    report["calibration_model_sha256"] != freeze["sha256"]["calibration_model"] or
                    report["seed"] != case["seed"] or report["body_role"] != plan["body_role"]):
                raise ValueError(f"Native report identity mismatch: {result}")
            faults = gate(report)
            entry = next((event for event in report["image_goal_capture_events"]
                          if event["time_s"] == 7.5), None)
            if not entry or not entry["used_for_control"] or entry["orange_pixels"] < 10:
                faults.append("no valid RGB goal used at entry")
            if entry and (entry.get("xy_error_truth_mm_scoring_only") is None or
                          entry["xy_error_truth_mm_scoring_only"] > 10):
                faults.append("entry RGB goal outside 10 mm check")
            if report["image_goal_missing_at_s"] is not None or report["image_goal_used_update_count"] != 1:
                faults.append("missing, expired or unlocked entry image goal")
            yaw = next((r["head_yaw_rad"] for r in report["timeseries"] if r["time_s"] == 7.5), None)
            if yaw is None or abs(yaw - case["gaze_head_yaw_rad"]) > .05:
                faults.append("head yaw missed entry target")
            row = {"seed": case["seed"], "pass": not faults, "faults": faults,
                   "rail_v": case["rail_v"], "delay_ms": case["delay_ms"],
                   "toy_offset_b_m": [case["toy_offset_x_m"], case["toy_offset_y_m"]],
                   "contact_s": report["phases"]["reach"]["continuous_finger_contact_s"],
                   "entry_goal_error_mm_scoring_only": None if entry is None else entry.get("xy_error_truth_mm_scoring_only"),
                   "entry_goal_correction_xy_mm": None if entry is None else entry.get("calibration_correction_xy_mm"),
                   "reach_both_feet_fraction": report["phases"]["reach"]["both_feet_fraction"],
                   "reach_rated_leg_fraction": report["phases"]["reach"]["rated_leg_fraction"],
                   "max_abs_unbounded_action": report["max_abs_unbounded_action"],
                   "result": str(result.relative_to(ROOT)), "trace": str(trace.relative_to(ROOT)),
                   "log": str(log.relative_to(ROOT))}
        rows.append(row)
        matrix = {"schema": "robot-learning-lab.cove50-calibrated-rgb-native-matrix.v1",
                  "body_role": plan["body_role"], "stage": args.stage,
                  "plan": str(plan_path.relative_to(ROOT)), "plan_sha256": sha(plan_path),
                  "freeze_sha256": sha(OUT / "calibrated-run-freeze.json"),
                  "pass_count": sum(row["pass"] for row in rows), "case_count": len(rows),
                  "cases": rows}
        (stage_dir / "matrix.json").write_text(json.dumps(matrix, indent=2, allow_nan=False) + "\n")
        print(json.dumps(row), flush=True)


if __name__ == "__main__":
    main()
