"""Native calibrated-RGB replay for the frozen v12 entry router."""

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
V12 = OUT / "v12"
HERE = Path(__file__).resolve().parent
BODY = ROOT / "artifacts/ape-cove50-transfer-20261008/cove50-direct-training.xml"
SQUAT = ROOT / "artifacts/ape-cove50-squat-recovery-20261009/squat-teacher-bc-blendmask.pt"
ACTOR = OUT / "v9/actor.pt"
ALTERNATE_ACTOR = OUT / "v10/actor.pt"
REFERENCE = ROOT / "artifacts/ape-cove50-hand-candidate-mjlab-20261010/reach-entry/development-selected.npz"
GEOMETRY = ROOT / "artifacts/ape-cove50-side-reach-20261009/geometry-center.json"
CALIBRATION = OUT / "rgb-calibration-model.npz"
EVALUATOR = HERE / "evaluate_adaptive_goal_v12.py"


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--stage", choices=("development", "fresh-reserved"), required=True)
    args = parser.parse_args()
    plan_path = V12 / "plan.json"
    plan = json.loads(plan_path.read_text())
    sources = {"body": BODY, "squat": SQUAT, "actor": ACTOR,
               "alternate_actor": ALTERNATE_ACTOR, "reference": REFERENCE,
               "geometry": GEOMETRY, "calibration": CALIBRATION, "evaluator": EVALUATOR}
    for name, path in sources.items():
        if sha(path) != plan["sources_sha256"][name]:
            raise ValueError(f"V12 source changed: {path}")
    pointer = json.loads((ROOT / "robot/ape-current-design.json").read_text())
    if (pointer["status"] != "rejected_no_selected_mechanical_design" or
            sha(ROOT / "robot/ape-current-design.json") != plan["sources_sha256"]["pointer"]):
        raise ValueError("Historical sandbox requires rejected-body pointer")
    if args.stage == "fresh-reserved":
        development = json.loads((V12 / "development/matrix.json").read_text())
        if development["case_count"] != 8 or development["pass_count"] != 8:
            raise ValueError("V12 development failed; fresh starts remain sealed")
    cases = plan["opened_development_cases" if args.stage == "development" else "fresh_reserved_cases"]
    stage_dir = V12 / args.stage
    stage_dir.mkdir(parents=True, exist_ok=True)
    rows = []
    for case in cases:
        stem = f"seed-{case['seed']}"
        result, trace, log = (stage_dir / f"{stem}.{suffix}" for suffix in ("json", "npz", "log"))
        command = [sys.executable, str(EVALUATOR), "--body-xml", str(BODY),
                   "--calibration-model", str(CALIBRATION), "--squat-checkpoint", str(SQUAT),
                   "--reach-checkpoint", str(ACTOR), "--reach-controller", "learned-residual",
                   "--alternate-reach-checkpoint", str(ALTERNATE_ACTOR),
                   "--reach-reference", str(REFERENCE), "--geometry", str(GEOMETRY),
                   "--entry", "continuous", "--seed", str(case["seed"]),
                   "--rail-v", str(case["rail_v"]), "--delay-ms", str(case["delay_ms"]),
                   "--toy-offset-x-m", str(case["toy_offset_x_m"]),
                   "--toy-offset-y-m", str(case["toy_offset_y_m"]),
                   "--pitch-offset-rad", ".105", "--roll-clearance-rad", "-.25",
                   "--terminal-roll-offset-rad", "-.08", "--terminal-roll-mode", "entry-error-x",
                   "--reach-duration-s", "9", "--gaze-head-yaw-rad", str(case["gaze_head_yaw_rad"]),
                   "--gaze-start-s", "0", "--gaze-ramp-s", "7", "--capture-stride-steps", "5",
                   "--max-goal-age-s", "9",
                   "--out", str(result), "--trace", str(trace)]
        if not result.exists():
            env = os.environ.copy()
            env["PYTHONPATH"] = str(ROOT) + (os.pathsep + env["PYTHONPATH"] if env.get("PYTHONPATH") else "")
            with log.open("w") as stream:
                code = subprocess.run(command, cwd=ROOT, env=env, stdout=stream,
                                      stderr=subprocess.STDOUT, check=False).returncode
        else:
            code = 0
        if code or not result.exists():
            row = {"seed": case["seed"], "pass": False, "faults": [f"evaluator exit {code}"],
                   "log": str(log.relative_to(ROOT))}
        else:
            report = json.loads(result.read_text())
            if (report["body_sha256"] != plan["sources_sha256"]["body"] or
                    report["reach_checkpoint_sha256"] != plan["sources_sha256"]["actor"] or
                    report["alternate_reach_checkpoint_sha256"] != plan["sources_sha256"]["alternate_actor"] or
                    report["evaluator_sha256"] != plan["sources_sha256"]["evaluator"] or
                    report["calibration_model_sha256"] != plan["sources_sha256"]["calibration"] or
                    report["seed"] != case["seed"]):
                raise ValueError(f"Native identity mismatch: {result}")
            faults = gate(report)
            entry = next((event for event in report["image_goal_capture_events"]
                          if event["time_s"] == 7.5), None)
            if not entry or not entry["used_for_control"] or entry["orange_pixels"] < 10:
                faults.append("no valid RGB goal at entry")
            if entry and (entry.get("xy_error_truth_mm_scoring_only") is None or
                          entry["xy_error_truth_mm_scoring_only"] > 10):
                faults.append("RGB error above 10 mm")
            mode = report["selection_mode"]
            if mode not in ("v10_locked", "v9_live", "v9_locked"):
                faults.append("invalid adaptive selection mode")
            if mode == "v9_live":
                if report["image_goal_lock_after_entry"] or report["image_goal_used_update_count"] < 2:
                    faults.append("live RGB updates missing")
            elif not report["image_goal_lock_after_entry"] or report["image_goal_used_update_count"] != 1:
                faults.append("entry RGB goal not locked")
            selected_sha = (plan["sources_sha256"]["alternate_actor"] if mode == "v10_locked"
                            else plan["sources_sha256"]["actor"])
            if report["selected_reach_checkpoint_sha256"] != selected_sha:
                faults.append("selected actor identity mismatch")
            if report["image_goal_missing_at_s"] is not None:
                faults.append("image goal missing during reach")
            row = {"seed": case["seed"], "pass": not faults, "faults": faults,
                   "selection_mode": mode,
                   "selection_features": report["selection_features"],
                   "rail_v": case["rail_v"], "delay_ms": case["delay_ms"],
                   "toy_offset_b_m": [case["toy_offset_x_m"], case["toy_offset_y_m"]],
                   "contact_s": report["phases"]["reach"]["continuous_finger_contact_s"],
                   "entry_goal_error_mm_scoring_only": None if entry is None else entry.get("xy_error_truth_mm_scoring_only"),
                   "reach_both_feet_fraction": report["phases"]["reach"]["both_feet_fraction"],
                   "reach_rated_leg_fraction": report["phases"]["reach"]["rated_leg_fraction"],
                   "result": str(result.relative_to(ROOT)), "trace": str(trace.relative_to(ROOT)),
                   "log": str(log.relative_to(ROOT))}
        rows.append(row)
        matrix = {"schema": "robot-learning-lab.cove50-calibrated-rgb-adaptive-v12-native.v1",
                  "body_role": plan["body_role"], "stage": args.stage,
                  "plan_sha256": sha(plan_path),
                  "pass_count": sum(r["pass"] for r in rows), "case_count": len(rows), "cases": rows}
        (stage_dir / "matrix.json").write_text(json.dumps(matrix, indent=2, allow_nan=False) + "\n")
        print(json.dumps(row), flush=True)


if __name__ == "__main__":
    main()
