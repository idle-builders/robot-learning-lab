"""Post hoc feedback-teacher replay of the four opened RGB reach misses."""

from __future__ import annotations

import json
import os
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(ROOT))
from experiments.compact_morphology.cove50_policy_transfer.run_squat_learned_reach_matrix import gate

OUT = ROOT / "artifacts/ape-cove50-reach-goal-channels-20261010"
EVALUATOR = Path(__file__).with_name("evaluate_calibrated_goal.py")
BODY = ROOT / "artifacts/ape-cove50-transfer-20261008/cove50-direct-training.xml"
SQUAT = ROOT / "artifacts/ape-cove50-squat-recovery-20261009/squat-teacher-bc-blendmask.pt"
FEEDBACK = ROOT / "artifacts/ape-cove50-side-reach-20261009/reach-feedback-v1/controller.pt"
REFERENCE = ROOT / "artifacts/ape-cove50-hand-candidate-mjlab-20261010/reach-entry/development-selected.npz"
GEOMETRY = ROOT / "artifacts/ape-cove50-side-reach-20261009/geometry-center.json"
CALIBRATION = OUT / "rgb-calibration-model.npz"


def main() -> None:
    plan = json.loads((OUT / "feedback-feasibility-plan.json").read_text())
    stage_dir = OUT / "posthoc-feedback"
    stage_dir.mkdir(parents=True, exist_ok=True)
    rows = []
    for case in plan["cases"]:
        stem = f"seed-{case['seed']}"
        result, trace, log = (stage_dir / f"{stem}.{suffix}" for suffix in ("json", "npz", "log"))
        command = [sys.executable, str(EVALUATOR),
                   "--body-xml", str(BODY), "--calibration-model", str(CALIBRATION),
                   "--squat-checkpoint", str(SQUAT), "--reach-checkpoint", str(FEEDBACK),
                   "--reach-controller", "feedback", "--reach-reference", str(REFERENCE),
                   "--geometry", str(GEOMETRY), "--entry", "continuous",
                   "--seed", str(case["seed"]), "--rail-v", str(case["rail_v"]),
                   "--delay-ms", str(case["delay_ms"]),
                   "--toy-offset-x-m", str(case["toy_offset_x_m"]),
                   "--toy-offset-y-m", str(case["toy_offset_y_m"]),
                   "--pitch-offset-rad", ".105", "--roll-clearance-rad", "-.25",
                   "--terminal-roll-offset-rad", "-.08", "--terminal-roll-mode", "entry-error-x",
                   "--reach-duration-s", "9", "--gaze-head-yaw-rad", str(case["gaze_head_yaw_rad"]),
                   "--gaze-start-s", "0", "--gaze-ramp-s", "7", "--capture-stride-steps", "5",
                   "--max-goal-age-s", "9", "--lock-goal-after-entry",
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
            faults = gate(report)
            entry = next((event for event in report["image_goal_capture_events"]
                          if event["time_s"] == 7.5), None)
            if not entry or not entry["used_for_control"] or entry["orange_pixels"] < 10:
                faults.append("no valid RGB goal at entry")
            if entry and (entry.get("xy_error_truth_mm_scoring_only") is None or
                          entry["xy_error_truth_mm_scoring_only"] > 10):
                faults.append("entry RGB error above 10 mm")
            row = {"seed": case["seed"], "pass": not faults, "faults": faults,
                   "contact_s": report["phases"]["reach"]["continuous_finger_contact_s"],
                   "entry_goal_error_mm_scoring_only": None if entry is None else entry.get("xy_error_truth_mm_scoring_only"),
                   "result": str(result.relative_to(ROOT)), "trace": str(trace.relative_to(ROOT)),
                   "log": str(log.relative_to(ROOT))}
        rows.append(row)
        matrix = {"schema": "robot-learning-lab.cove50-posthoc-feedback-feasibility.v1",
                  "body_role": "rejected_historical_direct_frame_sandbox", "stage": "posthoc-feedback",
                  "plan": str((OUT / "feedback-feasibility-plan.json").relative_to(ROOT)),
                  "pass_count": sum(r["pass"] for r in rows), "case_count": len(rows),
                  "cases": rows}
        (stage_dir / "matrix.json").write_text(json.dumps(matrix, indent=2, allow_nan=False) + "\n")
        print(json.dumps(row), flush=True)


if __name__ == "__main__":
    main()
