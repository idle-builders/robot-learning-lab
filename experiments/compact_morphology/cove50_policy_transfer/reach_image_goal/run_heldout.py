"""Run the frozen virtual-RGB-goal controller on disjoint native starts."""

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

OUT = ROOT / "artifacts/ape-cove50-reach-image-goal-20261010"
BODY = ROOT / "artifacts/ape-cove50-transfer-20261008/cove50-direct-training.xml"
ACTOR = ROOT / "artifacts/ape-cove50-squat-learned-reach-20261010/v7/actor.pt"
FREEZE = ROOT / "artifacts/ape-cove50-reach-viewpoint-20261010/heldout-freeze.json"
EVALUATOR = Path(__file__).with_name("evaluate_image_goal_handoff.py")
OBSERVER = Path(__file__).with_name("virtual_rgb_goal.py")


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--plan", type=Path, default=OUT / "heldout-plan.json")
    parser.add_argument("--out-dir", type=Path, default=OUT / "heldout")
    args = parser.parse_args()
    args.plan = args.plan.resolve()
    args.out_dir = args.out_dir.resolve()
    plan = json.loads(args.plan.read_text())
    for path, key in ((ROOT / "robot/ape-current-design.json", "selected_pointer_sha256"),
                      (BODY, "selected_body_sha256"), (ACTOR, "saved_reach_actor_sha256"),
                      (FREEZE, "virtual_camera_freeze_sha256"),
                      (EVALUATOR, "evaluator_sha256"), (OBSERVER, "observer_sha256")):
        assert sha(path) == plan[key], f"Frozen source changed: {path}"
    fixed = plan["fixed_control"]
    args.out_dir.mkdir(parents=True, exist_ok=True)
    rows = []
    for case in plan["cases"]:
        stem = f"seed-{case['seed']}"
        result = args.out_dir / f"{stem}.json"
        trace = args.out_dir / f"{stem}.npz"
        log = args.out_dir / f"{stem}.log"
        command = [sys.executable, str(EVALUATOR),
                   "--body-xml", str(BODY),
                   "--squat-checkpoint", str(ROOT / "artifacts/ape-cove50-squat-recovery-20261009/squat-teacher-bc-blendmask.pt"),
                   "--reach-checkpoint", str(ACTOR),
                   "--reach-controller", "learned-residual",
                   "--reach-reference", str(ROOT / "artifacts/ape-cove50-hand-candidate-mjlab-20261010/reach-entry/development-selected.npz"),
                   "--geometry", str(ROOT / "artifacts/ape-cove50-side-reach-20261009/geometry-center.json"),
                   "--entry", "continuous", "--seed", str(case["seed"]),
                   "--rail-v", str(case["rail_v"]), "--delay-ms", str(case["delay_ms"]),
                   "--toy-offset-x-m", str(case["toy_offset_x_m"]),
                   "--toy-offset-y-m", str(case["toy_offset_y_m"]),
                   "--pitch-offset-rad", ".105", "--roll-clearance-rad", "-.25",
                   "--terminal-roll-offset-rad", "-.08", "--terminal-roll-mode", "entry-error-x",
                   "--reach-duration-s", "9", "--gaze-head-yaw-rad", str(fixed["gaze_head_yaw_rad"]),
                   "--gaze-start-s", str(fixed["gaze_start_s"]),
                   "--gaze-ramp-s", str(fixed["gaze_ramp_s"]),
                   "--capture-stride-steps", str(fixed["capture_stride_steps"]),
                   "--max-goal-age-s", str(fixed["max_goal_age_s"]),
                   "--lock-goal-after-entry",
                   "--out", str(result), "--trace", str(trace)]
        env = os.environ.copy()
        env["PYTHONPATH"] = str(ROOT) + (os.pathsep + env["PYTHONPATH"] if env.get("PYTHONPATH") else "")
        if not result.exists():
            with log.open("w") as stream:
                completed = subprocess.run(command, cwd=ROOT, env=env, stdout=stream,
                                           stderr=subprocess.STDOUT, check=False)
            completed_code = completed.returncode
        else:
            completed_code = 0
        if completed_code or not result.exists():
            row = {"case": stem, **case, "pass": False,
                   "faults": [f"evaluator exit {completed_code}"],
                   "log": str(log.relative_to(ROOT))}
        else:
            report = json.loads(result.read_text())
            assert report["seed"] == case["seed"]
            assert report["rail_v"] == case["rail_v"] and report["delay_ms"] == case["delay_ms"]
            assert report["toy_offset_b_m"] == [case["toy_offset_x_m"], case["toy_offset_y_m"]]
            assert report["body_sha256"] == plan["selected_body_sha256"]
            assert report["reach_checkpoint_sha256"] == plan["saved_reach_actor_sha256"]
            assert report["evaluator_sha256"] == plan["evaluator_sha256"]
            faults = gate(report)
            captures = report["image_goal_capture_events"]
            entry = next((event for event in captures if event["time_s"] == 7.5), None)
            yaw = next((r["head_yaw_rad"] for r in report["timeseries"] if r["time_s"] == 7.5), None)
            if not entry or not entry["used_for_control"]:
                faults.append("no valid RGB goal used at reach entry")
            if report["image_goal_used_update_count"] != 1 or not report["image_goal_lock_after_entry"]:
                faults.append("control goal was not frozen from entry image")
            if report["image_goal_missing_at_s"] is not None:
                faults.append("missing or expired image goal")
            if yaw is None or abs(yaw - fixed["gaze_head_yaw_rad"]) > .05:
                faults.append("head yaw more than 0.05 rad from target at reach entry")
            row = {"case": stem, **case, "pass": not faults, "faults": faults,
                   "entry_goal_error_mm_scoring_only": None if entry is None else entry.get("xy_error_truth_mm_scoring_only"),
                   "entry_orange_pixels": None if entry is None else entry["orange_pixels"],
                   "entry_head_yaw_rad": yaw,
                   "accepted_capture_count": report["image_goal_accepted_capture_count"],
                   "used_update_count": report["image_goal_used_update_count"],
                   "contact_s": report["phases"]["reach"]["continuous_finger_contact_s"],
                   "reach_both_feet_fraction": report["phases"]["reach"]["both_feet_fraction"],
                   "reach_rated_leg_fraction": report["phases"]["reach"]["rated_leg_fraction"],
                   "max_abs_unbounded_action": report["max_abs_unbounded_action"],
                   "result": str(result.relative_to(ROOT)), "trace": str(trace.relative_to(ROOT)),
                   "log": str(log.relative_to(ROOT))}
        rows.append(row)
        print(json.dumps(row), flush=True)
        matrix = {"schema": "robot-learning-lab.cove50-virtual-rgb-entry-lock-heldout-matrix.v1",
                  "plan": str(args.plan.relative_to(ROOT)),
                  "native_pass_count": sum(r["pass"] for r in rows),
                  "native_sample_count": len(rows), "cases": rows}
        (args.out_dir / "matrix.json").write_text(json.dumps(matrix, indent=2, allow_nan=False) + "\n")
    print(json.dumps({"native_pass_count": matrix["native_pass_count"],
                      "native_sample_count": len(rows)}), flush=True)


if __name__ == "__main__":
    main()
