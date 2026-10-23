"""Run a predeclared image/truth channel diagnostic on rejected Direct Frame."""

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

ARTIFACT = ROOT / "artifacts/ape-cove50-reach-goal-channels-20261010"
SOURCES = {
    "body": ROOT / "artifacts/ape-cove50-transfer-20261008/cove50-direct-training.xml",
    "squat_checkpoint": ROOT / "artifacts/ape-cove50-squat-recovery-20261009/squat-teacher-bc-blendmask.pt",
    "reach_checkpoint": ROOT / "artifacts/ape-cove50-squat-learned-reach-20261010/v7/actor.pt",
    "reach_reference": ROOT / "artifacts/ape-cove50-hand-candidate-mjlab-20261010/reach-entry/development-selected.npz",
    "geometry": ROOT / "artifacts/ape-cove50-side-reach-20261009/geometry-center.json",
    "evaluator": Path(__file__).with_name("evaluate_goal_channel_ablation.py"),
    "observer": Path(__file__).with_name("virtual_rgb_goal.py"),
}


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--plan", type=Path, default=ARTIFACT / "plan.json")
    parser.add_argument("--out-dir", type=Path, default=ARTIFACT / "development")
    args = parser.parse_args()
    plan = json.loads(args.plan.read_text())
    pointer = json.loads((ROOT / "robot/ape-current-design.json").read_text())
    if pointer["status"] != "rejected_no_selected_mechanical_design":
        raise ValueError("Historical sandbox run requires an explicit rejected-body pointer")
    for key, path in SOURCES.items():
        if sha(path) != plan["sha256"][key]:
            raise ValueError(f"Pinned {key} source changed: {path}")
    args.out_dir.mkdir(parents=True, exist_ok=True)
    rows = []
    for case in plan["cases"]:
        name = (f"seed-{case['seed']}-route-{case['route_goal_source']}"
                f"-residual-{case['residual_goal_source']}")
        result = args.out_dir / f"{name}.json"
        trace = args.out_dir / f"{name}.npz"
        log = args.out_dir / f"{name}.log"
        command = [sys.executable, str(SOURCES["evaluator"]),
                   "--body-xml", str(SOURCES["body"]),
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
                   "--lock-goal-after-entry",
                   "--route-goal-source", case["route_goal_source"],
                   "--residual-goal-source", case["residual_goal_source"],
                   "--out", str(result), "--trace", str(trace)]
        if case["route_goal_source"] == "truth" or case["residual_goal_source"] == "truth":
            command.append("--counterfactual-ablation")
        env = os.environ.copy()
        env["PYTHONPATH"] = str(ROOT) + (os.pathsep + env["PYTHONPATH"] if env.get("PYTHONPATH") else "")
        if result.exists():
            code = 0
        else:
            with log.open("w") as stream:
                code = subprocess.run(command, cwd=ROOT, env=env, stdout=stream,
                                      stderr=subprocess.STDOUT, check=False).returncode
        if code or not result.exists():
            row = {"case": name, **case, "physical_gate_pass": False,
                   "faults": [f"evaluator exit {code}"], "log": str(log.relative_to(ROOT))}
        else:
            report = json.loads(result.read_text())
            if (report["body_sha256"] != plan["sha256"]["body"] or
                    report["reach_checkpoint_sha256"] != plan["sha256"]["reach_checkpoint"] or
                    report["evaluator_sha256"] != plan["sha256"]["evaluator"] or
                    report["seed"] != case["seed"] or
                    report["route_goal_source"] != case["route_goal_source"] or
                    report["residual_goal_source"] != case["residual_goal_source"]):
                raise ValueError(f"Report identity mismatch: {result}")
            faults = gate(report)
            entry = next((x for x in report["image_goal_capture_events"] if x["time_s"] == 7.5), None)
            if not entry or not entry["used_for_control"]:
                faults.append("no valid RGB at entry")
            if report["image_goal_missing_at_s"] is not None:
                faults.append("missing or expired image goal")
            row = {"case": name, **case, "counterfactual_truth_drives_control":
                   report["counterfactual_truth_drives_control"],
                   "physical_gate_pass": not faults, "faults": faults,
                   "contact_s": report["phases"]["reach"]["continuous_finger_contact_s"],
                   "route_entry_error_x_m": report["route_entry_error_x_m"],
                   "applied_terminal_roll_offset_rad": report["applied_terminal_roll_offset_rad"],
                   "entry_image_error_mm_scoring_only":
                   None if entry is None else entry.get("xy_error_truth_mm_scoring_only"),
                   "reach_both_feet_fraction": report["phases"]["reach"]["both_feet_fraction"],
                   "reach_rated_leg_fraction": report["phases"]["reach"]["rated_leg_fraction"],
                   "result": str(result.relative_to(ROOT)),
                   "trace": str(trace.relative_to(ROOT)), "log": str(log.relative_to(ROOT))}
        rows.append(row)
        matrix = {"schema": "robot-learning-lab.cove50-goal-channel-diagnostic.v1",
                  "body_role": "rejected_historical_direct_frame_sandbox",
                  "plan": str(args.plan.relative_to(ROOT)),
                  "cases": rows}
        (args.out_dir / "matrix.json").write_text(json.dumps(matrix, indent=2) + "\n")
        print(json.dumps(row), flush=True)


if __name__ == "__main__":
    main()
