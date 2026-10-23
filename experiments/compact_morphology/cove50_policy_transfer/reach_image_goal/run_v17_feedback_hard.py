"""Test the saved RGB-observed feedback teacher on opened v16 hard misses."""

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

BASE = ROOT / "artifacts/ape-cove50-reach-goal-channels-20261010"
V16 = BASE / "v16"
DEST = BASE / "v17/posthoc-feedback-hard"
EVALUATOR = Path(__file__).with_name("evaluate_calibrated_goal.py")
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
TARGETS = (10409, 10417, 10403, 10419)


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def plan() -> dict:
    fresh_path = V16 / "fresh-reserved/matrix.json"
    fresh = json.loads(fresh_path.read_text())
    prior = {row["seed"]: row for row in fresh["cases"]}
    if (fresh["case_count"] != 36 or fresh["pass_count"] != 30 or
            [prior[seed]["pass"] for seed in TARGETS] != [False, False, True, True]):
        raise ValueError("Expected sealed v16 misses and pass controls")
    pointer = json.loads(SOURCES["pointer"].read_text())
    if pointer["status"] != "rejected_no_selected_mechanical_design":
        raise ValueError("Diagnostic only on the historical rejected body")
    cases = {row["seed"]: row for row in
             json.loads((V16 / "fresh-reserved-plan.json").read_text())["cases"]}
    return {
        "schema": "robot-learning-lab.cove50-v17-hard-feedback-plan.v1",
        "role": "opened_failure_and_pass_control_diagnostic_only",
        "v16_fresh_matrix_sha256": sha(fresh_path),
        "source_sha256": {name: sha(path) for name, path in SOURCES.items()},
        "cases": [cases[seed] for seed in TARGETS],
        "controller": "saved observed-RGB feedback teacher; same start and authored route; goal locked after entry",
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--freeze-only", action="store_true")
    parser.add_argument("--only-seed", type=int, choices=TARGETS)
    args = parser.parse_args()
    frozen = plan()
    DEST.mkdir(parents=True, exist_ok=True)
    plan_path = DEST / "plan.json"
    expected = (json.dumps(frozen, indent=2) + "\n").encode()
    if plan_path.exists():
        if plan_path.read_bytes() != expected:
            raise ValueError("Frozen plan differs from current source")
    else:
        plan_path.write_bytes(expected)
    if args.freeze_only:
        print(f"frozen {plan_path} sha256={sha(plan_path)}")
        return
    for case in frozen["cases"]:
        if args.only_seed is not None and case["seed"] != args.only_seed:
            continue
        stem = f"seed-{case['seed']}"
        result, trace, log = (DEST / f"{stem}.{suffix}" for suffix in ("json", "npz", "log"))
        command = [sys.executable, str(EVALUATOR),
                   "--body-xml", str(SOURCES["body"]),
                   "--calibration-model", str(SOURCES["calibration"]),
                   "--squat-checkpoint", str(SOURCES["squat"]),
                   "--reach-checkpoint", str(SOURCES["teacher"]),
                   "--reach-controller", "feedback",
                   "--reach-reference", str(SOURCES["reference"]),
                   "--geometry", str(SOURCES["geometry"]),
                   "--entry", "continuous", "--seed", str(case["seed"]),
                   "--rail-v", str(case["rail_v"]), "--delay-ms", str(case["delay_ms"]),
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
            env["PYTHONPATH"] = os.pathsep.join(
                [str(ROOT), str(ROOT / "experiments/compact_morphology/mjlab_ape/src"),
                 "/workspace/microduck_rl/src"]
                + ([env["PYTHONPATH"]] if env.get("PYTHONPATH") else []))
            with log.open("w") as stream:
                code = subprocess.run(command, cwd=ROOT, env=env, stdout=stream,
                                      stderr=subprocess.STDOUT, check=False).returncode
            if code:
                raise RuntimeError(f"Feedback evaluator failed for {stem}: {log}")
        report = json.loads(result.read_text())
        if (report["reach_checkpoint_sha256"] != frozen["source_sha256"]["teacher"] or
                report["selected_pointer_sha256"] != frozen["source_sha256"]["pointer"]):
            raise ValueError(f"Teacher or body pointer changed for {stem}")
        faults = gate(report)
        entry = next(event for event in report["image_goal_capture_events"]
                     if event["time_s"] == 7.5)
        if not entry["used_for_control"] or report["image_goal_missing_at_s"] is not None:
            faults.append("invalid RGB goal")
        print(json.dumps({"seed": case["seed"], "pass": not faults,
                          "contact_s": report["phases"]["reach"]["continuous_finger_contact_s"],
                          "faults": faults}), flush=True)


if __name__ == "__main__":
    main()
