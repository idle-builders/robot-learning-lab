"""Probe an RGB-observed feedback teacher on unresolved V33 reach regions."""

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
OUT = BASE / "v34"
PLAN = OUT / "feedback-gap-plan.json"
DEST = OUT / "feedback-gap"
EVALUATOR = Path(__file__).with_name("evaluate_calibrated_goal.py")
SOURCE_PLAN = BASE / "v32/prior-failure-screen-plan.json"
SOURCE_MATRIX = BASE / "v33/forced-route-screen/matrix.json"
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
TARGETS = (10530, 10538, 10513, 10539, 10534)


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def freeze() -> dict:
    previous = json.loads(SOURCE_MATRIX.read_text())
    rows = {(row["seed"], row["route"]): row for row in previous["cases"]}
    if (not previous["complete"] or previous["case_count"] != 15 or
            any(rows[(seed, route)]["pass"] for seed, route in
                ((10530, "live"), (10530, "v10"), (10530, "specialist"),
                 (10538, "live"), (10538, "v10"), (10538, "specialist"),
                 (10513, "live"))) or
            not rows[(10539, "specialist")]["pass"] or
            not rows[(10534, "locked")]["pass"]):
        raise ValueError("Expected complete V33 gap and controls")
    cases = {row["seed"]: row for row in json.loads(SOURCE_PLAN.read_text())["cases"]}
    plan = {
        "schema": "robot-learning-lab.cove50-v34-feedback-gap-plan.v1",
        "role": "opened_RGB_feedback_teacher_diagnostic_not_fresh",
        "body_role": "rejected_historical_direct_frame_sandbox",
        "source_plan_sha256": sha(SOURCE_PLAN),
        "source_matrix_sha256": sha(SOURCE_MATRIX),
        "source_sha256": {name: sha(path) for name, path in SOURCES.items()},
        "runner_sha256": sha(Path(__file__)),
        "case_count": len(TARGETS),
        "cases": [{"seed": seed, "role": "route_gap" if seed in (10530, 10538) else
                   "unstable_old_rescue" if seed == 10513 else "pass_control",
                   **{key: cases[seed][key] for key in
                      ("rail_v", "delay_ms", "toy_offset_x_m", "toy_offset_y_m", "gaze_head_yaw_rad")}}
                  for seed in TARGETS],
        "qualification": "Saved feedback teacher sees calibrated RGB, not simulator cube truth. Native success can supply imitation data but does not qualify an automatic learned controller, a new score, or a selected physical body.",
    }
    OUT.mkdir(parents=True, exist_ok=True)
    encoded = json.dumps(plan, indent=2).encode() + b"\n"
    if PLAN.exists():
        if PLAN.read_bytes() != encoded:
            raise ValueError("Frozen V34 plan differs from current sources")
    else:
        PLAN.write_bytes(encoded)
    return plan


def run_case(case: dict, plan: dict) -> dict:
    seed = case["seed"]
    result, trace, log = (DEST / f"seed-{seed}.{suffix}" for suffix in ("json", "npz", "log"))
    command = [sys.executable, str(EVALUATOR),
               "--body-xml", str(SOURCES["body"]),
               "--calibration-model", str(SOURCES["calibration"]),
               "--squat-checkpoint", str(SOURCES["squat"]),
               "--reach-checkpoint", str(SOURCES["teacher"]),
               "--reach-controller", "feedback",
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
               "--max-goal-age-s", "9", "--lock-goal-after-entry",
               "--out", str(result), "--trace", str(trace)]
    if not result.exists() or not trace.exists():
        env = os.environ.copy()
        env["PYTHONPATH"] = os.pathsep.join(
            [str(ROOT), str(ROOT / "experiments/compact_morphology/mjlab_ape/src"),
             "/root/microduck_rl/src", "/workspace/microduck_rl/src"]
            + ([env["PYTHONPATH"]] if env.get("PYTHONPATH") else []))
        with log.open("w") as stream:
            code = subprocess.run(command, cwd=ROOT, env=env, stdout=stream,
                                  stderr=subprocess.STDOUT, check=False).returncode
        if code:
            raise RuntimeError(f"V34 native evaluator failed for seed {seed}: {log}")
    report = json.loads(result.read_text())
    hashes = plan["source_sha256"]
    if (report["evaluator_sha256"] != hashes["evaluator"] or
            report["body_sha256"] != hashes["body"] or
            report["selected_pointer_sha256"] != hashes["pointer"] or
            report["reach_checkpoint_sha256"] != hashes["teacher"]):
        raise ValueError(f"V34 identity mismatch: {result}")
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
    return {"seed": seed, "role": case["role"], "pass": not faults,
            "contact_s": report["phases"]["reach"]["continuous_finger_contact_s"],
            "min_cube_to_hand_distance_m": report["phases"]["reach"]["min_cube_to_hand_distance_m"],
            "entry_goal_error_mm_scoring_only": None if entry is None else
                                                entry.get("xy_error_truth_mm_scoring_only"),
            "faults": faults,
            "result": str(result.relative_to(ROOT)),
            "trace": str(trace.relative_to(ROOT)),
            "log": str(log.relative_to(ROOT))}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--freeze-only", action="store_true")
    parser.add_argument("--workers", type=int, default=4)
    args = parser.parse_args()
    if not 1 <= args.workers <= 6:
        raise ValueError("Use one to six native evaluators")
    plan = freeze()
    if args.freeze_only:
        print(f"frozen {PLAN} sha256={sha(PLAN)}")
        return
    DEST.mkdir(parents=True, exist_ok=True)
    rows = {}
    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        futures = {pool.submit(run_case, case, plan): case["seed"] for case in plan["cases"]}
        for future in as_completed(futures):
            row = future.result()
            rows[row["seed"]] = row
            ordered = [rows[case["seed"]] for case in plan["cases"] if case["seed"] in rows]
            matrix = {"schema": "robot-learning-lab.cove50-v34-feedback-gap.v1",
                      "role": "opened_RGB_feedback_teacher_diagnostic_not_fresh",
                      "body_role": "rejected_historical_direct_frame_sandbox",
                      "plan_sha256": sha(PLAN),
                      "complete": len(ordered) == len(plan["cases"]),
                      "case_count": len(ordered),
                      "pass_count": sum(item["pass"] for item in ordered),
                      "cases": ordered}
            (DEST / "matrix.json").write_text(json.dumps(matrix, indent=2) + "\n")
            print(json.dumps({key: row[key] for key in
                              ("seed", "role", "pass", "contact_s", "faults")}), flush=True)


if __name__ == "__main__":
    main()
