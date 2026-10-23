"""Run bounded goal-bias probes on opened v16 hard misses and pass controls."""

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
DEST = BASE / "v17/posthoc-hard-goal-bias"
EVALUATOR = Path(__file__).with_name("evaluate_goal_bias_v15_diagnostic.py")
SOURCES = {
    "body": ROOT / "artifacts/ape-cove50-transfer-20261008/cove50-direct-training.xml",
    "squat": ROOT / "artifacts/ape-cove50-squat-recovery-20261009/squat-teacher-bc-blendmask.pt",
    "actor": BASE / "v13/actor.pt",
    "alternate_actor": BASE / "v10/actor.pt",
    "reference": ROOT / "artifacts/ape-cove50-hand-candidate-mjlab-20261010/reach-entry/development-selected.npz",
    "geometry": ROOT / "artifacts/ape-cove50-side-reach-20261009/geometry-center.json",
    "calibration": BASE / "rgb-calibration-model.npz",
    "camera_freeze": ROOT / "artifacts/ape-cove50-reach-viewpoint-20261010/heldout-freeze.json",
    "virtual_rgb_observer": Path(__file__).with_name("virtual_rgb_goal.py"),
    "calibrated_rgb_observer": Path(__file__).with_name("calibrated_virtual_rgb_goal.py"),
    "evaluator": EVALUATOR,
    "pointer": ROOT / "robot/ape-current-design.json",
}
BIASES_MM = ((-2.0, 0.0), (2.0, 0.0), (0.0, -2.0), (0.0, 2.0))
TARGETS = (10409, 10417, 10403, 10419)


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def frozen_plan() -> dict:
    fresh_path = V16 / "fresh-reserved/matrix.json"
    fresh = json.loads(fresh_path.read_text())
    if fresh["case_count"] != 36 or fresh["pass_count"] != 30 or not fresh["complete"]:
        raise ValueError("Expected sealed complete v16 fresh matrix")
    prior = {row["seed"]: row for row in fresh["cases"]}
    if (prior[10409]["pass"] or prior[10417]["pass"] or
            not prior[10403]["pass"] or not prior[10419]["pass"]):
        raise ValueError("Opened hard-miss and pass-control roles changed")
    pointer = json.loads(SOURCES["pointer"].read_text())
    if pointer["status"] != "rejected_no_selected_mechanical_design":
        raise ValueError("This is a rejected-body diagnostic only")
    v16_plan = json.loads((V16 / "fresh-reserved-plan.json").read_text())
    cases = {row["seed"]: row for row in v16_plan["cases"]}
    return {
        "schema": "robot-learning-lab.cove50-v17-hard-goal-bias-plan.v1",
        "role": "opened_failure_and_pass_control_diagnostic_only",
        "v16_fresh_matrix_sha256": sha(fresh_path),
        "source_sha256": {name: sha(path) for name, path in SOURCES.items()},
        "cases": [cases[seed] for seed in TARGETS],
        "biases_body_xy_mm": [list(pair) for pair in BIASES_MM],
        "selector": "v14 selector and saved actors, equivalent to v16 branches on these four observed entries; RGB image selector before bias",
        "goal_channels": "prescribed offset after selector, applied to route and learned actor observations",
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--freeze-only", action="store_true")
    parser.add_argument("--only-seed", type=int, choices=TARGETS,
                        help="Run one opened seed for parallel compute")
    parser.add_argument("--skip-matrix", action="store_true",
                        help="Worker writes unique reports only; coordinator consolidates matrix")
    args = parser.parse_args()
    plan = frozen_plan()
    DEST.mkdir(parents=True, exist_ok=True)
    plan_path = DEST / "plan.json"
    planned_bytes = (json.dumps(plan, indent=2) + "\n").encode()
    if plan_path.exists():
        if plan_path.read_bytes() != planned_bytes:
            raise ValueError("Frozen diagnostic plan differs from current sources")
    else:
        plan_path.write_bytes(planned_bytes)
    if args.freeze_only:
        print(f"frozen {plan_path} sha256={sha(plan_path)}")
        return
    rows = []
    v16_rows = {row["seed"]: row for row in
                json.loads((V16 / "fresh-reserved/matrix.json").read_text())["cases"]}
    for case in plan["cases"]:
        if args.only_seed is not None and case["seed"] != args.only_seed:
            continue
        for bx, by in BIASES_MM:
            label = f"x{bx:+.0f}-y{by:+.0f}"
            stem = f"seed-{case['seed']}-{label}"
            result, trace, log = (DEST / f"{stem}.{suffix}" for suffix in ("json", "npz", "log"))
            command = [sys.executable, str(EVALUATOR),
                       "--body-xml", str(SOURCES["body"]),
                       "--calibration-model", str(SOURCES["calibration"]),
                       "--squat-checkpoint", str(SOURCES["squat"]),
                       "--reach-checkpoint", str(SOURCES["actor"]),
                       "--alternate-reach-checkpoint", str(SOURCES["alternate_actor"]),
                       "--reach-controller", "learned-residual",
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
                       "--max-goal-age-s", "9", "--goal-bias-body-x-mm", str(bx),
                       "--goal-bias-body-y-mm", str(by), "--posthoc-diagnostic",
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
                    raise RuntimeError(f"Evaluator failed for {stem}: {log}")
            report = json.loads(result.read_text())
            expected_mode = {"v16_v10_locked": "v10_locked", "v16_locked": "v14_locked"}[
                v16_rows[case["seed"]]["selection_mode"]]
            if (report["evaluator_sha256"] != plan["source_sha256"]["evaluator"] or
                    report["selection_mode"] != expected_mode or
                    report["posthoc_diagnostic_goal_bias_body_xy_mm"] != [bx, by] or
                    report["selected_pointer_sha256"] != plan["source_sha256"]["pointer"]):
                raise ValueError(f"Diagnostic identity mismatch: {result}")
            faults = gate(report)
            entry = next(event for event in report["image_goal_capture_events"]
                         if event["time_s"] == 7.5)
            if not entry["used_for_control"] or report["image_goal_missing_at_s"] is not None:
                faults.append("invalid RGB goal")
            row = {
                "seed": case["seed"], "bias_body_xy_mm": [bx, by],
                "selection_mode": report["selection_mode"],
                "pass": not faults,
                "contact_s": report["phases"]["reach"]["continuous_finger_contact_s"],
                "faults": faults,
                "result": str(result.relative_to(ROOT)),
                "trace": str(trace.relative_to(ROOT)),
                "log": str(log.relative_to(ROOT)),
            }
            rows.append(row)
            matrix = {
                "schema": "robot-learning-lab.cove50-v17-hard-goal-bias-native.v1",
                "role": "opened_failure_and_pass_control_diagnostic_only",
                "plan_sha256": sha(plan_path),
                "case_count": len(rows), "pass_count": sum(row["pass"] for row in rows),
                "cases": rows,
            }
            if not args.skip_matrix:
                (DEST / "matrix.json").write_text(json.dumps(matrix, indent=2) + "\n")
            print(json.dumps(row), flush=True)


if __name__ == "__main__":
    main()
