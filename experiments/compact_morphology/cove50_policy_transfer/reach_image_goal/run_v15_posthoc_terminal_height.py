"""Freeze and replay a Jacobian-guided arm-height probe on opened v14 starts."""

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
V14 = BASE / "v14"
DEST = BASE / "v15/posthoc-terminal-height"
EVALUATOR = Path(__file__).with_name("evaluate_terminal_height_v15_diagnostic.py")
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
    "kinematic_source_trace": V14 / "fresh-reserved/seed-10304.npz",
    "evaluator": EVALUATOR,
    "pointer": ROOT / "robot/ape-current-design.json",
}
SEEDS = (10304, 10303, 10302, 10307)
SCALES = (0.5, 1.0)


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def frozen_plan() -> dict:
    fresh_path = V14 / "fresh-reserved/matrix.json"
    fresh = json.loads(fresh_path.read_text())
    if fresh["case_count"] != 20 or fresh["pass_count"] != 16:
        raise ValueError("Need complete sealed v14 fresh evidence")
    pointer = json.loads(SOURCES["pointer"].read_text())
    if pointer["status"] != "rejected_no_selected_mechanical_design":
        raise ValueError("Historical diagnostic requires rejected-body pointer")
    cases = {case["seed"]: case for case in json.loads((V14 / "plan.json").read_text())["fresh_reserved_cases"]}
    baseline = {row["seed"]: row for row in fresh["cases"]}
    if baseline[10304]["pass"] or baseline[10303]["pass"] or not baseline[10302]["pass"] or not baseline[10307]["pass"]:
        raise ValueError("Opened fail/pass-control roles changed")
    return {
        "schema": "robot-learning-lab.cove50-v15-posthoc-terminal-height-plan.v1",
        "role": "opened_failure_and_pass_control_diagnostic_only",
        "v14_fresh_matrix_sha256": sha(fresh_path),
        "source_sha256": {name: sha(path) for name, path in SOURCES.items()},
        "cases": [cases[seed] for seed in SEEDS],
        "terminal_height_scales": list(SCALES),
        "purpose": "Test a bounded terminal four-arm route correction derived from the opened 10304 fingertip Jacobian; retain the v14 RGB selector and actor weights.",
    }


def run_case(case: dict, scale: float, plan: dict, baseline: dict) -> dict:
    seed = case["seed"]
    stem = f"seed-{seed}-height-{scale:.1f}"
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
               "--entry", "continuous", "--seed", str(seed),
               "--rail-v", str(case["rail_v"]), "--delay-ms", str(case["delay_ms"]),
               "--toy-offset-x-m", str(case["toy_offset_x_m"]),
               "--toy-offset-y-m", str(case["toy_offset_y_m"]),
               "--pitch-offset-rad", ".105", "--roll-clearance-rad", "-.25",
               "--terminal-roll-offset-rad", "-.08", "--terminal-roll-mode", "entry-error-x",
               "--reach-duration-s", "9", "--gaze-head-yaw-rad", str(case["gaze_head_yaw_rad"]),
               "--gaze-start-s", "0", "--gaze-ramp-s", "7", "--capture-stride-steps", "5",
               "--max-goal-age-s", "9", "--terminal-height-scale", str(scale),
               "--posthoc-diagnostic", "--out", str(result), "--trace", str(trace)]
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
    if (report["evaluator_sha256"] != plan["source_sha256"]["evaluator"] or
            report["selected_pointer_sha256"] != plan["source_sha256"]["pointer"] or
            report["selection_mode"] != baseline[seed]["selection_mode"] or
            report["posthoc_diagnostic_terminal_height_scale"] != scale):
        raise ValueError(f"Diagnostic identity mismatch: {result}")
    faults = gate(report)
    entry = next(event for event in report["image_goal_capture_events"] if event["time_s"] == 7.5)
    if not entry["used_for_control"] or report["image_goal_missing_at_s"] is not None:
        faults.append("invalid RGB goal")
    return {
        "seed": seed, "terminal_height_scale": scale,
        "selection_mode": report["selection_mode"],
        "pass": not faults,
        "contact_s": report["phases"]["reach"]["continuous_finger_contact_s"],
        "faults": faults,
        "result": str(result.relative_to(ROOT)),
        "trace": str(trace.relative_to(ROOT)),
        "log": str(log.relative_to(ROOT)),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--freeze-only", action="store_true")
    parser.add_argument("--workers", type=int, default=3)
    args = parser.parse_args()
    if not 1 <= args.workers <= 3:
        raise ValueError("Use one to three independent native evaluators")
    plan = frozen_plan()
    DEST.mkdir(parents=True, exist_ok=True)
    plan_path = DEST / "plan.json"
    planned_bytes = (json.dumps(plan, indent=2) + "\n").encode()
    if plan_path.exists():
        if plan_path.read_bytes() != planned_bytes:
            raise ValueError("Frozen diagnostic plan differs from sources")
    else:
        plan_path.write_bytes(planned_bytes)
    if args.freeze_only:
        print(f"frozen {plan_path} sha256={sha(plan_path)}")
        return
    baseline = {row["seed"]: row for row in json.loads((V14 / "fresh-reserved/matrix.json").read_text())["cases"]}
    jobs = [(case, scale) for case in plan["cases"] for scale in SCALES]
    rows = {}
    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        futures = {pool.submit(run_case, case, scale, plan, baseline): (case["seed"], scale)
                   for case, scale in jobs}
        for future in as_completed(futures):
            key = futures[future]
            rows[key] = future.result()
            print(json.dumps(rows[key]), flush=True)
    ordered = [rows[(case["seed"], scale)] for case, scale in jobs]
    matrix = {
        "schema": "robot-learning-lab.cove50-v15-posthoc-terminal-height-native.v1",
        "role": "opened_failure_and_pass_control_diagnostic_only",
        "plan_sha256": sha(plan_path),
        "case_count": len(ordered), "pass_count": sum(row["pass"] for row in ordered),
        "cases": ordered,
    }
    (DEST / "matrix.json").write_text(json.dumps(matrix, indent=2) + "\n")


if __name__ == "__main__":
    main()
