"""Screen saved reach routes on V32's unresolved opened failures.

Forced routes are diagnostic teacher visits. They cannot qualify an automatic
selector, and a prescribed goal bias cannot be used as a deployable input.
"""

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
from experiments.compact_morphology.cove50_policy_transfer.reach_image_goal import run_v16_opened_matrix as v16

BASE = ROOT / "artifacts/ape-cove50-reach-goal-channels-20261010"
OUT = BASE / "v33"
PLAN = OUT / "forced-route-screen-plan.json"
DEST = OUT / "forced-route-screen"
EVALUATOR = Path(__file__).with_name("evaluate_teacher_probe_v18.py")
SOURCE_PLAN = BASE / "v32/prior-failure-screen-plan.json"
SOURCE_MATRIX = BASE / "v32/prior-failure-screen/matrix.json"

ROUTES = {
    "locked": ("v16_locked", 0.0, 0.0),
    "live": ("v16_live", 0.0, 0.0),
    "v10": ("v16_v10_locked", 0.0, 0.0),
    "height": ("v16_v10_height_locked", 0.0, 0.0),
    "specialist": ("v16_specialist_locked", 0.0, 0.0),
    "v10_x2_teacher": ("v16_v10_locked", 2.0, 0.0),
}
# Twelve previously untried goal-preserving visits, plus three prior rescues.
VISITS = [
    (10530, "live"), (10530, "v10"), (10530, "specialist"),
    (10534, "locked"), (10534, "v10"), (10534, "specialist"),
    (10538, "live"), (10538, "v10"), (10538, "specialist"),
    (10539, "live"), (10539, "v10"), (10539, "specialist"),
    (10508, "v10_x2_teacher"), (10513, "live"), (10433, "specialist"),
]


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def freeze() -> dict:
    source = json.loads(SOURCE_PLAN.read_text())
    matrix = json.loads(SOURCE_MATRIX.read_text())
    if (not matrix["complete"] or matrix["case_count"] != 9 or
            {row["seed"] for row in matrix["cases"] if not row["pass"]} !=
            {10508, 10513, 10530, 10534, 10538, 10539, 10433}):
        raise ValueError("V32 unresolved failure set changed")
    cases = {case["seed"]: case for case in source["cases"]}
    plan = {
        "schema": "robot-learning-lab.cove50-v33-forced-route-screen-plan.v1",
        "role": "opened_prior_failure_diagnostic_not_fresh_not_automatic",
        "body_role": "rejected_historical_direct_frame_sandbox",
        "source_plan_sha256": sha(SOURCE_PLAN),
        "source_matrix_sha256": sha(SOURCE_MATRIX),
        "source_sha256": {name: sha(path) for name, path in v16.SOURCES.items()},
        "evaluator_sha256": sha(EVALUATOR),
        "runner_sha256": sha(Path(__file__)),
        "case_count": len(VISITS),
        "cases": [
            {"seed": seed, "route": route, "force_selection_mode": ROUTES[route][0],
             "extra_goal_bias_body_xy_mm": list(ROUTES[route][1:]),
             "role": "known_rescue_control" if seed in (10508, 10513, 10433) else
                     "unresolved_v32_failure",
             **{key: cases[seed][key] for key in
                ("rail_v", "delay_ms", "toy_offset_x_m", "toy_offset_y_m", "gaze_head_yaw_rad")}}
            for seed, route in VISITS
        ],
        "qualification": "Experimenter-forced actor or route, with one prescribed +2 mm goal bias control. Contact is a native diagnostic only. No automatic policy, retention gate, or fresh score is earned.",
    }
    OUT.mkdir(parents=True, exist_ok=True)
    encoded = json.dumps(plan, indent=2).encode() + b"\n"
    if PLAN.exists():
        if PLAN.read_bytes() != encoded:
            raise ValueError("Frozen V33 plan differs from current sources")
    else:
        PLAN.write_bytes(encoded)
    return plan


def run_case(case: dict, plan: dict) -> dict:
    stem = f"seed-{case['seed']}-{case['route']}"
    result, trace, log = (DEST / f"{stem}.{suffix}" for suffix in ("json", "npz", "log"))
    sources = v16.SOURCES
    command = [sys.executable, str(EVALUATOR),
               "--body-xml", str(sources["body"]),
               "--calibration-model", str(sources["calibration"]),
               "--squat-checkpoint", str(sources["squat"]),
               "--reach-checkpoint", str(sources["actor"]),
               "--alternate-reach-checkpoint", str(sources["alternate_actor"]),
               "--specialist-reach-checkpoint", str(sources["specialist_actor"]),
               "--force-selection-mode", case["force_selection_mode"],
               "--extra-goal-bias-body-x-mm", str(case["extra_goal_bias_body_xy_mm"][0]),
               "--extra-goal-bias-body-y-mm", str(case["extra_goal_bias_body_xy_mm"][1]),
               "--reach-controller", "learned-residual",
               "--reach-reference", str(sources["reference"]),
               "--geometry", str(sources["geometry"]),
               "--entry", "continuous", "--seed", str(case["seed"]),
               "--rail-v", str(case["rail_v"]), "--delay-ms", str(case["delay_ms"]),
               "--toy-offset-x-m", str(case["toy_offset_x_m"]),
               "--toy-offset-y-m", str(case["toy_offset_y_m"]),
               "--pitch-offset-rad", ".105", "--roll-clearance-rad", "-.25",
               "--terminal-roll-offset-rad", "-.08", "--terminal-roll-mode", "entry-error-x",
               "--reach-duration-s", "9", "--gaze-head-yaw-rad", str(case["gaze_head_yaw_rad"]),
               "--gaze-start-s", "0", "--gaze-ramp-s", "7", "--capture-stride-steps", "5",
               "--max-goal-age-s", "9", "--out", str(result), "--trace", str(trace)]
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
            raise RuntimeError(f"V33 native evaluator failed for {stem}: {log}")
    report = json.loads(result.read_text())
    hashes = plan["source_sha256"]
    if (report["evaluator_sha256"] != plan["evaluator_sha256"] or
            report["body_sha256"] != hashes["body"] or
            report["selected_pointer_sha256"] != hashes["pointer"] or
            report["reach_checkpoint_sha256"] != hashes["actor"] or
            report["alternate_reach_checkpoint_sha256"] != hashes["alternate_actor"] or
            report["specialist_reach_checkpoint_sha256"] != hashes["specialist_actor"] or
            report["selection_mode"] != case["force_selection_mode"] or
            report["extra_goal_bias_body_xy_mm"] != case["extra_goal_bias_body_xy_mm"]):
        raise ValueError(f"V33 identity mismatch: {result}")
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
    if case["force_selection_mode"] == "v16_live":
        if report["image_goal_lock_after_entry"] or report["image_goal_used_update_count"] < 2:
            faults.append("live RGB updates missing")
    elif not report["image_goal_lock_after_entry"] or report["image_goal_used_update_count"] != 1:
        faults.append("entry RGB goal not locked")
    return {"seed": case["seed"], "route": case["route"], "role": case["role"],
            "forced_mode": case["force_selection_mode"],
            "extra_goal_bias_body_xy_mm": case["extra_goal_bias_body_xy_mm"],
            "pass": not faults,
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
        futures = {pool.submit(run_case, case, plan): (case["seed"], case["route"])
                   for case in plan["cases"]}
        for future in as_completed(futures):
            row = future.result()
            rows[future] = row
            ordered = [rows[futures_inv] for futures_inv in futures if futures_inv in rows]
            matrix = {
                "schema": "robot-learning-lab.cove50-v33-forced-route-screen.v1",
                "role": "opened_prior_failure_diagnostic_not_fresh_not_automatic",
                "body_role": "rejected_historical_direct_frame_sandbox",
                "plan_sha256": sha(PLAN),
                "complete": len(ordered) == len(plan["cases"]),
                "case_count": len(ordered),
                "pass_count": sum(item["pass"] for item in ordered),
                "cases": ordered,
            }
            (DEST / "matrix.json").write_text(json.dumps(matrix, indent=2) + "\n")
            print(json.dumps({key: row[key] for key in
                              ("seed", "route", "pass", "contact_s", "faults")}), flush=True)


if __name__ == "__main__":
    main()
