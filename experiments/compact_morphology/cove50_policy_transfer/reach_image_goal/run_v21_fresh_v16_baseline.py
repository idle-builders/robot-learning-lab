"""Post-score paired v16 baseline on every disjoint v21 challenge start.

This uses the fresh set only after its v21 score is complete. It is a diagnostic,
never an additional fresh score or a change to the frozen promotion gate.
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

import numpy as np

ROOT = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(ROOT))
from experiments.compact_morphology.cove50_policy_transfer.run_squat_learned_reach_matrix import gate
from experiments.compact_morphology.cove50_policy_transfer.reach_image_goal import (
    run_v16_opened_matrix as v16,
)


OUT = ROOT / "artifacts/ape-cove50-reach-goal-channels-20261010/v21"
PLAN = OUT / "fresh-challenge-plan.json"
V21_MATRIX = OUT / "fresh-reserved/matrix.json"
DEST = OUT / "fresh-posthoc-v16-baseline"


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def run_case(case: dict, v21_row: dict, plan: dict) -> dict:
    seed = case["seed"]
    result, trace, log = (DEST / f"seed-{seed}.{suffix}" for suffix in ("json", "npz", "log"))
    sources = v16.SOURCES
    command = [sys.executable, str(sources["evaluator"]),
               "--body-xml", str(sources["body"]),
               "--calibration-model", str(sources["calibration"]),
               "--squat-checkpoint", str(sources["squat"]),
               "--reach-checkpoint", str(sources["actor"]),
               "--alternate-reach-checkpoint", str(sources["alternate_actor"]),
               "--specialist-reach-checkpoint", str(sources["specialist_actor"]),
               "--reach-controller", "learned-residual",
               "--reach-reference", str(sources["reference"]),
               "--geometry", str(sources["geometry"]),
               "--entry", "continuous", "--seed", str(seed),
               "--rail-v", str(case["rail_v"]), "--delay-ms", str(case["delay_ms"]),
               "--toy-offset-x-m", str(case["toy_offset_x_m"]),
               "--toy-offset-y-m", str(case["toy_offset_y_m"]),
               "--pitch-offset-rad", ".105", "--roll-clearance-rad", "-.25",
               "--terminal-roll-offset-rad", "-.08", "--terminal-roll-mode", "entry-error-x",
               "--reach-duration-s", "9", "--gaze-head-yaw-rad", str(case["gaze_head_yaw_rad"]),
               "--gaze-start-s", "0", "--gaze-ramp-s", "7", "--capture-stride-steps", "5",
               "--max-goal-age-s", "9", "--out", str(result), "--trace", str(trace)]
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
            raise RuntimeError(f"Posthoc v16 evaluator failed for seed {seed}: {log}")
    report = json.loads(result.read_text())
    v21_report = json.loads((ROOT / v21_row["result"]).read_text())
    with np.load(trace) as old_trace, np.load(ROOT / v21_row["trace"]) as new_trace:
        if (new_trace["qpos"].shape != old_trace["qpos"].shape or
                new_trace["action"].shape != old_trace["action"].shape):
            raise ValueError(f"Posthoc paired trace shape differs: {result}")
        qpos_delta = float(np.max(np.abs(new_trace["qpos"][375] - old_trace["qpos"][375])))
        action_delta = float(np.max(np.abs(new_trace["action"][:375] - old_trace["action"][:375])))
    x, y = report["selection_features"]["entry_error_xy_m"]
    nx, ny = v21_report["selection_features"]["entry_error_xy_m"]
    rgb_delta = max(abs(x - nx), abs(y - ny))
    mode = v16.mode_for_xy(x, y)
    hashes = plan["source_sha256"]
    selected_actor = (hashes["alternate_actor"] if mode.startswith("v16_v10") else
                      hashes["specialist_actor"] if mode == "v16_specialist_locked" else
                      hashes["actor"])
    expected_bias = -.002 if mode == "v16_bias_y_locked" else 0.0
    if (report["evaluator_sha256"] != sha(sources["evaluator"]) or
            report["body_sha256"] != hashes["body"] or
            report["selected_pointer_sha256"] != hashes["pointer"] or
            report["selected_reach_checkpoint_sha256"] != selected_actor or
            report["adaptive_variant"] != "v16_candidate" or
            report["selection_mode"] != mode or
            report["goal_bias_body_xy_m"] != [0.0, expected_bias] or
            v21_report["boundary_state_sha256"]["posture"] != report["boundary_state_sha256"]["posture"] or
            v21_report["first_action_sha256"] != report["first_action_sha256"] or
            qpos_delta > 5e-5 or action_delta > 1e-3 or rgb_delta > .0005):
        raise ValueError(f"Posthoc v16 paired identity or source mismatch: {result}")
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
    if mode == "v16_live":
        if report["image_goal_lock_after_entry"] or report["image_goal_used_update_count"] < 2:
            faults.append("live RGB updates missing")
    elif not report["image_goal_lock_after_entry"] or report["image_goal_used_update_count"] != 1:
        faults.append("entry RGB goal not locked")
    return {
        "seed": seed, "intended_stratum": case["intended_stratum"],
        "v16_selection_mode": mode, "v21_selection_mode": v21_row["selection_mode"],
        "v16_pass": not faults, "v21_pass": v21_row["pass"],
        "v16_contact_s": report["phases"]["reach"]["continuous_finger_contact_s"],
        "v21_contact_s": v21_row["contact_s"],
        "paired_reach_entry_qpos_max_abs": qpos_delta,
        "paired_pre_reach_action_max_abs": action_delta,
        "paired_rgb_entry_delta_m": rgb_delta,
        "v16_faults": faults,
        "result": str(result.relative_to(ROOT)),
        "trace": str(trace.relative_to(ROOT)),
        "log": str(log.relative_to(ROOT)),
        "v21_result": v21_row["result"],
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--workers", type=int, default=6)
    args = parser.parse_args()
    if not 1 <= args.workers <= 6:
        raise ValueError("Use one to six native evaluators")
    plan = json.loads(PLAN.read_text())
    fresh = json.loads(V21_MATRIX.read_text())
    if (not fresh["complete"] or fresh["case_count"] != 24 or
            fresh["plan_sha256"] != sha(PLAN)):
        raise ValueError("Wait for the complete frozen v21 fresh score")
    for name, path in v16.SOURCES.items():
        if name != "evaluator" and sha(path) != plan["source_sha256"][name]:
            raise ValueError(f"Source changed: {name}")
    v21_rows = {row["seed"]: row for row in fresh["cases"]}
    if set(v21_rows) != {case["seed"] for case in plan["cases"]}:
        raise ValueError("Fresh v21 coverage differs from frozen plan")
    DEST.mkdir(parents=True, exist_ok=True)
    rows = {}
    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        futures = {pool.submit(run_case, case, v21_rows[case["seed"]], plan): case
                   for case in plan["cases"]}
        for future in as_completed(futures):
            case = futures[future]
            row = future.result()
            rows[case["seed"]] = row
            ordered = [rows[case["seed"]] for case in plan["cases"] if case["seed"] in rows]
            matrix = {
                "schema": "robot-learning-lab.cove50-v21-fresh-v16-posthoc-paired-baseline.v1",
                "role": "posthoc_diagnostic_not_fresh_score",
                "fresh_plan_sha256": sha(PLAN),
                "v21_fresh_matrix_sha256": sha(V21_MATRIX),
                "v16_evaluator_sha256": sha(v16.SOURCES["evaluator"]),
                "complete": len(ordered) == 24,
                "case_count": len(ordered),
                "v16_pass_count": sum(item["v16_pass"] for item in ordered),
                "v21_pass_count": sum(item["v21_pass"] for item in ordered),
                "v21_rescues": [item["seed"] for item in ordered
                                if item["v21_pass"] and not item["v16_pass"]],
                "v21_regressions": [item["seed"] for item in ordered
                                    if item["v16_pass"] and not item["v21_pass"]],
                "cases": ordered,
            }
            (DEST / "matrix.json").write_text(json.dumps(matrix, indent=2) + "\n")
            print(json.dumps(row), flush=True)


if __name__ == "__main__":
    main()
