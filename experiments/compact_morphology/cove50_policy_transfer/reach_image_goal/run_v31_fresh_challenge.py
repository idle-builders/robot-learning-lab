"""Score a frozen disjoint native challenge for the automatic V31 RGB reach."""

from __future__ import annotations

import argparse
from collections import Counter
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
from experiments.compact_morphology.cove50_policy_transfer.reach_image_goal import (
    run_v16_opened_matrix as v16,
    run_v21_opened_gate as v21,
)

BASE = ROOT / "artifacts/ape-cove50-reach-goal-channels-20261010"
OUT = BASE / "v31"
PLAN = OUT / "fresh-challenge-plan.json"
OPENED_PLAN = OUT / "opened-gate-plan.json"
OPENED_MATRIX = OUT / "opened-native/matrix.json"
DEST = OUT / "fresh-reserved"
EVALUATOR = Path(__file__).with_name("evaluate_template_goal_v31.py")
TEMPLATE = Path(__file__).with_name("template_virtual_rgb_goal_v30.py")


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def frozen_plan() -> dict:
    plan = json.loads(PLAN.read_text())
    opened = json.loads(OPENED_MATRIX.read_text())
    if (plan["opened_plan_sha256"] != sha(OPENED_PLAN) or
            not opened["complete"] or not opened["advance_gate_pass"] or
            opened["case_count"] != 24 or opened["plan_sha256"] != sha(OPENED_PLAN) or
            len(plan["cases"]) != 32 or plan["case_count"] != 32 or
            len({case["seed"] for case in plan["cases"]}) != 32 or
            {case["seed"] for case in plan["cases"]} != set(range(10701, 10733)) or
            any(case["gaze_head_yaw_rad"] != -1.5 for case in plan["cases"])):
        raise ValueError("V31 opened native gate must pass before disjoint fresh score")
    sources = {key: sha(path) for key, path in v16.SOURCES.items() if key != "evaluator"}
    sources.update({"evaluator": sha(EVALUATOR), "template": sha(TEMPLATE)})
    if sources != plan["source_sha256"] or plan["runner_sha256"] != sha(Path(__file__)):
        raise ValueError("Fresh native source differs from freeze")
    return plan


def run_case(case: dict, plan: dict) -> dict:
    seed = case["seed"]
    result, trace, log = (DEST / f"seed-{seed}.{suffix}" for suffix in ("json", "npz", "log"))
    sources = v16.SOURCES
    command = [sys.executable, str(EVALUATOR),
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
               "--reach-duration-s", "9", "--gaze-head-yaw-rad", "-1.5",
               "--gaze-start-s", "0", "--gaze-ramp-s", "7", "--capture-stride-steps", "5",
               "--max-goal-age-s", "9", "--out", str(result), "--trace", str(trace)]
    if not result.exists():
        env = os.environ.copy()
        env["PYTHONPATH"] = os.pathsep.join([
            str(ROOT), str(ROOT / "experiments/compact_morphology/mjlab_ape/src"),
            "/workspace/microduck_rl/src"] +
            ([env["PYTHONPATH"]] if env.get("PYTHONPATH") else []))
        with log.open("w") as stream:
            code = subprocess.run(command, cwd=ROOT, env=env, stdout=stream,
                                  stderr=subprocess.STDOUT, check=False).returncode
        if code:
            raise RuntimeError(f"V31 fresh evaluator failed for {seed}: {log}")
    report = json.loads(result.read_text())
    hashes = plan["source_sha256"]
    x, y = report["selection_features"]["entry_error_xy_m"]
    mode = ("v16_v10_height_locked" if x < -.012 and y < .002
            else v21.mode_for_xy(x, y))
    if (report["evaluator_sha256"] != hashes["evaluator"] or
            report["template_source_sha256"] != hashes["template"] or
            report["body_sha256"] != hashes["body"] or
            report["selected_pointer_sha256"] != hashes["pointer"] or
            report["adaptive_variant"] != "v31_template_gaze_height_router_candidate" or
            report["toy_goal_source"] != "template_matched_virtual_rgb_exact_pose_known_cube" or
            report["selection_mode"] != mode or
            report["gaze_head_yaw_rad"] != -1.5):
        raise ValueError(f"V31 fresh source or router mismatch: {seed}")
    faults = gate(report)
    entry = next((event for event in report["image_goal_capture_events"]
                  if event["time_s"] == 7.5), None)
    if (entry is None or not entry["used_for_control"] or
            not entry.get("template_applied") or
            entry.get("template_candidate_count") != 202):
        faults.append("template RGB goal missing at entry")
    if entry and (entry.get("xy_error_truth_mm_scoring_only") is None or
                  entry["xy_error_truth_mm_scoring_only"] > 3):
        faults.append("RGB goal error above 3 mm")
    if report["image_goal_missing_at_s"] is not None:
        faults.append("image goal missing during reach")
    if mode == "v16_live":
        if report["image_goal_lock_after_entry"] or report["image_goal_used_update_count"] < 2:
            faults.append("live RGB updates missing")
    elif not report["image_goal_lock_after_entry"] or report["image_goal_used_update_count"] != 1:
        faults.append("entry RGB goal not locked")
    return {"seed": seed, "intended_stratum": case["intended_stratum"],
            "selection_mode": mode, "pass": not faults,
            "contact_s": report["phases"]["reach"]["continuous_finger_contact_s"],
            "rail_v": case["rail_v"], "delay_ms": case["delay_ms"],
            "toy_offset_b_m": [case["toy_offset_x_m"], case["toy_offset_y_m"]],
            "entry_error_body_xy_m": [x, y],
            "entry_goal_error_mm_scoring_only": None if entry is None else entry.get("xy_error_truth_mm_scoring_only"),
            "template_pixels_disagree": None if entry is None else entry.get("template_pixels_disagree"),
            "reach_both_feet_fraction": report["phases"]["reach"]["both_feet_fraction"],
            "reach_rated_leg_fraction": report["phases"]["reach"]["rated_leg_fraction"],
            "faults": faults,
            "result": str(result.relative_to(ROOT)), "trace": str(trace.relative_to(ROOT)),
            "log": str(log.relative_to(ROOT))}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--workers", type=int, default=6)
    args = parser.parse_args()
    if not 1 <= args.workers <= 6:
        raise ValueError("Use one to six native evaluators")
    plan = frozen_plan()
    DEST.mkdir(parents=True, exist_ok=True)
    rows = {}
    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        futures = {pool.submit(run_case, case, plan): case for case in plan["cases"]}
        for future in as_completed(futures):
            case = futures[future]
            row = future.result()
            rows[case["seed"]] = row
            ordered = [rows[case["seed"]] for case in plan["cases"] if case["seed"] in rows]
            selections = Counter(item["selection_mode"] for item in ordered)
            strata = {name: {"cases": sum(item["intended_stratum"] == name for item in ordered),
                             "passes": sum(item["intended_stratum"] == name and item["pass"] for item in ordered)}
                      for name in sorted({case["intended_stratum"] for case in plan["cases"]})}
            matrix = {"schema": "robot-learning-lab.cove50-v31-disjoint-fresh-native.v1",
                      "role": "disjoint_fresh_score_rejected_historical_body",
                      "body_role": "rejected_historical_direct_frame_sandbox",
                      "plan_sha256": sha(PLAN), "opened_matrix_sha256": sha(OPENED_MATRIX),
                      "complete": len(ordered) == 32, "case_count": len(ordered),
                      "pass_count": sum(item["pass"] for item in ordered),
                      "selection_counts": dict(selections), "strata": strata,
                      "promotion_gate_pass": (len(ordered) == 32 and all(item["pass"] for item in ordered) and
                                              selections["v16_v10_height_locked"] >= 4 and
                                              selections["v21_scoped_y_locked"] >= 4),
                      "cases": ordered}
            (DEST / "matrix.json").write_text(json.dumps(matrix, indent=2) + "\n")
            print(json.dumps(row), flush=True)


if __name__ == "__main__":
    main()
