"""Gate rendered-template RGB, legal gaze, and corner routing on opened starts."""

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
    run_v21_opened_gate as v21,
)

BASE = ROOT / "artifacts/ape-cove50-reach-goal-channels-20261010"
OUT = BASE / "v31"
PLAN = OUT / "opened-gate-plan.json"
DEST = OUT / "opened-native"
EVALUATOR = Path(__file__).with_name("evaluate_template_goal_v31.py")
TEMPLATE = Path(__file__).with_name("template_virtual_rgb_goal_v30.py")
SOURCE_MATRIX = BASE / "v21/fresh-reserved/matrix.json"
OFFLINE = OUT / "offline-report.json"


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def run_case(case: dict, plan: dict, baseline: dict) -> dict:
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
               "--reach-duration-s", "9", "--gaze-head-yaw-rad", str(case["gaze_head_yaw_rad"]),
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
            raise RuntimeError(f"V31 evaluator failed for seed {seed}: {log}")
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
            report["gaze_head_yaw_rad"] != -1.5 or
            report["selection_mode"] != mode):
        raise ValueError(f"V31 source, gaze, or mode mismatch: {seed}")
    faults = gate(report)
    entry = next((event for event in report["image_goal_capture_events"]
                  if event["time_s"] == 7.5), None)
    if (entry is None or not entry["used_for_control"] or
            not entry.get("template_applied") or
            entry.get("template_candidate_count") != 202):
        faults.append("template RGB goal missing at reach entry")
    if entry and (entry.get("xy_error_truth_mm_scoring_only") is None or
                  entry["xy_error_truth_mm_scoring_only"] > 3):
        faults.append("template RGB error above 3 mm")
    if report["image_goal_missing_at_s"] is not None:
        faults.append("image goal missing during reach")
    with np.load(trace) as new_trace, np.load(ROOT / baseline["trace"]) as old_trace:
        paired_qpos = float(np.max(np.abs(new_trace["qpos"][:375] - old_trace["qpos"][:375])))
        paired_action = float(np.max(np.abs(new_trace["action"][:375] - old_trace["action"][:375])))
    if case["baseline_gaze_head_yaw_rad"] == -1.5 and (paired_qpos > 5e-5 or paired_action > .001):
        faults.append("pre-reach native action/state drift")
    return {"seed": seed, "intended_stratum": case["intended_stratum"],
            "selection_mode": mode, "baseline_selection_mode": baseline["selection_mode"],
            "baseline_gaze_head_yaw_rad": case["baseline_gaze_head_yaw_rad"],
            "new_gaze_head_yaw_rad": case["gaze_head_yaw_rad"],
            "baseline_pass": baseline["pass"], "baseline_contact_s": baseline["contact_s"],
            "pass": not faults,
            "contact_s": report["phases"]["reach"]["continuous_finger_contact_s"],
            "entry_goal_error_mm_scoring_only": (None if entry is None else
                                                 entry.get("xy_error_truth_mm_scoring_only")),
            "template_pixels_disagree": None if entry is None else entry.get("template_pixels_disagree"),
            "template_correction_xy_mm": None if entry is None else entry.get("template_correction_xy_mm"),
            "paired_pre_reach_qpos_max_abs": paired_qpos,
            "paired_pre_reach_action_max_abs": paired_action,
            "faults": faults,
            "result": str(result.relative_to(ROOT)), "trace": str(trace.relative_to(ROOT)),
            "log": str(log.relative_to(ROOT))}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--workers", type=int, default=6)
    args = parser.parse_args()
    if not 1 <= args.workers <= 6:
        raise ValueError("Use one to six evaluators")
    plan = json.loads(PLAN.read_text())
    if (plan["runner_sha256"] != sha(Path(__file__)) or
            plan["source_matrix_sha256"] != sha(SOURCE_MATRIX) or
            plan["source_plan_sha256"] != sha(BASE / "v21/fresh-challenge-plan.json") or
            plan["offline_report_sha256"] != sha(OFFLINE) or
            plan["v29_matrix_sha256"] != sha(BASE / "v29/opened-gaze/matrix.json") or
            plan["v30_matrix_sha256"] != sha(BASE / "v30/opened-native/matrix.json") or
            len(plan["cases"]) != 24 or
            {case["seed"] for case in plan["cases"]} != set(range(10601, 10625)) or
            any(case["gaze_head_yaw_rad"] != -1.5 for case in plan["cases"]) or
            not json.loads(OFFLINE.read_text())["offline_gate_pass"]):
        raise ValueError("Frozen V31 opened plan or offline gate changed")
    sources = {key: sha(path) for key, path in v16.SOURCES.items() if key != "evaluator"}
    sources.update({"evaluator": sha(EVALUATOR), "template": sha(TEMPLATE)})
    if sources != plan["source_sha256"]:
        raise ValueError("Native source changed")
    baseline = {row["seed"]: row for row in json.loads(SOURCE_MATRIX.read_text())["cases"]}
    DEST.mkdir(parents=True, exist_ok=True)
    rows = {}
    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        futures = {pool.submit(run_case, case, plan, baseline[case["seed"]]): case
                   for case in plan["cases"]}
        for future in as_completed(futures):
            case = futures[future]
            row = future.result()
            rows[case["seed"]] = row
            ordered = [rows[case["seed"]] for case in plan["cases"] if case["seed"] in rows]
            matrix = {"schema": "robot-learning-lab.cove50-v31-opened-template-gaze-native.v1",
                      "role": "opened_automatic_template_gaze_height_router_gate_not_fresh",
                      "body_role": "rejected_historical_direct_frame_sandbox",
                      "plan_sha256": sha(PLAN), "complete": len(ordered) == 24,
                      "case_count": len(ordered), "pass_count": sum(item["pass"] for item in ordered),
                      "rescues": [item["seed"] for item in ordered if item["pass"] and not item["baseline_pass"]],
                      "regressions": [item["seed"] for item in ordered if item["baseline_pass"] and not item["pass"]],
                      "advance_gate_pass": (len(ordered) == 24 and all(item["pass"] for item in ordered) and
                                            any(item["seed"] == 10615 and
                                                item["selection_mode"] == "v16_v10_height_locked"
                                                for item in ordered)),
                      "cases": ordered}
            (DEST / "matrix.json").write_text(json.dumps(matrix, indent=2) + "\n")
            print(json.dumps(row), flush=True)


if __name__ == "__main__":
    main()
