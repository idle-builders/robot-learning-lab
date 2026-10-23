"""Probe saved RGB branches around the two opened v21 fresh misses and controls.

These experimenter-forced branches are diagnostic development evidence only.
They cannot change the sealed v21 score or qualify an automatic v22 router.
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
    run_v18_development_variance as v18,
)


BASE = ROOT / "artifacts/ape-cove50-reach-goal-channels-20261010"
V21 = BASE / "v21"
OUT = BASE / "v22"
PLAN = OUT / "opened-fresh-failure-probe-plan.json"
DEST = OUT / "opened-branch-probes"
SOURCE_SCORE = V21 / "fresh-reserved/matrix.json"
SOURCE_PLAN = V21 / "fresh-challenge-plan.json"
EVALUATOR = v18.TEACHER_EVALUATOR
POSITIVE_SEEDS = (10604, 10601, 10602)
NEGATIVE_SEEDS = (10615, 10613, 10617)
POSITIVE_PROBES = (
    ("positive_y2", "v16_locked", 0.0, 2.0),
    ("positive_x2", "v16_locked", 2.0, 0.0),
    ("live_rgb", "v16_live", 0.0, 0.0),
)
NEGATIVE_PROBES = (
    ("v10_height", "v16_v10_height_locked", 0.0, 0.0),
    ("v10_locked", "v16_v10_locked", 0.0, 0.0),
    ("live_rgb", "v16_live", 0.0, 0.0),
)


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def frozen_plan() -> dict:
    scored = json.loads(SOURCE_SCORE.read_text())
    fresh = json.loads(SOURCE_PLAN.read_text())
    if (not scored["complete"] or scored["case_count"] != 24 or
            scored["pass_count"] != 22 or scored["promotion_gate_pass"] or
            scored["plan_sha256"] != sha(SOURCE_PLAN)):
        raise ValueError("Need sealed failed v21 disjoint score")
    rows = {row["seed"]: row for row in scored["cases"]}
    if (rows[10604]["pass"] or rows[10615]["pass"] or
            any(not rows[seed]["pass"] for seed in
                (*POSITIVE_SEEDS[1:], *NEGATIVE_SEEDS[1:]))):
        raise ValueError("V21 failure/control roles changed")
    original = {case["seed"]: case for case in fresh["cases"]}
    tasks = []
    for seed in POSITIVE_SEEDS:
        for label, mode, bx, by in POSITIVE_PROBES:
            tasks.append({**original[seed], "probe": label, "forced_mode": mode,
                          "extra_goal_bias_body_xy_mm": [bx, by],
                          "baseline_result": rows[seed]["result"],
                          "baseline_pass": rows[seed]["pass"],
                          "baseline_contact_s": rows[seed]["contact_s"]})
    for seed in NEGATIVE_SEEDS:
        for label, mode, bx, by in NEGATIVE_PROBES:
            tasks.append({**original[seed], "probe": label, "forced_mode": mode,
                          "extra_goal_bias_body_xy_mm": [bx, by],
                          "baseline_result": rows[seed]["result"],
                          "baseline_pass": rows[seed]["pass"],
                          "baseline_contact_s": rows[seed]["contact_s"]})
    if len(tasks) != 18:
        raise ValueError("Expected 18 paired opened diagnostics")
    pointer = json.loads(v16.SOURCES["pointer"].read_text())
    if pointer["status"] != "rejected_no_selected_mechanical_design":
        raise ValueError("Historical rejected-body sandbox only")
    return {
        "schema": "robot-learning-lab.cove50-v22-opened-fresh-failure-probe-plan.v1",
        "role": "postscore_opened_forced_branch_diagnostic_not_automatic_or_fresh",
        "body_role": "rejected_historical_direct_frame_sandbox",
        "v21_fresh_plan_sha256": sha(SOURCE_PLAN),
        "v21_fresh_matrix_sha256": sha(SOURCE_SCORE),
        "source_sha256": {name: sha(path) for name, path in v16.SOURCES.items()},
        "teacher_evaluator_sha256": sha(EVALUATOR),
        "runner_sha256": sha(Path(__file__)),
        "paired_pre_reach_tolerance": {"qpos_max_abs": 5e-5,
                                         "action_max_abs": 1e-3,
                                         "rgb_entry_max_abs_m": 5e-4},
        "tasks": tasks,
        "qualification": "Probe three saved actor/goal alternatives for each newly opened failure and two passing same-region controls. Forced modes and toy truth are not automatic inference. Any candidate requires a new observable selector, opened retention gate, and separately frozen disjoint native test.",
    }


def freeze(plan: dict) -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    expected = (json.dumps(plan, indent=2) + "\n").encode()
    if PLAN.exists():
        if PLAN.read_bytes() != expected:
            raise ValueError("Frozen v22 diagnostic plan differs")
    else:
        PLAN.write_bytes(expected)


def run_task(task: dict, plan: dict) -> dict:
    seed, probe = task["seed"], task["probe"]
    stem = f"seed-{seed}-{probe}"
    result, trace, log = (DEST / f"{stem}.{suffix}" for suffix in ("json", "npz", "log"))
    sources = v16.SOURCES
    bx, by = task["extra_goal_bias_body_xy_mm"]
    command = [sys.executable, str(EVALUATOR),
               "--body-xml", str(sources["body"]),
               "--calibration-model", str(sources["calibration"]),
               "--squat-checkpoint", str(sources["squat"]),
               "--reach-checkpoint", str(sources["actor"]),
               "--alternate-reach-checkpoint", str(sources["alternate_actor"]),
               "--specialist-reach-checkpoint", str(sources["specialist_actor"]),
               "--force-selection-mode", task["forced_mode"],
               "--extra-goal-bias-body-x-mm", str(bx),
               "--extra-goal-bias-body-y-mm", str(by),
               "--reach-controller", "learned-residual",
               "--reach-reference", str(sources["reference"]),
               "--geometry", str(sources["geometry"]),
               "--entry", "continuous", "--seed", str(seed),
               "--rail-v", str(task["rail_v"]), "--delay-ms", str(task["delay_ms"]),
               "--toy-offset-x-m", str(task["toy_offset_x_m"]),
               "--toy-offset-y-m", str(task["toy_offset_y_m"]),
               "--pitch-offset-rad", ".105", "--roll-clearance-rad", "-.25",
               "--terminal-roll-offset-rad", "-.08", "--terminal-roll-mode", "entry-error-x",
               "--reach-duration-s", "9", "--gaze-head-yaw-rad", str(task["gaze_head_yaw_rad"]),
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
            raise RuntimeError(f"V22 probe evaluator failed for {stem}: {log}")
    report = json.loads(result.read_text())
    baseline_path = ROOT / task["baseline_result"]
    baseline = json.loads(baseline_path.read_text())
    with np.load(trace) as new_trace, np.load(baseline_path.with_suffix(".npz")) as old_trace:
        if (new_trace["qpos"].shape != old_trace["qpos"].shape or
                new_trace["action"].shape != old_trace["action"].shape):
            raise ValueError(f"Paired trace shape changed: {stem}")
        qpos_delta = float(np.max(np.abs(new_trace["qpos"][375] - old_trace["qpos"][375])))
        action_delta = float(np.max(np.abs(new_trace["action"][:375] - old_trace["action"][:375])))
    x, y = report["selection_features"]["entry_error_xy_m"]
    ox, oy = baseline["selection_features"]["entry_error_xy_m"]
    rgb_delta = max(abs(x-ox), abs(y-oy))
    tolerance = plan["paired_pre_reach_tolerance"]
    if (report["evaluator_sha256"] != plan["teacher_evaluator_sha256"] or
            report["body_sha256"] != plan["source_sha256"]["body"] or
            report["selected_pointer_sha256"] != plan["source_sha256"]["pointer"] or
            report["adaptive_variant"] != "v18_opened_derived_teacher_probe" or
            report["selection_mode"] != task["forced_mode"] or
            report["extra_goal_bias_body_xy_mm"] != [bx, by] or
            report["boundary_state_sha256"]["posture"] != baseline["boundary_state_sha256"]["posture"] or
            report["first_action_sha256"] != baseline["first_action_sha256"] or
            qpos_delta > tolerance["qpos_max_abs"] or
            action_delta > tolerance["action_max_abs"] or
            rgb_delta > tolerance["rgb_entry_max_abs_m"]):
        raise ValueError(f"V22 probe source or paired start mismatch: {stem}")
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
    if task["forced_mode"] == "v16_live":
        if report["image_goal_lock_after_entry"] or report["image_goal_used_update_count"] < 2:
            faults.append("live RGB updates missing")
    elif not report["image_goal_lock_after_entry"] or report["image_goal_used_update_count"] != 1:
        faults.append("entry RGB goal not locked")
    return {
        "seed": seed, "intended_stratum": task["intended_stratum"],
        "probe": probe, "forced_mode": task["forced_mode"],
        "extra_goal_bias_body_xy_mm": [bx, by],
        "baseline_pass": task["baseline_pass"],
        "baseline_contact_s": task["baseline_contact_s"],
        "pass": not faults,
        "contact_s": report["phases"]["reach"]["continuous_finger_contact_s"],
        "entry_error_body_xy_m": [x, y],
        "entry_goal_error_mm_scoring_only": None if entry is None else entry.get("xy_error_truth_mm_scoring_only"),
        "paired_reach_entry_qpos_max_abs": qpos_delta,
        "paired_pre_reach_action_max_abs": action_delta,
        "paired_rgb_entry_delta_m": rgb_delta,
        "faults": faults,
        "result": str(result.relative_to(ROOT)),
        "trace": str(trace.relative_to(ROOT)),
        "log": str(log.relative_to(ROOT)),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--freeze-only", action="store_true")
    parser.add_argument("--workers", type=int, default=6)
    args = parser.parse_args()
    if not 1 <= args.workers <= 6:
        raise ValueError("Use one to six native evaluators")
    plan = frozen_plan()
    freeze(plan)
    if args.freeze_only:
        print(f"frozen {PLAN} sha256={sha(PLAN)}")
        return
    DEST.mkdir(parents=True, exist_ok=True)
    rows = {}
    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        futures = {pool.submit(run_task, task, plan): task for task in plan["tasks"]}
        for future in as_completed(futures):
            task = futures[future]
            row = future.result()
            rows[(task["seed"], task["probe"])] = row
            ordered = [rows[(task["seed"], task["probe"])] for task in plan["tasks"]
                       if (task["seed"], task["probe"]) in rows]
            matrix = {
                "schema": "robot-learning-lab.cove50-v22-opened-forced-branch-native.v1",
                "role": "postscore_opened_diagnostic_not_automatic_or_fresh",
                "plan_sha256": sha(PLAN),
                "complete": len(ordered) == len(plan["tasks"]),
                "case_count": len(ordered),
                "pass_count": sum(item["pass"] for item in ordered),
                "rescues": [[item["seed"], item["probe"]] for item in ordered
                            if item["pass"] and not item["baseline_pass"]],
                "regressions": [[item["seed"], item["probe"]] for item in ordered
                                if not item["pass"] and item["baseline_pass"]],
                "cases": ordered,
            }
            (DEST / "matrix.json").write_text(json.dumps(matrix, indent=2) + "\n")
            print(json.dumps(row), flush=True)


if __name__ == "__main__":
    main()
