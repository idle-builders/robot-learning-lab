"""Test small post-actor shoulder-roll corrections on opened RGB cases."""

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
    run_v22_opened_failure_probes as helper,
)


BASE = ROOT / "artifacts/ape-cove50-reach-goal-channels-20261010"
V21 = BASE / "v21"
OUT = BASE / "v27"
PLAN = OUT / "opened-post-actor-roll-plan.json"
DEST = OUT / "opened-post-actor-roll"
SEEDS = (10604, 10601, 10602)
ROUTES = (("post_minus010", -.010), ("post_minus020", -.020))
SOURCES = helper.v16.SOURCES
EVALUATOR = Path(__file__).with_name("evaluate_post_actor_roll_v27.py")


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def entry(report: dict) -> dict:
    return next(event for event in report["image_goal_capture_events"]
                if event["time_s"] == 7.5)


def frozen_plan() -> dict:
    score_path = V21 / "fresh-reserved/matrix.json"
    challenge_path = V21 / "fresh-challenge-plan.json"
    prior_path = BASE / "v26/opened-terminal-roll/matrix.json"
    score = json.loads(score_path.read_text())
    challenge = json.loads(challenge_path.read_text())
    prior = json.loads(prior_path.read_text())
    if (not score["complete"] or score["case_count"] != 24 or
            score["pass_count"] != 22 or score["promotion_gate_pass"] or
            score["plan_sha256"] != sha(challenge_path)):
        raise ValueError("Need sealed failed V21 score")
    rows = {row["seed"]: row for row in score["cases"]}
    if ([rows[seed]["pass"] for seed in SEEDS] != [False, True, True] or
            any(rows[seed]["selection_mode"] != "v21_scoped_y_locked"
                for seed in SEEDS)):
        raise ValueError("Need opened scoped-Y miss and controls")
    if (not prior["complete"] or prior["case_count"] != 6 or
            prior["rescues"] or prior["regressions"]):
        raise ValueError("Need failed V26 route-offset screen and retained controls")
    cases = {case["seed"]: case for case in challenge["cases"]}
    pointer = json.loads(SOURCES["pointer"].read_text())
    if pointer["status"] != "rejected_no_selected_mechanical_design":
        raise ValueError("Historical rejected-body diagnostic only")
    tasks = [{**cases[seed], "probe": label,
              "post_actor_terminal_roll_offset_rad": roll,
              "baseline_result": rows[seed]["result"],
              "baseline_pass": rows[seed]["pass"],
              "baseline_contact_s": rows[seed]["contact_s"]}
             for seed in SEEDS for label, roll in ROUTES]
    return {
        "schema": "robot-learning-lab.cove50-v27-opened-post-actor-roll-plan.v1",
        "role": "postscore_opened_post_actor_arm_diagnostic_not_automatic_or_fresh",
        "body_role": "rejected_historical_direct_frame_sandbox",
        "v21_fresh_matrix_sha256": sha(score_path),
        "v21_fresh_plan_sha256": sha(challenge_path),
        "v26_opened_matrix_sha256": sha(prior_path),
        "source_sha256": {name: sha(path) for name, path in SOURCES.items()},
        "teacher_evaluator_sha256": sha(EVALUATOR),
        "runner_sha256": sha(Path(__file__)),
        "paired_pre_reach_tolerance": {"qpos_max_abs": 5e-5,
                                        "action_max_abs": 1e-3,
                                        "rgb_entry_max_abs_m": 5e-4},
        "diagnostic_basis": "V26 fixed terminal route offsets of -0.010 and -0.020 rad left 10604 at zero contact. In the native report, the saved actor increased its shoulder-roll residual by about +0.020 and +0.041 raw units, almost canceling those authored-route changes. Apply the same bounded physical target offsets after the saved actor output; this opened-state diagnosis is not an inference input.",
        "tasks": tasks,
        "qualification": "Saved v13 RGB actor and -2 mm observed-goal Y shift with a bounded post-actor physical shoulder-roll target correction. These are experimenter-prescribed opened replays. Any automatic candidate needs varied pass retention and a separately frozen disjoint native score; no new body is qualified.",
    }


def run_case(case: dict, plan: dict) -> dict:
    seed, probe = case["seed"], case["probe"]
    stem = f"seed-{seed}-{probe}"
    result, trace, log = (DEST / f"{stem}.{suffix}"
                          for suffix in ("json", "npz", "log"))
    command = [sys.executable, str(EVALUATOR),
               "--body-xml", str(SOURCES["body"]),
               "--calibration-model", str(SOURCES["calibration"]),
               "--squat-checkpoint", str(SOURCES["squat"]),
               "--reach-checkpoint", str(SOURCES["actor"]),
               "--alternate-reach-checkpoint", str(SOURCES["alternate_actor"]),
               "--specialist-reach-checkpoint", str(SOURCES["specialist_actor"]),
               "--force-selection-mode", "v16_locked",
               "--extra-goal-bias-body-y-mm", "-2",
               "--reach-controller", "learned-residual",
               "--reach-reference", str(SOURCES["reference"]),
               "--geometry", str(SOURCES["geometry"]),
               "--entry", "continuous", "--seed", str(seed),
               "--rail-v", str(case["rail_v"]),
               "--delay-ms", str(case["delay_ms"]),
               "--toy-offset-x-m", str(case["toy_offset_x_m"]),
               "--toy-offset-y-m", str(case["toy_offset_y_m"]),
               "--pitch-offset-rad", ".105", "--roll-clearance-rad", "-.25",
               "--terminal-roll-offset-rad", "-.08",
               "--terminal-roll-mode", "entry-error-x",
               "--post-actor-terminal-roll-offset-rad", str(case["post_actor_terminal_roll_offset_rad"]),
               "--reach-duration-s", "9",
               "--gaze-head-yaw-rad", str(case["gaze_head_yaw_rad"]),
               "--gaze-start-s", "0", "--gaze-ramp-s", "7",
               "--capture-stride-steps", "5", "--max-goal-age-s", "9",
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
            raise RuntimeError(f"Terminal-roll evaluator failed for {stem}: {log}")
    report = json.loads(result.read_text())
    baseline_path = ROOT / case["baseline_result"]
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
            report["adaptive_variant"] != "v27_opened_post_actor_roll_probe" or
            report["selection_mode"] != "v16_locked" or
            report["extra_goal_bias_body_xy_mm"] != [0.0, -2.0] or
            report["terminal_roll_mode"] != "entry-error-x" or
            report["terminal_roll_offset_rad"] != -.08 or
            report["post_actor_terminal_roll_offset_rad"] != case["post_actor_terminal_roll_offset_rad"] or
            report["boundary_state_sha256"]["posture"] != baseline["boundary_state_sha256"]["posture"] or
            report["first_action_sha256"] != baseline["first_action_sha256"] or
            qpos_delta > tolerance["qpos_max_abs"] or
            action_delta > tolerance["action_max_abs"] or
            rgb_delta > tolerance["rgb_entry_max_abs_m"]):
        raise ValueError(f"Terminal-roll source or paired start mismatch: {stem}")
    faults = gate(report)
    capture = entry(report)
    if (not capture["used_for_control"] or capture["orange_pixels"] < 10 or
            capture.get("xy_error_truth_mm_scoring_only") is None or
            capture["xy_error_truth_mm_scoring_only"] > 10 or
            report["image_goal_missing_at_s"] is not None or
            not report["image_goal_lock_after_entry"] or
            report["image_goal_used_update_count"] != 1):
        faults.append("invalid locked RGB goal")
    return {
        "seed": seed, "probe": probe,
        "post_actor_terminal_roll_offset_rad": case["post_actor_terminal_roll_offset_rad"],
        "baseline_pass": case["baseline_pass"],
        "baseline_contact_s": case["baseline_contact_s"],
        "pass": not faults,
        "contact_s": report["phases"]["reach"]["continuous_finger_contact_s"],
        "entry_error_body_xy_m": [x, y],
        "entry_goal_error_mm_scoring_only": capture["xy_error_truth_mm_scoring_only"],
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
    OUT.mkdir(parents=True, exist_ok=True)
    expected = (json.dumps(plan, indent=2) + "\n").encode()
    if PLAN.exists():
        if PLAN.read_bytes() != expected:
            raise ValueError("Frozen V27 post-actor roll plan differs")
    else:
        PLAN.write_bytes(expected)
    if args.freeze_only:
        print(f"frozen {PLAN} sha256={sha(PLAN)}")
        return
    DEST.mkdir(parents=True, exist_ok=True)
    rows = {}
    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        futures = {pool.submit(run_case, case, plan): case for case in plan["tasks"]}
        for future in as_completed(futures):
            row = future.result()
            rows[(row["seed"], row["probe"])] = row
            ordered = [rows[(case["seed"], case["probe"])]
                       for case in plan["tasks"]
                       if (case["seed"], case["probe"]) in rows]
            matrix = {
                "schema": "robot-learning-lab.cove50-v27-opened-post-actor-roll-native.v1",
                "role": plan["role"],
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
