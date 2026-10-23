"""Native gate for the varied-data v19 arm specialist on opened starts."""

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
from experiments.compact_morphology.cove50_policy_transfer.reach_image_goal import run_v16_opened_matrix as v16


BASE = ROOT / "artifacts/ape-cove50-reach-goal-channels-20261010"
V18 = BASE / "v18"
OUT = BASE / "v19"
EVALUATOR = Path(__file__).with_name("evaluate_scoped_actor_v19.py")
ACTOR = OUT / "actor-scoped.pt"
PLAN = OUT / "opened-probe-plan.json"
DEST = OUT / "opened-native"
TEACHER_SEEDS = (10509, 10510, 10511, 10512)
GENERALIZATION_SEEDS = (10517, 10518, 10519, 10520)
RETENTION_SEEDS = (10417, 10419, 10420, 10422, 10423)
ADVANCE_SEEDS = (*TEACHER_SEEDS, 10417, 10419, 10420)


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def frozen_plan() -> dict:
    training_path = OUT / "actor-training-report.json"
    training = json.loads(training_path.read_text())
    if (training["role"] != "offline_opened_data_training_only" or
            training["actor_sha256"] != sha(ACTOR)):
        raise ValueError("No sealed v19 actor fit")
    dev_path = V18 / "development-variance-plan.json"
    dev = json.loads(dev_path.read_text())
    base18 = json.loads((V18 / "development-baseline/matrix.json").read_text())
    base16_path = BASE / "v16/fresh-reserved-plan.json"
    base16 = json.loads(base16_path.read_text())
    matrix16 = json.loads((BASE / "v16/fresh-reserved/matrix.json").read_text())
    if (not base18["complete"] or base18["case_count"] != 48 or
            not matrix16["complete"] or matrix16["case_count"] != 36):
        raise ValueError("Need complete opened source baselines")
    cases18 = {case["seed"]: case for case in dev["cases"]}
    cases16 = {case["seed"]: case for case in base16["cases"]}
    rows18 = {row["seed"]: row for row in base18["cases"]}
    rows16 = {row["seed"]: row for row in matrix16["cases"]}
    cases = []
    for seed in (*TEACHER_SEEDS, *GENERALIZATION_SEEDS, *RETENTION_SEEDS):
        case = cases18[seed] if seed in cases18 else cases16[seed]
        baseline = rows18[seed] if seed in rows18 else rows16[seed]
        report_path = ROOT / baseline["result"]
        cohort = ("varied_teacher_neighborhood" if seed in TEACHER_SEEDS else
                  "other_failure_neighborhood_generalization" if seed in GENERALIZATION_SEEDS else
                  "original_opened_control")
        cases.append({"seed": seed, "cohort": cohort, "rail_v": case["rail_v"],
                      "delay_ms": case["delay_ms"],
                      "toy_offset_x_m": case["toy_offset_x_m"],
                      "toy_offset_y_m": case["toy_offset_y_m"],
                      "gaze_head_yaw_rad": case["gaze_head_yaw_rad"],
                      "baseline_report": baseline["result"],
                      "baseline_report_sha256": sha(report_path),
                      "baseline_pass": baseline["pass"],
                      "baseline_contact_s": baseline["contact_s"]})
    return {
        "schema": "robot-learning-lab.cove50-v19-scoped-opened-probe-plan.v1",
        "role": "opened_native_development_gate_only",
        "body_role": "rejected_historical_direct_frame_sandbox",
        "training_report_sha256": sha(training_path),
        "actor_sha256": sha(ACTOR),
        "evaluator_sha256": sha(EVALUATOR),
        "v18_development_plan_sha256": sha(dev_path),
        "v16_fresh_plan_sha256": sha(base16_path),
        "source_sha256": {name: sha(path) for name, path in v16.SOURCES.items()},
        "runner_sha256": sha(Path(__file__)),
        "paired_pre_reach_tolerance": {"qpos_max_abs": 5e-5, "action_max_abs": 1e-3},
        "cases": cases,
        "advance_seeds": list(ADVANCE_SEEDS),
        "advance_gate": "The unshifted v19 actor must pass all four varied 10417 neighbors and the original miss, with no regression on the two original passing controls, before routing or disjoint fresh evaluation. Four 10422 neighbors plus its original miss and adjacent pass probe generalization without entering this gate.",
    }


def freeze(plan: dict) -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    expected = (json.dumps(plan, indent=2) + "\n").encode()
    if PLAN.exists():
        if PLAN.read_bytes() != expected:
            raise ValueError("Frozen v19 native plan differs from sources")
    else:
        PLAN.write_bytes(expected)


def run_case(case: dict, plan: dict) -> dict:
    seed = case["seed"]
    result, trace, log = (DEST / f"seed-{seed}.{suffix}" for suffix in ("json", "npz", "log"))
    sources = v16.SOURCES
    command = [sys.executable, str(EVALUATOR),
               "--body-xml", str(sources["body"]),
               "--calibration-model", str(sources["calibration"]),
               "--squat-checkpoint", str(sources["squat"]),
               "--reach-checkpoint", str(ACTOR),
               "--alternate-reach-checkpoint", str(sources["alternate_actor"]),
               "--specialist-reach-checkpoint", str(sources["specialist_actor"]),
               "--force-selection-mode", "v16_locked",
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
            raise RuntimeError(f"V19 actor evaluator failed for seed {seed}: {log}")
    report = json.loads(result.read_text())
    old_path = ROOT / case["baseline_report"]
    old = json.loads(old_path.read_text())
    with np.load(trace) as new_trace, np.load(old_path.with_suffix(".npz")) as old_trace:
        if new_trace["qpos"].shape != old_trace["qpos"].shape or  \
                new_trace["action"].shape != old_trace["action"].shape:
            raise ValueError(f"V19 paired trace shape differs: {result}")
        entry_step = 375
        qpos_delta = float(np.max(np.abs(new_trace["qpos"][entry_step] -
                                           old_trace["qpos"][entry_step])))
        action_delta = float(np.max(np.abs(new_trace["action"][:entry_step] -
                                            old_trace["action"][:entry_step])))
    if (sha(old_path) != case["baseline_report_sha256"] or
            report["evaluator_sha256"] != plan["evaluator_sha256"] or
            report["adaptive_variant"] != "v19_scoped_actor_development_probe" or
            report["selected_pointer_sha256"] != plan["source_sha256"]["pointer"] or
            report["body_sha256"] != plan["source_sha256"]["body"] or
            report["reach_checkpoint_sha256"] != plan["actor_sha256"] or
            report["selected_reach_checkpoint_sha256"] != plan["actor_sha256"] or
            report["selection_mode"] != "v16_locked" or
            report["goal_bias_body_xy_m"] != [0.0, 0.0] or
            report["boundary_state_sha256"]["posture"] != old["boundary_state_sha256"]["posture"] or
            report["first_action_sha256"] != old["first_action_sha256"] or
            qpos_delta > plan["paired_pre_reach_tolerance"]["qpos_max_abs"] or
            action_delta > plan["paired_pre_reach_tolerance"]["action_max_abs"]):
        raise ValueError(f"V19 actor or paired-start identity mismatch: {result}")
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
    if not report["image_goal_lock_after_entry"] or report["image_goal_used_update_count"] != 1:
        faults.append("entry RGB goal not locked")
    return {
        "seed": seed, "cohort": case["cohort"],
        "pass": not faults,
        "contact_s": report["phases"]["reach"]["continuous_finger_contact_s"],
        "baseline_pass": case["baseline_pass"],
        "baseline_contact_s": case["baseline_contact_s"],
        "rail_v": case["rail_v"], "delay_ms": case["delay_ms"],
        "toy_offset_b_m": [case["toy_offset_x_m"], case["toy_offset_y_m"]],
        "entry_goal_error_mm_scoring_only": None if entry is None else entry.get("xy_error_truth_mm_scoring_only"),
        "paired_reach_entry_qpos_max_abs": qpos_delta,
        "paired_pre_reach_action_max_abs": action_delta,
        "reach_both_feet_fraction": report["phases"]["reach"]["both_feet_fraction"],
        "reach_rated_leg_fraction": report["phases"]["reach"]["rated_leg_fraction"],
        "faults": faults,
        "result": str(result.relative_to(ROOT)), "trace": str(trace.relative_to(ROOT)),
        "log": str(log.relative_to(ROOT)),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--freeze-only", action="store_true")
    parser.add_argument("--workers", type=int, default=4)
    args = parser.parse_args()
    if not 1 <= args.workers <= 6:
        raise ValueError("Use one to six independent native evaluators")
    plan = frozen_plan()
    freeze(plan)
    if args.freeze_only:
        print(f"frozen {PLAN} sha256={sha(PLAN)}")
        return
    DEST.mkdir(parents=True, exist_ok=True)
    rows = {}
    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        futures = {pool.submit(run_case, case, plan): case for case in plan["cases"]}
        for future in as_completed(futures):
            case = futures[future]
            row = future.result()
            rows[case["seed"]] = row
            ordered = [rows[case["seed"]] for case in plan["cases"] if case["seed"] in rows]
            matrix = {
                "schema": "robot-learning-lab.cove50-v19-scoped-arm-opened-native.v1",
                "role": "opened_development_not_disjoint_fresh",
                "plan_sha256": sha(PLAN),
                "complete": len(ordered) == len(plan["cases"]),
                "case_count": len(ordered),
                "pass_count": sum(item["pass"] for item in ordered),
                "paired_rescues": [item["seed"] for item in ordered
                                   if item["pass"] and not item["baseline_pass"]],
                "paired_regressions": [item["seed"] for item in ordered
                                       if not item["pass"] and item["baseline_pass"]],
                "advance_gate_pass": (len(ordered) == len(plan["cases"]) and
                                      all(rows[seed]["pass"] for seed in plan["advance_seeds"])),
                "cases": ordered,
            }
            (DEST / "matrix.json").write_text(json.dumps(matrix, indent=2) + "\n")
            print(json.dumps(row), flush=True)


if __name__ == "__main__":
    main()
