"""Screen the saved -2 mm Y RGB-goal correction on opened nearby controls.

V18 already measured this prescribed branch around 10417 and 10422. These
eight additional opened-derived visits expose its contact effect where a
future observed-entry router might select it. No automatic router is scored.
"""

from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
import hashlib
import json
from pathlib import Path

import numpy as np

from experiments.compact_morphology.cove50_policy_transfer.reach_image_goal import (
    run_v18_development_variance as v18,
)


ROOT = Path(__file__).resolve().parents[4]
BASE = ROOT / "artifacts/ape-cove50-reach-goal-channels-20261010"
V18 = BASE / "v18"
OUT = BASE / "v20"
DEST = OUT / "opened-bias-screen"
PLAN = OUT / "opened-bias-screen-plan.json"
SEEDS = (10522, 10523, 10524, 10543, 10544, 10546, 10547, 10548)


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def frozen_plan() -> dict:
    development_path = V18 / "development-variance-plan.json"
    baseline_path = V18 / "development-baseline/matrix.json"
    development = json.loads(development_path.read_text())
    baseline = json.loads(baseline_path.read_text())
    if not baseline["complete"] or baseline["case_count"] != 48:
        raise ValueError("Need complete v18 opened-derived automatic baseline")
    cases = {row["seed"]: row for row in development["cases"]}
    rows = {row["seed"]: row for row in baseline["cases"]}
    selected = []
    for seed in SEEDS:
        case, row = cases[seed], rows[seed]
        source = ROOT / row["result"]
        report = json.loads(source.read_text())
        x, y = report["selection_features"]["entry_error_xy_m"]
        if (row["selection_mode"] != "v16_locked" or
                not (x > .0015 or (x > -.001 and y > .009))):
            raise ValueError(f"Seed {seed} is outside the opened RGB selector screen")
        selected.append({**case, "assigned_teacher_probe": "v14_locked_minus_y2mm",
                         "baseline_result": row["result"],
                         "baseline_report_sha256": sha(source),
                         "baseline_pass": row["pass"],
                         "baseline_contact_s": row["contact_s"],
                         "entry_error_body_xy_m": [x, y]})
    return {
        "schema": "robot-learning-lab.cove50-v20-scoped-bias-screen-plan.v1",
        "role": "opened_native_prescribed_diagnostic_not_router_or_fresh_score",
        "body_role": "rejected_historical_direct_frame_sandbox",
        "v18_development_plan_sha256": sha(development_path),
        "v18_baseline_matrix_sha256": sha(baseline_path),
        "source_sha256": {name: sha(path) for name, path in v18.v16.SOURCES.items()},
        "evaluator_sha256": sha(v18.TEACHER_EVALUATOR),
        "runner_sha256": sha(Path(__file__)),
        "paired_pre_reach_tolerance": {"qpos_max_abs": 5e-5, "action_max_abs": 1e-3},
        "cases": selected,
        "qualification": "Only an experimenter-prescribed -2 mm body-Y correction on eight opened-derived starts. Prior v18 teacher successes do not make this an automatic RGB router. Any router needs an opened retention gate and disjoint fresh native test.",
    }


def freeze(plan: dict) -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    expected = (json.dumps(plan, indent=2) + "\n").encode()
    if PLAN.exists():
        if PLAN.read_bytes() != expected:
            raise ValueError("Frozen v20 screen plan differs from current sources")
    else:
        PLAN.write_bytes(expected)


def run_case(case: dict, plan: dict) -> dict:
    result = v18.teacher_case(case, {
        "baseline_source_sha256": plan["source_sha256"],
        "teacher_evaluator_sha256": plan["evaluator_sha256"],
    })
    seed = case["seed"]
    old_path = ROOT / case["baseline_result"]
    new_path = ROOT / result["result"]
    old = json.loads(old_path.read_text())
    new = json.loads(new_path.read_text())
    with np.load(new_path.with_suffix(".npz")) as new_trace, np.load(old_path.with_suffix(".npz")) as old_trace:
        if (new_trace["qpos"].shape != old_trace["qpos"].shape or
                new_trace["action"].shape != old_trace["action"].shape):
            raise ValueError(f"Paired trace shape differs for {seed}")
        entry_step = 375
        qpos_delta = float(np.max(np.abs(new_trace["qpos"][entry_step] -
                                           old_trace["qpos"][entry_step])))
        action_delta = float(np.max(np.abs(new_trace["action"][:entry_step] -
                                            old_trace["action"][:entry_step])))
    if (sha(old_path) != case["baseline_report_sha256"] or
            new["boundary_state_sha256"]["posture"] != old["boundary_state_sha256"]["posture"] or
            new["first_action_sha256"] != old["first_action_sha256"] or
            qpos_delta > plan["paired_pre_reach_tolerance"]["qpos_max_abs"] or
            action_delta > plan["paired_pre_reach_tolerance"]["action_max_abs"]):
        raise ValueError(f"Paired start or source changed for {seed}")
    return {**result,
            "baseline_pass": case["baseline_pass"],
            "baseline_contact_s": case["baseline_contact_s"],
            "entry_error_body_xy_m": case["entry_error_body_xy_m"],
            "paired_reach_entry_qpos_max_abs": qpos_delta,
            "paired_pre_reach_action_max_abs": action_delta}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--freeze-only", action="store_true")
    parser.add_argument("--workers", type=int, default=2)
    args = parser.parse_args()
    if not 1 <= args.workers <= 6:
        raise ValueError("Use one to six native evaluators")
    plan = frozen_plan()
    freeze(plan)
    if args.freeze_only:
        print(f"frozen {PLAN} sha256={sha(PLAN)}")
        return
    DEST.mkdir(parents=True, exist_ok=True)
    v18.TEACHER = DEST
    rows = {}
    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        futures = {pool.submit(run_case, case, plan): case for case in plan["cases"]}
        for future in as_completed(futures):
            case = futures[future]
            row = future.result()
            rows[case["seed"]] = row
            ordered = [rows[case["seed"]] for case in plan["cases"] if case["seed"] in rows]
            matrix = {
                "schema": "robot-learning-lab.cove50-v20-prescribed-bias-opened-screen.v1",
                "role": "opened_development_not_automatic_or_fresh",
                "plan_sha256": sha(PLAN),
                "complete": len(ordered) == len(plan["cases"]),
                "case_count": len(ordered),
                "pass_count": sum(item["pass"] for item in ordered),
                "paired_rescues": [item["seed"] for item in ordered
                                   if item["pass"] and not item["baseline_pass"]],
                "paired_regressions": [item["seed"] for item in ordered
                                       if not item["pass"] and item["baseline_pass"]],
                "cases": ordered,
            }
            (DEST / "matrix.json").write_text(json.dumps(matrix, indent=2) + "\n")
            print(json.dumps(row), flush=True)


if __name__ == "__main__":
    main()
