"""Pair a candidate contact-target shift with all varied V106 native passes."""

from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
import hashlib
import json
import os
from pathlib import Path

from experiments.compact_morphology.cove50_policy_transfer.reach_image_goal import (
    run_v95_native_case as baseline,
    run_v108_native_case as shifted,
)


ROOT = Path(__file__).resolve().parents[4]
BASE = ROOT / "artifacts/ape-cove50-reach-goal-channels-20261010"
OUT = BASE / "v110"
PLAN = OUT / "varied-shift-retention-plan.json"
DEST = OUT / "native-varied-shift-retention"
V106_PLAN = BASE / "v106/varied-native-development-plan.json"
V106_MATRIX = BASE / "v106/native-varied-development/matrix.json"
SHIFT_M = -.002


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def freeze() -> dict:
    source_plan = json.loads(V106_PLAN.read_text())
    source_matrix = json.loads(V106_MATRIX.read_text())
    if (not source_matrix["complete"] or
            source_matrix["case_count"] != 24 or
            source_matrix["plan_sha256"] != sha(V106_PLAN) or
            source_plan["source_sha256"]["evaluator"] != sha(baseline.EVALUATOR)):
        raise ValueError("V110 requires complete, source-matched V106 baselines")
    rows = {row["id"]: row for row in source_matrix["cases"]}
    if len(rows) != 24:
        raise ValueError("V106 baselines are incomplete")
    sources = {**source_plan["source_sha256"],
               "evaluator": sha(shifted.EVALUATOR),
               "runner": sha(Path(shifted.__file__))}
    cases = []
    baseline_reports = []
    for source_case in source_plan["cases"]:
        row = rows[source_case["id"]]
        report_path = ROOT / row["result"]
        report = json.loads(report_path.read_text())
        if (not row["pass"] or row["contact_s"] < 1.2 or
                report["evaluator_sha256"] != source_plan["source_sha256"]["evaluator"] or
                report["body_sha256"] != sources["body"] or
                report["selected_pointer_sha256"] != sources["pointer"] or
                report["selection_mode"] != row["selection_mode"]):
            raise ValueError(f"V106 full-gate baseline unavailable: {source_case['id']}")
        baseline_reports.append({"id": source_case["id"], "path": row["result"],
                                 "sha256": sha(report_path)})
        case = {key: value for key, value in source_case.items()
                if key not in ("id", "development_role", "source_condition_seed")}
        cases.append({**case, "id": f"retention-{case['seed']}-neg2",
                      "diagnostic_goal_shift_body_y_m": SHIFT_M,
                      "baseline_contact_s": row["contact_s"],
                      "baseline_selection_mode": row["selection_mode"]})
    plan = {
        "schema": "robot-learning-lab.cove50-v110-varied-shift-retention-plan.v1",
        "role": "opened_paired_varied_candidate_target_retention_not_score",
        "body_role": "rejected_historical_direct_frame_sandbox",
        "source_sha256": sources,
        "baseline_plan_sha256": sha(V106_PLAN),
        "baseline_matrix_sha256": sha(V106_MATRIX),
        "baseline_reports": baseline_reports,
        "runner_sha256": sha(Path(__file__)),
        "diagnostic_goal_shift_body_y_m": SHIFT_M,
        "case_count": len(cases), "cases": cases,
        "method": ("On all 24 varied V106 starts that passed full native RGB, "
                   "finger contact, two-foot support, rated effort, action and "
                   "collision checks, replay the exact same physical cases "
                   "with a -2 mm post-selection body-Y target shift. Preserve "
                   "the automatic observed-RGB route and compare paired "
                   "contact. The frozen V106 reports are the zero-shift "
                   "baselines; none of these opened starts is a new score."),
        "decision_gate": ("Do not promote a blanket target correction if any "
                          "of the 24 paired cases loses the full gate or the "
                          "1.2 s contact margin. Report contact reductions "
                          "and route exposure even when the gate passes. A "
                          "candidate that passes still needs unchanged prior "
                          "retention and wholly new disjoint native scoring."),
    }
    OUT.mkdir(parents=True, exist_ok=True)
    encoded = (json.dumps(plan, indent=2) + "\n").encode()
    if PLAN.exists() and PLAN.read_bytes() != encoded:
        raise ValueError("Frozen V110 plan or sources changed")
    PLAN.write_bytes(encoded)
    return plan


def write_matrix(plan: dict, rows: dict[str, dict]) -> dict:
    ordered = [rows[case["id"]] for case in plan["cases"] if case["id"] in rows]
    result = {
        "schema": "robot-learning-lab.cove50-v110-varied-shift-retention.v1",
        "role": plan["role"], "body_role": plan["body_role"],
        "plan_sha256": sha(PLAN), "complete": len(ordered) == len(plan["cases"]),
        "case_count": len(ordered),
        "pass_count": sum(row["pass"] for row in ordered),
        "margin_1p2_s_count": sum(row["margin_1p2_s_met"] for row in ordered),
        "regression_count": sum(not row["pass"] for row in ordered),
        "contact_reduction_count": sum(row["contact_delta_s"] < 0 for row in ordered),
        "cases": ordered,
    }
    temporary = DEST / "matrix.json.tmp"
    temporary.write_text(json.dumps(result, indent=2) + "\n")
    os.replace(temporary, DEST / "matrix.json")
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--freeze-only", action="store_true")
    parser.add_argument("--workers", type=int, default=2)
    args = parser.parse_args()
    if args.workers not in (1, 2):
        raise ValueError("Use one or two native evaluators")
    plan = freeze()
    print(f"frozen {PLAN} sha256={sha(PLAN)}", flush=True)
    if args.freeze_only:
        return
    DEST.mkdir(parents=True, exist_ok=True)
    old = DEST / "matrix.json"
    existing = json.loads(old.read_text()) if old.exists() else None
    if existing and existing["plan_sha256"] != sha(PLAN):
        raise ValueError("Existing V110 matrix belongs to another plan")
    rows = {row["id"]: row for row in existing["cases"]} if existing else {}

    def run_case(case: dict) -> dict:
        row = shifted.run_case(case, plan, DEST)
        if row["selection_mode"] != case["baseline_selection_mode"]:
            raise ValueError("Target shift changed the automatic entry route")
        row.update({"baseline_selection_mode": case["baseline_selection_mode"],
                    "baseline_contact_s": case["baseline_contact_s"],
                    "diagnostic_goal_shift_body_y_m": SHIFT_M,
                    "contact_delta_s": round(row["contact_s"] -
                                             case["baseline_contact_s"], 3),
                    "rail_v": case["rail_v"], "delay_ms": case["delay_ms"],
                    "gaze_head_yaw_rad": case["gaze_head_yaw_rad"],
                    "toy_offset_x_m": case["toy_offset_x_m"],
                    "toy_offset_y_m": case["toy_offset_y_m"]})
        return row

    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        futures = {pool.submit(run_case, case): case for case in plan["cases"]
                   if case["id"] not in rows}
        for future in as_completed(futures):
            row = future.result()
            rows[row["id"]] = row
            write_matrix(plan, rows)
            print(json.dumps({key: row[key] for key in
                              ("id", "baseline_contact_s", "contact_s",
                               "contact_delta_s", "pass", "faults")}), flush=True)
    matrix = write_matrix(plan, rows)
    print(json.dumps({key: matrix[key] for key in
                      ("case_count", "pass_count", "margin_1p2_s_count",
                       "contact_reduction_count")}), flush=True)


if __name__ == "__main__":
    main()
