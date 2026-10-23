"""Opened V150 live-route failure: compare two prescribed arm routes on CUDA.

This posthoc diagnostic cannot count as a fresh score or a learned router.
"""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path

from experiments.compact_morphology.cove50_policy_transfer.reach_image_goal import (
    run_v95_native_case as native,
    run_v139_reserved_contact as score_runner,
)
from experiments.compact_morphology.cove50_policy_transfer.reach_image_goal.summarize_v144_supported_variance import (
    verify_report,
)
from experiments.compact_morphology.cove50_policy_transfer.reach_image_goal.summarize_v147_new_seed_pairs import (
    verify_trace,
)


ROOT = Path(__file__).resolve().parents[4]
BASE = ROOT / "artifacts/ape-cove50-reach-goal-channels-20261010"
V150_PLAN = BASE / "v150/paired-contact-training-plan.json"
V150_MATRIX = BASE / "v150/native-paired-contact/matrix.json"
RESERVE = BASE / "v147/untouched-score-reserve-plan.json"
POINTER = ROOT / "robot/ape-current-design.json"
OUT = BASE / "v151"
PLAN = OUT / "opened-live-route-ablation-plan.json"
DEST = OUT / "native-opened-live-route-ablation"
MATRIX = DEST / "matrix.json"
EVALUATORS = {
    "forced-specialist": ("evaluate_forced_specialist_v102.py",
                          "v16_specialist_locked", "diagnostic_force_specialist"),
    "forced-scoped": ("evaluate_forced_scoped_v107.py",
                      "v21_scoped_y_locked", "diagnostic_force_scoped"),
}


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def write_json(path: Path, value: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2) + "\n")
    os.replace(temporary, path)


def freeze() -> dict:
    prior = json.loads(V150_PLAN.read_text())
    current = json.loads(V150_MATRIX.read_text())
    reserve = json.loads(RESERVE.read_text())
    case = next(row for row in prior["cases"] if row["seed"] == 24591)
    row = next(row for row in current["cases"] if row["seed"] == 24591)
    source = prior["source_sha256"]
    if (prior["case_count"] != 48 or current["plan_sha256"] != sha(V150_PLAN) or
            source["pointer"] != sha(POINTER) or
            json.loads(POINTER.read_text())["selected_design_id"] is not None or
            source["untouched_score_reserve"] != sha(RESERVE) or
            reserve["status"] != "sealed_no_contact_visits" or
            row["selection_mode"] != "v16_live" or
            row["entry_observed_features"] != case["entry_observed_features"] or
            row["screen_to_full_entry_faults"] or
            any(choice["entry_parity_faults"] for choice in row["choices"].values()) or
            any(value != 0 for value in (row["baseline_contact_s"],
                row["choices"]["neg2"]["contact_s"],
                row["choices"]["pos2"]["contact_s"]))):
        raise ValueError("Opened live-route failure or sealed source changed")
    saved = {}
    for label, report, trace in (
            ("unchanged", row["baseline_result"], row["baseline_trace"]),
            *((label, choice["result"], choice["trace"])
              for label, choice in row["choices"].items())):
        saved[label] = {"report": report, "report_sha256": sha(ROOT / report),
                        "trace": trace, "trace_sha256": sha(ROOT / trace)}
    cases = []
    for label, (filename, route, flag) in EVALUATORS.items():
        cases.append({**case, "id": f"opened-live-24591-{label}",
                      "diagnostic_route": label, "expected_selection_mode": route,
                      "diagnostic_flag": flag,
                      "evaluator_sha256": sha(Path(__file__).with_name(filename))})
    plan = {
        "schema": "robot-learning-lab.cove50-v151-opened-live-route-ablation-plan.v1",
        "role": "posthoc_opened_route_diagnostic_not_independent_score",
        "body_role": "rejected_historical_direct_frame_sandbox",
        "timing_disclosure": ("Frozen after V150 seed 24591 had 0.00 s contact "
                              "unchanged, -2 mm and +2 mm, and after 17 of 48 "
                              "V150 triplets had been inspected. The remaining "
                              "V150 collection and the separate score reserve "
                              "were not used to choose these two forced routes."),
        "question": ("Does changing only the arm-route choice recover any "
                     "native contact on opened seed 24591? A forced route "
                     "diagnosis is not an observable automatic selector."),
        "source_sha256": {**source,
                          "v150_plan": sha(V150_PLAN),
                          "v151_runner": sha(Path(__file__)),
                          "forced_specialist_evaluator":
                          cases[0]["evaluator_sha256"],
                          "forced_scoped_evaluator":
                          cases[1]["evaluator_sha256"]},
        "opened_baseline_sources": saved,
        "baseline_observed_features": row["full_entry_observed_features"],
        "case_count": len(cases), "cases": cases,
    }
    encoded = json.dumps(plan, indent=2) + "\n"
    OUT.mkdir(parents=True, exist_ok=True)
    if PLAN.exists() and PLAN.read_text() != encoded:
        raise ValueError("Frozen V151 plan changed")
    PLAN.write_text(encoded)
    return plan


def run_case(case: dict, plan: dict) -> dict:
    filename, route, flag = EVALUATORS[case["diagnostic_route"]]
    if (case["expected_selection_mode"] != route or
            case["diagnostic_flag"] != flag):
        raise ValueError("Frozen forced route changed")
    original = native.EVALUATOR
    native.EVALUATOR = Path(__file__).with_name(filename)
    try:
        result = native.run_case(case, {"source_sha256": {
            **plan["source_sha256"], "evaluator": case["evaluator_sha256"]}}, DEST)
    finally:
        native.EVALUATOR = original
    report = verify_report(result["result"], result["trace"], case, 0,
                           result["contact_s"], case["evaluator_sha256"])
    verify_trace(result["trace"], report)
    observed = score_runner.observations.observed_features(report)
    baseline = plan["baseline_observed_features"]
    numeric_faults = score_runner.entry_parity(
        baseline, {**observed, "selection_mode": baseline["selection_mode"],
                   "route_goal_source": baseline["route_goal_source"]})
    if (not report.get(flag) or report["selection_mode"] != route or
            report["body_sha256"] != plan["source_sha256"]["body"] or
            report["selected_pointer_sha256"] != plan["source_sha256"]["pointer"] or
            report["reach_checkpoint_sha256"] != plan["source_sha256"]["actor"]):
        raise ValueError(f"Forced route source differed: {case['id']}")
    return {**result, "diagnostic_route": case["diagnostic_route"],
            "forced_route": route, "observed_features": observed,
            "numeric_rgb_entry_faults": numeric_faults,
            "full_gate_pass_with_rgb_parity": result["pass"] and not numeric_faults,
            "report_sha256": sha(ROOT / result["result"]),
            "trace_sha256": sha(ROOT / result["trace"])}


def main() -> None:
    plan = freeze()
    DEST.mkdir(parents=True, exist_ok=True)
    old = json.loads(MATRIX.read_text()) if MATRIX.exists() else None
    if old and old["plan_sha256"] != sha(PLAN):
        raise ValueError("Existing V151 matrix has another plan")
    rows = {row["id"]: row for row in old["cases"]} if old else {}
    print(f"V151 opened route plan sha256={sha(PLAN)}", flush=True)
    for case in plan["cases"]:
        if case["id"] in rows:
            continue
        result = run_case(case, plan)
        rows[case["id"]] = result
        ordered = [rows[item["id"]] for item in plan["cases"] if item["id"] in rows]
        write_json(MATRIX, {
            "schema": "robot-learning-lab.cove50-v151-opened-live-route-ablation.v1",
            "role": plan["role"], "body_role": plan["body_role"],
            "plan_sha256": sha(PLAN), "case_count": len(ordered),
            "planned_count": plan["case_count"],
            "complete": len(ordered) == plan["case_count"],
            "cases": ordered,
        })
        print(f"V151 {case['diagnostic_route']}: {result['contact_s']:.2f} s, "
              f"full {'pass' if result['full_gate_pass_with_rgb_parity'] else 'fail'}; "
              f"{len(ordered)}/{plan['case_count']}", flush=True)


if __name__ == "__main__":
    main()
