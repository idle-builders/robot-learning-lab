"""Diagnose failed V139 abstentions with a source-matched −2 mm visit."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path

from experiments.compact_morphology.cove50_policy_transfer.reach_image_goal import (
    build_v118_observable_pairs as source_check,
    run_v108_native_case as shifted,
    run_v139_reserved_contact as score_runner,
)


ROOT = Path(__file__).resolve().parents[4]
BASE = ROOT / "artifacts/ape-cove50-reach-goal-channels-20261010"
SCORE_PLAN = BASE / "v139/reserved-contact-score-plan.json"
SCORE_MATRIX = BASE / "v139/native-reserved-contact/matrix.json"
OUT = BASE / "v141"
PLAN = OUT / "opened-score-failure-plan.json"
DEST = OUT / "native-opened-failure-shifts"


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def write_json(path: Path, data: dict, *, frozen: bool = False) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    encoded = json.dumps(data, indent=2) + "\n"
    if frozen and path.exists() and path.read_text() != encoded:
        raise ValueError(f"Frozen diagnostic plan changed: {path}")
    temp = path.with_suffix(path.suffix + ".tmp")
    temp.write_text(encoded)
    os.replace(temp, path)


def freeze() -> dict:
    score_plan = json.loads(SCORE_PLAN.read_text())
    score = json.loads(SCORE_MATRIX.read_text())
    if (not score["complete"] or score["case_count"] != 24 or
            score["plan_sha256"] != sha(SCORE_PLAN) or
            score["score_gate_met"]):
        raise ValueError("Complete failed V139 score required for opened diagnosis")
    source_rows = {row["id"]: row for row in score["cases"]}
    cases = []
    for original in score_plan["cases"]:
        row = source_rows[original["id"]]
        if ((row["candidate_pass"] and row["candidate_margin_1p2_s_met"]) or
                original["frozen_decision"]["shift_body_y_m"] < 0):
            continue
        baseline = ROOT / row["baseline_result"]
        cases.append({**original,
                      "score_candidate_faults": row["candidate_faults"],
                      "source_baseline_result": row["baseline_result"],
                      "source_baseline_sha256": sha(baseline),
                      "source_baseline_trace": row["baseline_trace"],
                      "source_baseline_trace_sha256": sha(ROOT / row["baseline_trace"]),
                      "source_baseline_contact_s": row["baseline_contact_s"],
                      "source_baseline_native_pass": row["baseline_native_pass"]})
    plan = {
        "schema": "robot-learning-lab.cove50-v141-opened-score-failures-plan.v1",
        "role": "post_score_opened_diagnosis_not_independent_score",
        "body_role": "rejected_historical_direct_frame_sandbox",
        "source_sha256": {"score_plan": sha(SCORE_PLAN),
                          "score_matrix": sha(SCORE_MATRIX),
                          "shift_evaluator": score_plan["source_sha256"]["shift_evaluator"],
                          "runner": sha(Path(__file__))},
        "method": ("After recording V139's failed score, replay only its "
                   "failed frozen-unchanged choices with a prescribed −2 mm "
                   "target. Compare to each exact saved baseline and source "
                   "check native RGB entry. This opens those score starts "
                   "for diagnosis; a rescue cannot repair V139's gate."),
        "case_count": len(cases), "cases": cases,
    }
    write_json(PLAN, plan, frozen=True)
    return plan


def run_case(case: dict, plan: dict) -> dict:
    score_plan = json.loads(SCORE_PLAN.read_text())
    source = score_plan["source_sha256"]
    if (sha(ROOT / case["source_baseline_result"]) !=
            case["source_baseline_sha256"] or
            sha(ROOT / case["source_baseline_trace"]) !=
            case["source_baseline_trace_sha256"]):
        raise ValueError(f"Saved V139 baseline changed: {case['id']}")
    native = shifted.run_case(
        {**case, "id": case["id"] + "-opened-neg2",
         "diagnostic_goal_shift_body_y_m": -.002},
        {"source_sha256": {**source,
                           "evaluator": source["shift_evaluator"]}}, DEST)
    source_check.make_pair(
        "V141_post_score_opened_failure_diagnostic", case, case,
        case["source_baseline_result"], native["result"],
        case["source_baseline_contact_s"], native["contact_s"],
        case["source_baseline_native_pass"], native["pass"])
    baseline = json.loads((ROOT / case["source_baseline_result"]).read_text())
    candidate = json.loads((ROOT / native["result"]).read_text())
    features = score_runner.observations.observed_features(candidate)
    baseline_features = score_runner.observations.observed_features(baseline)
    entry_faults = score_runner.entry_parity(baseline_features, features)
    return {
        "id": case["id"], "seed": case["seed"],
        "cohort": case["cohort"], "route_variant": case["route_variant"],
        "selection_mode": native["selection_mode"],
        "original_frozen_decision": case["frozen_decision"],
        "baseline_contact_s": case["source_baseline_contact_s"],
        "diagnostic_contact_s": native["contact_s"],
        "contact_delta_s": round(native["contact_s"] -
                                 case["source_baseline_contact_s"], 3),
        "diagnostic_native_pass": native["pass"],
        "diagnostic_pass": native["pass"] and not entry_faults,
        "diagnostic_margin_1p2_s_met": native["margin_1p2_s_met"],
        "baseline_faults": case["score_candidate_faults"],
        "diagnostic_faults": native["faults"] + entry_faults,
        "entry_parity_faults": entry_faults,
        "baseline_result": case["source_baseline_result"],
        "baseline_trace": case["source_baseline_trace"],
        "diagnostic_result": native["result"],
        "diagnostic_trace": native["trace"],
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--freeze-only", action="store_true")
    args = parser.parse_args()
    plan = freeze()
    print(f"V141 plan sha256={sha(PLAN)} opened failures={plan['case_count']}",
          flush=True)
    if args.freeze_only:
        return
    DEST.mkdir(parents=True, exist_ok=True)
    path = DEST / "matrix.json"
    old = json.loads(path.read_text()) if path.exists() else None
    if old and old["plan_sha256"] != sha(PLAN):
        raise ValueError("Existing V141 matrix belongs to another plan")
    rows = {row["id"]: row for row in old["cases"]} if old else {}
    for case in plan["cases"]:
        if case["id"] in rows:
            continue
        row = run_case(case, plan)
        rows[row["id"]] = row
        ordered = [rows[item["id"]] for item in plan["cases"] if item["id"] in rows]
        write_json(path, {
            "schema": "robot-learning-lab.cove50-v141-opened-score-failures.v1",
            "role": plan["role"], "body_role": plan["body_role"],
            "plan_sha256": sha(PLAN),
            "case_count": len(ordered), "planned_count": plan["case_count"],
            "complete": len(ordered) == plan["case_count"],
            "diagnostic_pass_count": sum(item["diagnostic_pass"] for item in ordered),
            "cases": ordered,
        })
        print(f"V141 {row['id']}: {row['baseline_contact_s']:.2f}→"
              f"{row['diagnostic_contact_s']:.2f}s "
              f"{'pass' if row['diagnostic_pass'] else 'fail'}; "
              f"{len(ordered)}/{plan['case_count']}", flush=True)


if __name__ == "__main__":
    main()
