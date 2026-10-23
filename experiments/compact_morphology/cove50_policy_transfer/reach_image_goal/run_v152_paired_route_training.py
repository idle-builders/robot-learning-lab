"""Freeze and collect varied automatic-versus-prescribed route training pairs.

V152 uses only the 48 already frozen V150 training starts. It is not a fresh
score, an automatic selector, or evidence on a selected mechanical body.
"""

from __future__ import annotations

import argparse
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
V149_SUMMARY = BASE / "v149/entry-exposure-summary.json"
V151_PLAN = BASE / "v151/opened-live-route-ablation-plan.json"
V151_MATRIX = BASE / "v151/native-opened-live-route-ablation/matrix.json"
V151_SUMMARY = BASE / "v151/source-checked-opened-live-route-summary.json"
RESERVE = BASE / "v147/untouched-score-reserve-plan.json"
POINTER = ROOT / "robot/ape-current-design.json"
OUT = BASE / "v152"
PLAN = OUT / "paired-route-training-plan.json"
DEST = OUT / "native-paired-routes"
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
    if PLAN.exists():
        return json.loads(PLAN.read_text())
    source_plan = json.loads(V150_PLAN.read_text())
    source_matrix = json.loads(V150_MATRIX.read_text())
    v151 = json.loads(V151_SUMMARY.read_text())
    reserve = json.loads(RESERVE.read_text())
    if (source_plan["case_count"] != 48 or len(source_plan["cases"]) != 48 or
            source_matrix["plan_sha256"] != sha(V150_PLAN) or
            source_matrix["case_count"] != len(source_matrix["cases"]) or
            source_plan["source_sha256"]["pointer"] != sha(POINTER) or
            source_plan["source_sha256"]["untouched_score_reserve"] != sha(RESERVE) or
            reserve["status"] != "sealed_no_contact_visits" or
            json.loads(POINTER.read_text())["selected_design_id"] is not None or
            v151["checked_forced_visit_count"] != 2 or
            v151["source_sha256"]["matrix"] != sha(V151_MATRIX) or
            v151["source_sha256"]["plan"] != sha(V151_PLAN) or
            {case["seed"] for case in source_plan["cases"]} &
            {case["seed"] for case in reserve["cases"]}):
        raise ValueError("V150 training source, V151 diagnosis, or reserve changed")
    cases = []
    for source in source_plan["cases"]:
        label = ("forced-specialist" if source["entry_selection_mode"] ==
                 "v21_scoped_y_locked" else "forced-scoped")
        filename, route, flag = EVALUATORS[label]
        cases.append({**source, "id": f"route-pair-{source['seed']}",
                      "v150_case_id": source["id"],
                      "prescribed_label": label,
                      "prescribed_route": route,
                      "prescribed_flag": flag,
                      "prescribed_evaluator_sha256":
                      sha(Path(__file__).with_name(filename)),
                      "reuse_v151_source": source["seed"] == 24591})
    if (len(cases) != 48 or
            sum(case["reuse_v151_source"] for case in cases) != 1 or
            next(case for case in cases if case["reuse_v151_source"])
            ["prescribed_label"] != "forced-scoped"):
        raise ValueError("Expected exactly one matching V151 source reuse")
    source_hashes = {
        "v150_plan": sha(V150_PLAN),
        "v149_entry_summary": sha(V149_SUMMARY),
        "v151_plan": sha(V151_PLAN),
        "v151_matrix": sha(V151_MATRIX),
        "v151_summary": sha(V151_SUMMARY),
        "untouched_score_reserve": sha(RESERVE),
        "pointer": sha(POINTER),
        "runner": sha(Path(__file__)),
        **{label.replace("-", "_") + "_evaluator":
           sha(Path(__file__).with_name(filename))
           for label, (filename, _, _) in EVALUATORS.items()},
    }
    plan = {
        "schema": "robot-learning-lab.cove50-v152-paired-route-training-plan.v1",
        "role": "varied_prescribed_route_training_not_independent_score",
        "body_role": "rejected_historical_direct_frame_sandbox",
        "timing_disclosure": (
            f"Frozen after {source_matrix['case_count']}/48 V150 target triplets "
            "and both V151 forced-route outcomes were visible. All 48 seed "
            "identities and physical starts were selected in V150 before those "
            "contacts. Prescribed route is determined only by the prior V149 "
            "entry mode, not by V150 contact. V151 seed 24591 forced-scoped "
            "native source will be reused, not rerun. This is training data."),
        "question": (
            "Across varied training starts, when does changing the arm route "
            "alone improve or harm full native contact under the same observed "
            "RGB entry? Can a later observed-input selector learn that distinction?"),
        "route_rule": (
            "If V149 entry chose v21_scoped_y_locked, prescribe "
            "v16_specialist_locked; otherwise prescribe v21_scoped_y_locked."),
        "training_audit_gate": (
            "Any fitted observed-input selector needs whole-seed withholding, "
            "both physical cohorts represented among held failures, selected "
            "full-gate and 1.2 s margin repairs with zero selected regressions "
            "or unmeasured alternatives. Then check older-start native retention "
            "and online integration before opening the 24-start score reserve."),
        "source_sha256": source_hashes,
        "case_count": 48, "new_native_visit_count": 47,
        "reused_native_visit_count": 1,
        "cases": cases,
    }
    write_json(PLAN, plan)
    return plan


def validate_plan(plan: dict) -> tuple[dict, dict]:
    source_plan = json.loads(V150_PLAN.read_text())
    v150 = json.loads(V150_MATRIX.read_text())
    reserve = json.loads(RESERVE.read_text())
    refs = plan["source_sha256"]
    if (refs["runner"] != sha(Path(__file__)) or
            refs["v150_plan"] != sha(V150_PLAN) or
            refs["v149_entry_summary"] != sha(V149_SUMMARY) or
            refs["v151_plan"] != sha(V151_PLAN) or
            refs["v151_matrix"] != sha(V151_MATRIX) or
            refs["v151_summary"] != sha(V151_SUMMARY) or
            refs["untouched_score_reserve"] != sha(RESERVE) or
            refs["pointer"] != sha(POINTER) or
            reserve["status"] != "sealed_no_contact_visits" or
            json.loads(POINTER.read_text())["selected_design_id"] is not None or
            v150["plan_sha256"] != sha(V150_PLAN) or
            not v150["complete"] or v150["case_count"] != 48 or
            len(v150["cases"]) != 48 or
            len(plan["cases"]) != 48):
        raise ValueError("Complete V150 source and sealed V152 route plan required")
    for label, (filename, _, _) in EVALUATORS.items():
        if refs[label.replace("-", "_") + "_evaluator"] != sha(
                Path(__file__).with_name(filename)):
            raise ValueError(f"Forced evaluator changed: {label}")
    if [case["v150_case_id"] for case in plan["cases"]] != [
            case["id"] for case in source_plan["cases"]]:
        raise ValueError("V152 case order or membership changed")
    return source_plan, v150


def run_case(case: dict, v150_case: dict, baseline: dict,
             source_hashes: dict) -> dict:
    filename, route, flag = EVALUATORS[case["prescribed_label"]]
    if (case["prescribed_route"] != route or
            case["prescribed_flag"] != flag or
            case["prescribed_evaluator_sha256"] !=
            sha(Path(__file__).with_name(filename))):
        raise ValueError(f"Prescribed route changed: {case['id']}")
    base_report = verify_report(
        baseline["baseline_result"], baseline["baseline_trace"], case, 0,
        baseline["baseline_contact_s"], source_hashes["evaluator"])
    verify_trace(baseline["baseline_trace"], base_report)
    base_observed = score_runner.observations.observed_features(base_report)
    if (base_observed != baseline["full_entry_observed_features"] or
            score_runner.entry_parity(case["entry_observed_features"], base_observed) or
            baseline["screen_to_full_entry_faults"] or
            baseline["selection_mode"] != case["entry_selection_mode"] or
            base_report["body_sha256"] != source_hashes["body"] or
            base_report["selected_pointer_sha256"] != source_hashes["pointer"] or
            base_report["reach_checkpoint_sha256"] != source_hashes["actor"]):
        raise ValueError(f"Baseline source or RGB entry changed: {case['id']}")
    if case["reuse_v151_source"]:
        v151 = json.loads(V151_MATRIX.read_text())
        result = next(row for row in v151["cases"] if row["id"] ==
                      "opened-live-24591-forced-scoped")
        if (v151["plan_sha256"] != sha(V151_PLAN) or
                not v151["complete"] or v151["case_count"] != 2 or
                result["forced_route"] != route or
                result["numeric_rgb_entry_faults"]):
            raise ValueError("V151 reused route source changed")
    else:
        original = native.EVALUATOR
        native.EVALUATOR = Path(__file__).with_name(filename)
        try:
            result = native.run_case(
                case, {"source_sha256": {**source_hashes,
                         "evaluator": case["prescribed_evaluator_sha256"]}}, DEST)
        finally:
            native.EVALUATOR = original
    forced_report = verify_report(
        result["result"], result["trace"], case, 0, result["contact_s"],
        case["prescribed_evaluator_sha256"])
    verify_trace(result["trace"], forced_report)
    forced_observed = score_runner.observations.observed_features(forced_report)
    numeric_faults = score_runner.entry_parity(
        base_observed, {**forced_observed,
                        "selection_mode": base_observed["selection_mode"],
                        "route_goal_source": base_observed["route_goal_source"]})
    if (numeric_faults or not forced_report.get(flag) or
            forced_report["selection_mode"] != route or
            forced_report["body_sha256"] != source_hashes["body"] or
            forced_report["selected_pointer_sha256"] != source_hashes["pointer"] or
            forced_report["reach_checkpoint_sha256"] != source_hashes["actor"]):
        raise ValueError(f"Forced route source or RGB entry differed: {case['id']}")
    passed = result["pass"] and not numeric_faults
    return {
        "id": case["id"], "seed": case["seed"],
        "cohort": case["cohort"], "route_variant": case["route_variant"],
        "entry_observed_features": base_observed,
        "automatic_route": baseline["selection_mode"],
        "prescribed_route": route,
        "prescribed_label": case["prescribed_label"],
        "reused_v151_source": case["reuse_v151_source"],
        "baseline_contact_s": baseline["baseline_contact_s"],
        "baseline_pass": baseline["baseline_pass"],
        "baseline_margin_1p2_s_met": baseline["baseline_margin_1p2_s_met"],
        "baseline_faults": baseline["baseline_faults"],
        "baseline_report": baseline["baseline_result"],
        "baseline_trace": baseline["baseline_trace"],
        "baseline_report_sha256": sha(ROOT / baseline["baseline_result"]),
        "baseline_trace_sha256": sha(ROOT / baseline["baseline_trace"]),
        "prescribed_contact_s": result["contact_s"],
        "prescribed_pass": passed,
        "prescribed_margin_1p2_s_met": result["margin_1p2_s_met"],
        "prescribed_faults": result["faults"] + numeric_faults,
        "numeric_rgb_entry_faults": numeric_faults,
        "prescribed_report": result["result"],
        "prescribed_trace": result["trace"],
        "prescribed_report_sha256": sha(ROOT / result["result"]),
        "prescribed_trace_sha256": sha(ROOT / result["trace"]),
    }


def write_matrix(plan: dict, rows: dict[str, dict]) -> dict:
    ordered = [rows[case["id"]] for case in plan["cases"] if case["id"] in rows]
    value = {
        "schema": "robot-learning-lab.cove50-v152-paired-route-training.v1",
        "role": plan["role"], "body_role": plan["body_role"],
        "plan_sha256": sha(PLAN),
        "case_count": len(ordered), "planned_count": 48,
        "complete": len(ordered) == 48,
        "source_reused_count": sum(row["reused_v151_source"] for row in ordered),
        "numeric_rgb_entry_failure_count": sum(bool(row["numeric_rgb_entry_faults"])
                                               for row in ordered),
        "prescribed_rescue_count": sum(not row["baseline_pass"] and
                                       row["prescribed_pass"] for row in ordered),
        "prescribed_regression_count": sum(row["baseline_pass"] and
                                           not row["prescribed_pass"] for row in ordered),
        "prescribed_margin_repair_count": sum(
            not row["baseline_margin_1p2_s_met"] and
            row["prescribed_margin_1p2_s_met"] and row["prescribed_pass"]
            for row in ordered),
        "prescribed_margin_loss_count": sum(
            row["baseline_margin_1p2_s_met"] and
            not row["prescribed_margin_1p2_s_met"] for row in ordered),
        "cases": ordered,
    }
    write_json(MATRIX, value)
    return value


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--freeze-only", action="store_true")
    args = parser.parse_args()
    plan = freeze()
    print(f"V152 paired-route plan sha256={sha(PLAN)}; "
          "48 varied pairs, 47 new forced CUDA visits", flush=True)
    if args.freeze_only:
        return
    source_plan, v150 = validate_plan(plan)
    source_hashes = source_plan["source_sha256"]
    v150_rows = {row["id"]: row for row in v150["cases"]}
    old = json.loads(MATRIX.read_text()) if MATRIX.exists() else None
    if old and old["plan_sha256"] != sha(PLAN):
        raise ValueError("Existing V152 matrix belongs to another plan")
    rows = {row["id"]: row for row in old["cases"]} if old else {}
    for case in plan["cases"]:
        if case["id"] in rows:
            continue
        v150_case = next(source for source in source_plan["cases"]
                         if source["id"] == case["v150_case_id"])
        row = run_case(case, v150_case, v150_rows[case["v150_case_id"]],
                       source_hashes)
        rows[case["id"]] = row
        value = write_matrix(plan, rows)
        print(f"V152 {case['id']}: automatic {row['baseline_contact_s']:.2f}s, "
              f"prescribed {row['prescribed_contact_s']:.2f}s; "
              f"{value['case_count']}/48", flush=True)


if __name__ == "__main__":
    main()
