"""Whole-seed route-selector audit on frozen V158 development contacts.

The older V152 pairs are training data in every fold. Each new-seed V158
development family is withheld whole. Invalid RGB pairs stay unresolved if a
rule selects them. No held V158 or independent V147 contact is read.
"""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path

from experiments.compact_morphology.cove50_policy_transfer.reach_image_goal import (
    audit_v153_route_selector as previous,
    run_v139_reserved_contact as score_runner,
)


ROOT = Path(__file__).resolve().parents[4]
BASE = ROOT / "artifacts/ape-cove50-reach-goal-channels-20261010"
METHOD = BASE / "v160/new-seed-selector-audit-method-freeze.json"
METHOD_SOURCE = Path(__file__).with_name(
    "freeze_v160_new_seed_selector_audit_method.py")
V152_PLAN = BASE / "v152/paired-route-training-plan.json"
V152_MATRIX = BASE / "v152/native-paired-routes/matrix.json"
V152_SUMMARY = BASE / "v152/partial-source-checked-paired-route-summary.json"
V158_METHOD = BASE / "v158/new-seed-contact-selection-method-freeze.json"
V158_PLAN = BASE / "v158/new-seed-contact-plan.json"
V158_MATRIX = BASE / "v158/native-development-contact/matrix.json"
V158_SUMMARY = BASE / "v158/source-checked-development-contact-summary.json"
RESERVE = BASE / "v147/untouched-score-reserve-plan.json"
POINTER = ROOT / "robot/ape-current-design.json"
OUT = BASE / "v160/new-seed-selector-grouped-audit.json"


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def checked_sources(summary: dict) -> None:
    for item in summary["source_files"]:
        for key in ("report", "trace"):
            path = ROOT / item[key]
            if not path.is_file() or sha(path) != item[f"{key}_sha256"]:
                raise ValueError(f"Native source changed: {path}")


def load_rows() -> tuple[list[dict], list[dict], dict]:
    needed = (METHOD, V152_PLAN, V152_MATRIX, V152_SUMMARY, V158_METHOD,
              V158_PLAN, V158_MATRIX, V158_SUMMARY, RESERVE, POINTER)
    if any(not path.is_file() for path in needed):
        raise ValueError("Complete source-checked V158 development contacts required")
    method = json.loads(METHOD.read_text())
    old_plan = json.loads(V152_PLAN.read_text())
    old_matrix = json.loads(V152_MATRIX.read_text())
    old_summary = json.loads(V152_SUMMARY.read_text())
    plan = json.loads(V158_PLAN.read_text())
    matrix = json.loads(V158_MATRIX.read_text())
    summary = json.loads(V158_SUMMARY.read_text())
    reserve = json.loads(RESERVE.read_text())
    pointer = json.loads(POINTER.read_text())
    refs = method["source_sha256"]
    if (refs["audit_source"] != sha(Path(__file__)) or
            refs["method_source"] != sha(METHOD_SOURCE) or
            refs["v153_fit_source"] != sha(Path(previous.__file__)) or
            refs["v152_plan"] != sha(V152_PLAN) or
            refs["v152_matrix"] != sha(V152_MATRIX) or
            refs["v152_summary"] != sha(V152_SUMMARY) or
            refs["v158_selection_method"] != sha(V158_METHOD) or
            refs["reserve"] != sha(RESERVE) or
            refs["pointer"] != sha(POINTER) or
            method["contains_contact_results"] or
            reserve["status"] != "sealed_no_contact_visits" or
            pointer["selected_design_id"] is not None or
            pointer["status"] != "rejected_no_selected_mechanical_design" or
            old_summary["source_sha256"]["plan"] != sha(V152_PLAN) or
            old_summary["source_sha256"]["matrix"] != sha(V152_MATRIX) or
            old_summary["source_sha256"]["checker"] != sha(
                Path(__file__).with_name("summarize_v152_paired_route_training.py")) or
            old_summary["source_checked_pair_count"] != 47 or
            old_summary["numeric_rgb_entry_failure_count"] != 0 or
            old_matrix["case_count"] != 47 or old_matrix["complete"] or
            old_matrix["plan_sha256"] != sha(V152_PLAN) or
            len(old_plan["cases"]) != 48 or
            plan["source_sha256"]["method"] != sha(V158_METHOD) or
            matrix["plan_sha256"] != sha(V158_PLAN) or
            not matrix["complete"] or
            matrix["case_count"] != plan["development_count"] or
            matrix["held_contact_visit_count"] != 0 or
            summary["source_sha256"]["plan"] != sha(V158_PLAN) or
            summary["source_sha256"]["matrix"] != sha(V158_MATRIX) or
            summary["source_sha256"]["checker"] != sha(
                Path(__file__).with_name(
                    "summarize_v158_new_seed_development_contact.py")) or
            not summary["complete"] or
            summary["source_checked_family_count"] != matrix["case_count"] or
            summary["held_contact_visit_count"] != 0 or
            plan["contains_contact_results"] or
            plan["held_seed_ids"] != method["held_seed_ids"]):
        raise ValueError("Frozen selector method or source-checked training boundary changed")
    checked_sources(old_summary)
    checked_sources(summary)
    old_cases = {case["id"]: case for case in old_plan["cases"]}
    new_cases = {case["id"]: case for case in plan["cases"]
                 if case["split"] == "development"}
    if (len(old_cases) != 48 or len(new_cases) != plan["development_count"] or
            {row["id"] for row in matrix["cases"]} != set(new_cases) or
            len({row["seed"] for row in matrix["cases"]}) != len(matrix["cases"]) or
            {row["seed"] for row in matrix["cases"]} &
            {row["seed"] for row in old_matrix["cases"]} or
            {row["seed"] for row in matrix["cases"]} &
            {row["seed"] for row in reserve["cases"]}):
        raise ValueError("Old, development and reserved seed groups overlap or differ")
    old_rows = []
    for row in old_matrix["cases"]:
        case = old_cases[row["id"]]
        if (row["seed"] != case["seed"] or
                row["prescribed_route"] != case["prescribed_route"] or
                row["numeric_rgb_entry_faults"] or
                score_runner.entry_parity(case["entry_observed_features"],
                                          row["entry_observed_features"])):
            raise ValueError(f"V152 historical training row changed: {row['id']}")
        old_rows.append({
            "seed_audit_only": row["seed"],
            "cohort_audit_only": row["cohort"],
            "features": row["entry_observed_features"],
            "prescribed_route": row["prescribed_route"],
            "baseline_contact_s": row["baseline_contact_s"],
            "baseline_pass": row["baseline_pass"],
            "baseline_margin": row["baseline_margin_1p2_s_met"],
            "alternate_contact_s": row["prescribed_contact_s"],
            "alternate_pass": row["prescribed_pass"],
            "alternate_margin": row["prescribed_margin_1p2_s_met"],
        })
    development = []
    for row in matrix["cases"]:
        case = new_cases[row["id"]]
        if (row["seed"] != case["seed"] or
                row["split"] != "development" or
                row["prescribed_route"] != case["prescribed_route"] or
                row["cohort"] != case["cohort"]):
            raise ValueError(f"V158 development case changed: {row['id']}")
        development.append({
            "id": row["id"],
            "seed_audit_only": row["seed"],
            "cohort_audit_only": row["cohort"],
            "features": row["automatic_entry_observed_features"],
            "prescribed_route": row["prescribed_route"],
            "matched_pair_valid": row["matched_pair_valid"],
            "invalid_reasons": row["failure_reasons"],
            "baseline_contact_s": row["automatic_contact_s"],
            "baseline_pass": row["automatic_native_pass"],
            "baseline_margin": row["automatic_margin_1p2_s_met"],
            "alternate_contact_s": row["prescribed_contact_s"],
            "alternate_pass": row["prescribed_native_pass"],
            "alternate_margin": row["prescribed_margin_1p2_s_met"],
        })
    return old_rows, development, {
        "method": sha(METHOD), "v152_plan": sha(V152_PLAN),
        "v152_matrix": sha(V152_MATRIX), "v152_summary": sha(V152_SUMMARY),
        "v158_method": sha(V158_METHOD), "v158_plan": sha(V158_PLAN),
        "v158_matrix": sha(V158_MATRIX), "v158_summary": sha(V158_SUMMARY),
        "reserve": sha(RESERVE), "pointer": sha(POINTER),
    }


def decision(rules: list[dict], row: dict) -> str:
    if row["features"] is None:
        return "unchanged"
    return previous.predict(rules, row)


def audit(old: list[dict], development: list[dict]) -> tuple[list[dict], dict]:
    decisions = []
    for row in development:
        training = old + [other for other in development
                          if other["seed_audit_only"] != row["seed_audit_only"]
                          and other["matched_pair_valid"] and
                          other["features"] is not None]
        rules = previous.fit(training)
        choice = decision(rules, row)
        selected = choice != "unchanged"
        valid = row["matched_pair_valid"]
        rescue = (selected and valid and not row["baseline_pass"] and
                  row["alternate_pass"] and row["alternate_margin"])
        regression = (selected and valid and row["baseline_pass"] and
                      not row["alternate_pass"])
        margin_loss = (selected and valid and row["baseline_margin"] and
                       not row["alternate_margin"])
        unresolved = selected and not valid
        decisions.append({
            "id": row["id"], "seed_audit_only": row["seed_audit_only"],
            "cohort_audit_only": row["cohort_audit_only"],
            "automatic_route": (row["features"] or {}).get("selection_mode"),
            "selected_route": choice,
            "matched_pair_valid": valid,
            "baseline_contact_s": row["baseline_contact_s"],
            "prescribed_contact_s": row["alternate_contact_s"],
            "baseline_full_gate_pass": row["baseline_pass"],
            "prescribed_full_gate_pass": row["alternate_pass"],
            "baseline_margin_1p2_s_met": row["baseline_margin"],
            "prescribed_margin_1p2_s_met": row["alternate_margin"],
            "selected_full_gate_margin_rescue": rescue,
            "selected_full_gate_regression": regression,
            "selected_margin_loss": margin_loss,
            "selected_unresolved_alternative": unresolved,
            "failure_reasons": (row["invalid_reasons"] if unresolved else
                                ["selected route failed native full gate"] if
                                selected and not row["alternate_pass"] else
                                ["selected route missed 1.2 s margin"] if
                                selected and not row["alternate_margin"] else []),
        })
    chosen = [row for row in decisions if row["selected_route"] != "unchanged"]
    rescues = [row for row in chosen if row["selected_full_gate_margin_rescue"]]
    summary = {
        "development_seed_count": len(development),
        "valid_matched_pair_count": sum(row["matched_pair_valid"] for row in development),
        "invalid_matched_pair_count": sum(not row["matched_pair_valid"] for row in development),
        "held_out_selected_count": len(chosen),
        "held_out_full_gate_margin_rescue_count": len(rescues),
        "held_out_rescue_cohorts": sorted({row["cohort_audit_only"] for row in rescues}),
        "held_out_full_gate_regression_count": sum(
            row["selected_full_gate_regression"] for row in chosen),
        "held_out_margin_loss_count": sum(
            row["selected_margin_loss"] for row in chosen),
        "held_out_unresolved_count": sum(
            row["selected_unresolved_alternative"] for row in chosen),
    }
    return decisions, summary


def main() -> None:
    old, development, sources = load_rows()
    decisions, result = audit(old, development)
    expected_cohorts = {"opened_roll_high_neighborhood",
                        "opened_pitch_low_neighborhood"}
    gate = (result["held_out_full_gate_margin_rescue_count"] >= 2 and
            set(result["held_out_rescue_cohorts"]) == expected_cohorts and
            result["held_out_full_gate_regression_count"] == 0 and
            result["held_out_margin_loss_count"] == 0 and
            result["held_out_unresolved_count"] == 0)
    valid_development = [row for row in development
                         if row["matched_pair_valid"] and row["features"] is not None]
    full_rules = previous.fit(old + valid_development)
    report = {
        "schema": "robot-learning-lab.cove50-v160-new-seed-selector-audit.v1",
        "role": "whole_new_seed_development_audit_not_held_or_independent_score",
        "body_role": "rejected_historical_direct_frame_sandbox",
        "source_sha256": sources,
        "historical_training_seed_count": len(old),
        "audit_split": "Leave each V158 development simulator seed out; retain only source-valid V152 historical and other V158 development pairs in each fit.",
        "runtime_inputs": "Used controlled RGB entry, goal source, automatic route and motor/body settings only; seed and cohort are audit fields, never runtime inputs.",
        "development_gate_met": gate,
        **result,
        "held_out_decisions": decisions,
        "full_fit_rules_diagnostic_only": full_rules,
        "held_contact_visit_count": 0,
        "independent_score_visit_count": 0,
        "next_step": ("Freeze exact full-fit model and inspect older-start retention and online integration before held contact" if gate else
                      "No selector promotion or held contact; collect new source-frozen development families if independent failure support is insufficient"),
    }
    OUT.parent.mkdir(parents=True, exist_ok=True)
    temporary = OUT.with_suffix(".json.tmp")
    temporary.write_text(json.dumps(report, indent=2) + "\n")
    os.replace(temporary, OUT)
    print(json.dumps({"development_gate_met": gate, **result}))


if __name__ == "__main__":
    main()
