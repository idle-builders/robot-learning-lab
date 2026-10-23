"""Select one contact start per new seed family from checked RGB entries.

The rule is frozen in V158 before V156 entries. This builder waits for the full
V156 camera screen and V157 RGB repeatability diagnostic; it runs no contact.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[4]
BASE = ROOT / "artifacts/ape-cove50-reach-goal-channels-20261010"
METHOD = BASE / "v158/new-seed-contact-selection-method-freeze.json"
OUT = BASE / "v158/new-seed-contact-plan.json"
V156_PLAN = BASE / "v156/independent-seed-entry-family-plan.json"
V156_MATRIX = BASE / "v156/native-entry-only/matrix.json"
V156_SUMMARY = BASE / "v156/source-checked-entry-family-summary.json"
V157_PLAN = BASE / "v157/exact-rgb-repeatability-plan.json"
V157_MATRIX = BASE / "v157/native-entry-only/matrix.json"
V157_SUMMARY = BASE / "v157/source-checked-rgb-repeatability-summary.json"
V149_PLAN = BASE / "v149/new-seed-route-exposure-plan.json"
V150_PLAN = BASE / "v150/paired-contact-training-plan.json"
RESERVE = BASE / "v147/untouched-score-reserve-plan.json"
POINTER = ROOT / "robot/ape-current-design.json"
METHOD_SOURCE = Path(__file__).with_name("freeze_v158_new_seed_contact_method.py")
V156_CHECKER = Path(__file__).with_name("summarize_v156_independent_seed_entries.py")
V157_CHECKER = Path(__file__).with_name("summarize_v157_rgb_repeatability.py")
FORCED_EVALUATORS = {
    "v21_scoped_y_locked": Path(__file__).with_name("evaluate_forced_scoped_v107.py"),
    "v16_specialist_locked": Path(__file__).with_name("evaluate_forced_specialist_v102.py"),
}


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def checked_json(path: Path) -> dict:
    return json.loads(path.read_text())


def image_for(report: Path) -> Path:
    return report.with_name(report.stem + "-entry.png")


def main() -> None:
    prerequisites = (V156_MATRIX, V156_SUMMARY, V157_MATRIX, V157_SUMMARY)
    missing = [str(path.relative_to(ROOT)) for path in prerequisites
               if not path.is_file()]
    if missing:
        raise ValueError(f"Complete source-checked RGB screens required: {missing}")
    method = checked_json(METHOD)
    entry_plan = checked_json(V156_PLAN)
    entry_matrix = checked_json(V156_MATRIX)
    entry_summary = checked_json(V156_SUMMARY)
    repeat_plan = checked_json(V157_PLAN)
    repeat_matrix = checked_json(V157_MATRIX)
    repeat_summary = checked_json(V157_SUMMARY)
    earlier_entry = checked_json(V149_PLAN)
    native_plan = checked_json(V150_PLAN)
    reserve = checked_json(RESERVE)
    pointer = checked_json(POINTER)
    refs = method["source_sha256"]
    if (refs["method_source"] != sha(METHOD_SOURCE) or
            refs["v156_entry_plan"] != sha(V156_PLAN) or
            refs["v157_repeatability_plan"] != sha(V157_PLAN) or
            refs["untouched_score_reserve"] != sha(RESERVE) or
            refs["mechanical_pointer"] != sha(POINTER) or
            pointer["selected_design_id"] is not None or
            reserve["status"] != "sealed_no_contact_visits" or
            method["contains_contact_results"] or
            method["contains_policy_result"] or
            entry_plan["family_count"] != 24 or
            entry_plan["case_count"] != 168 or
            entry_plan["source_sha256"]["v149_plan"] != sha(V149_PLAN) or
            entry_plan["source_sha256"]["original_v95_evaluator"] !=
            native_plan["source_sha256"]["evaluator"] or
            any(entry_plan["source_sha256"][key] !=
                native_plan["source_sha256"][key] for key in (
                    "body", "pointer", "actor", "alternate_actor",
                    "specialist_actor", "calibration", "geometry",
                    "reference", "squat")) or
            entry_matrix["plan_sha256"] != sha(V156_PLAN) or
            not entry_matrix["complete"] or entry_matrix["case_count"] != 168 or
            entry_matrix["contains_contact_outcomes"] or
            entry_summary["source_sha256"]["plan"] != sha(V156_PLAN) or
            entry_summary["source_sha256"]["matrix"] != sha(V156_MATRIX) or
            entry_summary["source_sha256"]["checker"] != sha(V156_CHECKER) or
            not entry_summary["complete"] or
            entry_summary["source_checked_entry_count"] != 168 or
            repeat_plan["source_sha256"]["v156_plan"] != sha(V156_PLAN) or
            repeat_matrix["plan_sha256"] != sha(V157_PLAN) or
            not repeat_matrix["complete"] or repeat_matrix["case_count"] != 24 or
            repeat_matrix["contains_contact_outcomes"] or
            repeat_summary["source_sha256"]["plan"] != sha(V157_PLAN) or
            repeat_summary["source_sha256"]["matrix"] != sha(V157_MATRIX) or
            repeat_summary["source_sha256"]["checker"] != sha(V157_CHECKER) or
            not repeat_summary["complete"] or
            repeat_summary["source_checked_repeat_count"] != 24 or
            len({row["id"] for row in entry_matrix["cases"]}) != 168):
        raise ValueError("Complete, source-checked RGB screens and sealed reserve required")
    if (len({row["seed"] for row in repeat_matrix["cases"]}) != 24 or
            {row["seed"] for row in repeat_matrix["cases"]} !=
            {family["seed"] for family in entry_plan["families"]}):
        raise ValueError("V157 does not cover every new seed family")
    for summary in (entry_summary, repeat_summary):
        for source in summary["source_files"]:
            for key in ("report", "trace", "image"):
                if key in source and sha(ROOT / source[key]) != source[f"{key}_sha256"]:
                    raise ValueError(f"Source file changed after RGB check: {source[key]}")
    cases = {case["id"]: case for case in entry_plan["cases"]}
    rows = {row["id"]: row for row in entry_matrix["cases"]}
    anchor_routes = {case["seed"]: case["route_variant"]
                     for case in earlier_entry["cases"]}
    if set(rows) != set(cases):
        raise ValueError("V156 native matrix does not cover exact frozen entries")
    held = set(method["held_seed_ids"])
    if (len(held) != 6 or
            held != {24606, 24612, 24615, 24618, 24621, 24624}):
        raise ValueError("V158 held seed split changed")
    selected = []
    unselected = []
    for family in entry_plan["families"]:
        available = [rows[case["id"]] for case in entry_plan["cases"]
                     if case["family_id"] == family["id"] and
                     rows[case["id"]]["entry_available"]]
        if not available:
            unselected.append({"family_id": family["id"], "seed": family["seed"],
                               "stratum": family["stratum"],
                               "split": "held" if family["seed"] in held else
                               "development",
                               "reason": "no source-checked controlled RGB entry"})
            continue
        target = [row for row in available
                  if row["selection_mode"] == family["target_route_exposure"]]
        pool = target or available
        row = min(pool, key=lambda item: (
            cases[item["id"]]["toy_delta_x_m"] ** 2 +
            cases[item["id"]]["toy_delta_y_m"] ** 2, item["id"]))
        case = cases[row["id"]]
        report = ROOT / row["result"]
        trace = report.with_suffix(".npz")
        image = image_for(report)
        if (row["seed"] != family["seed"] or
                row["family_id"] != family["id"] or
                row["observed_features"] is None or
                row["observed_features"]["selection_mode"] !=
                row["selection_mode"] or
                sha(report) != row["result_sha256"] or
                sha(trace) != row["trace_sha256"] or
                not image.is_file()):
            raise ValueError(f"Selected RGB source changed: {row['id']}")
        alternate = ("v16_specialist_locked" if row["selection_mode"] ==
                     "v21_scoped_y_locked" else "v21_scoped_y_locked")
        selected.append({
            **case,
            "id": f"contact-{family['seed']}",
            "route_variant": anchor_routes[family["opened_anchor_seed_audit_only"]],
            "source_v156_entry_id": row["id"],
            "split": "held" if family["seed"] in held else "development",
            "target_route_absent": not bool(target),
            "automatic_route_at_screen": row["selection_mode"],
            "prescribed_route": alternate,
            "entry_observed_features": row["observed_features"],
            "entry_report": row["result"],
            "entry_report_sha256": row["result_sha256"],
            "entry_trace_sha256": row["trace_sha256"],
            "entry_image": str(image.relative_to(ROOT)),
            "entry_image_sha256": sha(image),
        })
    if (len(selected) + len(unselected) != 24 or
            len({row["seed"] for row in selected}) != len(selected) or
            {row["seed"] for row in selected} &
            {case["seed"] for case in reserve["cases"]}):
        raise ValueError("V158 family or sealed-score membership changed")
    plan = {
        "schema": "robot-learning-lab.cove50-v158-new-seed-contact-plan.v1",
        "role": "source_checked_entry_selected_development_and_held_contact_plan_not_result",
        "body_role": method["body_role"],
        "source_sha256": {
            "method": sha(METHOD), "builder": sha(Path(__file__)),
            "development_runner": sha(Path(__file__).with_name(
                "run_v158_new_seed_development_contact.py")),
            "v156_plan": sha(V156_PLAN), "v156_matrix": sha(V156_MATRIX),
            "v156_summary": sha(V156_SUMMARY),
            "v157_plan": sha(V157_PLAN), "v157_matrix": sha(V157_MATRIX),
            "v157_summary": sha(V157_SUMMARY),
            "v149_plan": sha(V149_PLAN), "v150_plan": sha(V150_PLAN),
            "untouched_score_reserve": sha(RESERVE),
            "mechanical_pointer": sha(POINTER),
        },
        "native_source_sha256": native_plan["source_sha256"],
        "forced_evaluator_sha256": {
            route: sha(path) for route, path in FORCED_EVALUATORS.items()},
        "selection_method": method["entry_selection_rule"],
        "held_split_rule": method["held_split_rule"],
        "held_seed_ids": method["held_seed_ids"],
        "family_count": 24,
        "selected_count": len(selected),
        "unselected_count": len(unselected),
        "development_count": sum(row["split"] == "development" for row in selected),
        "held_count": sum(row["split"] == "held" for row in selected),
        "contains_contact_results": False,
        "cases": selected,
        "unselected_families": unselected,
    }
    encoded = json.dumps(plan, indent=2) + "\n"
    OUT.parent.mkdir(parents=True, exist_ok=True)
    if OUT.exists() and OUT.read_text() != encoded:
        raise ValueError("V158 contact plan changed after freeze")
    OUT.write_text(encoded)
    print(f"V158 selected {len(selected)}/24 new-seed families for future contact; "
          f"{plan['development_count']} development, {plan['held_count']} held; "
          f"sha256={sha(OUT)}")


if __name__ == "__main__":
    main()
