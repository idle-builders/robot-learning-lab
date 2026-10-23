"""Collect frozen new-seed development route pairs, never held contacts.

Every physical start comes from the V158 camera-only selection plan. A mismatch
in the used RGB goal is retained as an invalid visit, never replaced by a
diagnostic repeat or a different toy placement in the same seed family.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path

import numpy as np

from experiments.compact_morphology.cove50_policy_transfer.reach_image_goal import (
    build_v158_new_seed_contact_plan as builder,
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
PLAN = BASE / "v158/new-seed-contact-plan.json"
METHOD = BASE / "v158/new-seed-contact-selection-method-freeze.json"
RESERVE = BASE / "v147/untouched-score-reserve-plan.json"
POINTER = ROOT / "robot/ape-current-design.json"
DEST = BASE / "v158/native-development-contact"
MATRIX = DEST / "matrix.json"
ENTRY_QPOS_MAX_ABS_TOLERANCE = 1e-4
FORCED = {
    "v21_scoped_y_locked": ("evaluate_forced_scoped_v107.py",
                            "diagnostic_force_scoped"),
    "v16_specialist_locked": ("evaluate_forced_specialist_v102.py",
                              "diagnostic_force_specialist"),
}


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def write_json(path: Path, value: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2) + "\n")
    os.replace(temporary, path)


def validate_plan(plan: dict) -> list[dict]:
    # Recompute the one-start-per-family choice from every frozen V156/V157
    # source before accepting a resumed contact plan. The builder refuses any
    # changed plan bytes; selected-entry hashes alone would miss a changed
    # unselected entry or a changed family-level route ranking.
    builder.main()
    if plan != json.loads(PLAN.read_text()):
        raise ValueError("V158 contact plan changed while validating native visits")
    method = json.loads(METHOD.read_text())
    reserve = json.loads(RESERVE.read_text())
    pointer = json.loads(POINTER.read_text())
    refs = plan["source_sha256"]
    native_source = plan["native_source_sha256"]
    development = [case for case in plan["cases"]
                   if case["split"] == "development"]
    held = [case for case in plan["cases"] if case["split"] == "held"]
    if (refs["builder"] != sha(Path(builder.__file__)) or
            refs["development_runner"] != sha(Path(__file__)) or
            refs["method"] != sha(METHOD) or
            refs["untouched_score_reserve"] != sha(RESERVE) or
            refs["mechanical_pointer"] != sha(POINTER) or
            native_source["evaluator"] != sha(native.EVALUATOR) or
            any(plan["forced_evaluator_sha256"][route] !=
                sha(Path(__file__).with_name(filename))
                for route, (filename, _) in FORCED.items()) or
            reserve["status"] != "sealed_no_contact_visits" or
            pointer["selected_design_id"] is not None or
            pointer["status"] != "rejected_no_selected_mechanical_design" or
            plan["contains_contact_results"] or
            plan["development_count"] != len(development) or
            plan["held_count"] != len(held) or
            len({case["seed"] for case in plan["cases"]}) !=
            len(plan["cases"]) or
            {case["seed"] for case in held} !=
            set(method["held_seed_ids"]) -
            {row["seed"] for row in plan["unselected_families"]} or
            any(case["prescribed_route"] !=
                ("v16_specialist_locked" if case["automatic_route_at_screen"] ==
                 "v21_scoped_y_locked" else "v21_scoped_y_locked")
                for case in plan["cases"])):
        raise ValueError("V158 development contact source or held split changed")
    for case in plan["cases"]:
        report = ROOT / case["entry_report"]
        if (sha(report) != case["entry_report_sha256"] or
                sha(report.with_suffix(".npz")) !=
                case["entry_trace_sha256"] or
                sha(ROOT / case["entry_image"]) !=
                case["entry_image_sha256"]):
            raise ValueError(f"Selected RGB entry source changed: {case['id']}")
    return development


def checked_result(case: dict, result: dict, evaluator_sha: str) -> tuple[dict, dict | None]:
    report = verify_report(result["result"], result["trace"], case, 0,
                           result["contact_s"], evaluator_sha)
    verify_trace(result["trace"], report)
    try:
        observed = score_runner.observations.observed_features(report)
    except ValueError:
        observed = None
    return report, observed


def entry_state_gap(automatic_report: dict, prescribed_report: dict,
                    automatic_trace: str, prescribed_trace: str
                    ) -> tuple[float, float, list[str]]:
    """Compare saved pre-reach native states before contact routes diverge."""
    def entry_index(report: dict) -> int:
        times = [row["time_s"] for row in report["timeseries"]]
        index = min(range(len(times)),
                    key=lambda index: abs(times[index] - report["reach_start_s"]))
        if abs(times[index] - report["reach_start_s"]) > .011:
            raise ValueError("No saved native reach-entry state")
        return index

    first, second = entry_index(automatic_report), entry_index(prescribed_report)
    with np.load(ROOT / automatic_trace, allow_pickle=False) as before, \
            np.load(ROOT / prescribed_trace, allow_pickle=False) as after:
        a = before["qpos"][first].astype(np.float64)
        b = after["qpos"][second].astype(np.float64)
    if a.shape != b.shape:
        raise ValueError("Paired native reach-entry state dimensions differ")
    max_gap = float(np.max(np.abs(a - b)))
    root_gap = float(np.linalg.norm(a[:3] - b[:3]))
    faults = ([f"saved native reach-entry qpos gap {max_gap:.6g} exceeds "
               f"{ENTRY_QPOS_MAX_ABS_TOLERANCE:.6g} mixed m/rad tolerance"]
              if max_gap > ENTRY_QPOS_MAX_ABS_TOLERANCE else [])
    return max_gap, root_gap, faults


def run_pair(case: dict, plan: dict) -> dict:
    source = plan["native_source_sha256"]
    automatic = native.run_case(
        {**case, "id": case["id"] + "-automatic"},
        {"source_sha256": source}, DEST)
    base_report, base_observed = checked_result(case, automatic,
                                                source["evaluator"])
    if (base_report["body_sha256"] != source["body"] or
            base_report["selected_pointer_sha256"] != source["pointer"] or
            base_report["reach_checkpoint_sha256"] != source["actor"]):
        raise ValueError(f"Automatic native source changed: {case['id']}")
    screen_rgb_faults = score_runner.entry_parity(
        case["entry_observed_features"], base_observed)
    screen_report = json.loads((ROOT / case["entry_report"]).read_text())
    screen_gap, screen_root_gap, screen_physical_faults = entry_state_gap(
        screen_report, base_report,
        str(Path(case["entry_report"]).with_suffix(".npz")),
        automatic["trace"])
    base = {
        "id": case["id"], "seed": case["seed"],
        "family_id": case["family_id"], "stratum": case["stratum"],
        "cohort": case["cohort"], "split": "development",
        "source_v156_entry_id": case["source_v156_entry_id"],
        "target_route_absent": case["target_route_absent"],
        "screen_route": case["automatic_route_at_screen"],
        "automatic_route": automatic["selection_mode"],
        "prescribed_route": case["prescribed_route"],
        "screen_entry_observed_features": case["entry_observed_features"],
        "automatic_entry_observed_features": base_observed,
        "screen_to_full_rgb_faults": screen_rgb_faults,
        "screen_to_full_physical_faults": screen_physical_faults,
        "screen_to_full_entry_qpos_max_abs_gap": screen_gap,
        "screen_to_full_entry_root_gap_m": screen_root_gap,
        "automatic_contact_s": automatic["contact_s"],
        "automatic_native_pass": automatic["pass"],
        "automatic_margin_1p2_s_met": automatic["margin_1p2_s_met"],
        "automatic_faults": automatic["faults"],
        "automatic_report": automatic["result"],
        "automatic_trace": automatic["trace"],
        "automatic_report_sha256": sha(ROOT / automatic["result"]),
        "automatic_trace_sha256": sha(ROOT / automatic["trace"]),
    }
    if screen_rgb_faults or screen_physical_faults:
        return {**base, "status": "invalid_screen_to_full_entry_parity",
                "matched_pair_valid": False,
                "prescribed_rgb_faults": [],
                "physical_start_faults": [],
                "paired_entry_qpos_max_abs_gap": None,
                "paired_entry_root_gap_m": None,
                "failure_reasons": screen_rgb_faults + screen_physical_faults,
                "prescribed_report": None, "prescribed_trace": None,
                "prescribed_contact_s": None, "prescribed_native_pass": None,
                "prescribed_margin_1p2_s_met": None,
                "prescribed_faults": []}
    filename, flag = FORCED[case["prescribed_route"]]
    forced_evaluator = Path(__file__).with_name(filename)
    original_evaluator = native.EVALUATOR
    native.EVALUATOR = forced_evaluator
    try:
        prescribed = native.run_case(
            {**case, "id": case["id"] + "-prescribed"},
            {"source_sha256": {**source,
                               "evaluator": plan["forced_evaluator_sha256"]
                               [case["prescribed_route"]]}}, DEST)
    finally:
        native.EVALUATOR = original_evaluator
    forced_report, forced_observed = checked_result(
        case, prescribed, plan["forced_evaluator_sha256"][case["prescribed_route"]])
    if (not forced_report.get(flag) or
            forced_report["selection_mode"] != case["prescribed_route"] or
            forced_report["body_sha256"] != source["body"] or
            forced_report["selected_pointer_sha256"] != source["pointer"] or
            forced_report["reach_checkpoint_sha256"] != source["actor"]):
        raise ValueError(f"Prescribed native source changed: {case['id']}")
    paired_observed = (None if forced_observed is None else
                       {**forced_observed,
                        "selection_mode": base_observed["selection_mode"],
                        "route_goal_source": base_observed["route_goal_source"]})
    paired_faults = score_runner.entry_parity(base_observed, paired_observed)
    max_gap, root_gap, physical_faults = entry_state_gap(
        base_report, forced_report, automatic["trace"], prescribed["trace"])
    valid = not paired_faults and not physical_faults
    return {**base,
            "status": ("valid_matched_native_pair" if valid else
                       "invalid_automatic_to_prescribed_entry_parity"),
            "matched_pair_valid": valid,
            "prescribed_rgb_faults": paired_faults,
            "physical_start_faults": physical_faults,
            "paired_entry_qpos_max_abs_gap": max_gap,
            "paired_entry_root_gap_m": root_gap,
            "failure_reasons": paired_faults + physical_faults,
            "prescribed_entry_observed_features": forced_observed,
            "prescribed_contact_s": prescribed["contact_s"],
            "prescribed_native_pass": prescribed["pass"],
            "prescribed_margin_1p2_s_met": prescribed["margin_1p2_s_met"],
            "prescribed_faults": prescribed["faults"],
            "prescribed_report": prescribed["result"],
            "prescribed_trace": prescribed["trace"],
            "prescribed_report_sha256": sha(ROOT / prescribed["result"]),
            "prescribed_trace_sha256": sha(ROOT / prescribed["trace"]),
            }


def write_matrix(plan: dict, cases: list[dict], rows: dict[str, dict]) -> dict:
    ordered = [rows[case["id"]] for case in cases if case["id"] in rows]
    valid = [row for row in ordered if row["matched_pair_valid"]]
    result = {
        "schema": "robot-learning-lab.cove50-v158-new-seed-development-contact.v1",
        "role": "paired_new_seed_development_contact_not_held_or_independent_score",
        "body_role": plan["body_role"], "split": "development",
        "plan_sha256": sha(PLAN),
        "case_count": len(ordered), "planned_count": len(cases),
        "complete": len(ordered) == len(cases),
        "valid_matched_pair_count": len(valid),
        "invalid_entry_parity_count": len(ordered) - len(valid),
        "invalid_rgb_parity_count": sum(bool(row["screen_to_full_rgb_faults"] or
                                             row["prescribed_rgb_faults"])
                                        for row in ordered),
        "invalid_physical_start_count": sum(bool(
            row["screen_to_full_physical_faults"] or row["physical_start_faults"])
                                            for row in ordered),
        "automatic_full_gate_failure_count": sum(
            not row["automatic_native_pass"] for row in valid),
        "prescribed_full_gate_rescue_count": sum(
            not row["automatic_native_pass"] and row["prescribed_native_pass"]
            for row in valid),
        "prescribed_full_gate_regression_count": sum(
            row["automatic_native_pass"] and not row["prescribed_native_pass"]
            for row in valid),
        "held_contact_visit_count": 0,
        "cases": ordered,
    }
    write_json(MATRIX, result)
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--max-new", type=int, default=None,
                        help="Bound this resumed development collection, not the plan")
    args = parser.parse_args()
    if args.max_new is not None and args.max_new < 1:
        raise ValueError("--max-new must be positive")
    if not PLAN.is_file():
        raise ValueError("Freeze V158 contact plan after complete V156/V157 before contact")
    plan = json.loads(PLAN.read_text())
    cases = validate_plan(plan)
    DEST.mkdir(parents=True, exist_ok=True)
    old = json.loads(MATRIX.read_text()) if MATRIX.exists() else None
    if old and (old["plan_sha256"] != sha(PLAN) or
                old["split"] != "development" or
                old["held_contact_visit_count"]):
        raise ValueError("Existing V158 matrix belongs to another plan or split")
    rows = {row["id"]: row for row in old["cases"]} if old else {}
    if old and (old["case_count"] != len(rows) or
                not set(rows) <= {case["id"] for case in cases}):
        raise ValueError("Existing V158 development case membership changed")
    for row in rows.values():
        if (row["split"] != "development" or
                sha(ROOT / row["automatic_report"]) !=
                row["automatic_report_sha256"] or
                sha(ROOT / row["automatic_trace"]) !=
                row["automatic_trace_sha256"] or
                (row["prescribed_report"] is not None and
                 (sha(ROOT / row["prescribed_report"]) !=
                  row["prescribed_report_sha256"] or
                  sha(ROOT / row["prescribed_trace"]) !=
                  row["prescribed_trace_sha256"]))):
            raise ValueError(f"Existing V158 native source changed: {row['id']}")
    remaining = [case for case in cases if case["id"] not in rows]
    for case in remaining[:args.max_new]:
        row = run_pair(case, plan)
        rows[row["id"]] = row
        matrix = write_matrix(plan, cases, rows)
        print(f"V158 {case['id']}: {row['status']} "
              f"{matrix['case_count']}/{len(cases)} development starts; "
              f"{matrix['valid_matched_pair_count']} valid pairs", flush=True)


if __name__ == "__main__":
    main()
