"""Collect source-frozen V150 paired full-native contact training on CUDA."""

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
    run_v139_reserved_contact as score_runner,
)


ROOT = Path(__file__).resolve().parents[4]
BASE = ROOT / "artifacts/ape-cove50-reach-goal-channels-20261010"
PLAN = BASE / "v150/paired-contact-training-plan.json"
RESERVE = BASE / "v147/untouched-score-reserve-plan.json"
ENTRY_MATRIX = BASE / "v149/entry-exposure-inspector-matrix.json"
ENTRY_SUMMARY = BASE / "v149/entry-exposure-summary.json"
METHOD = BASE / "v150/selection-method-freeze.json"
POINTER = ROOT / "robot/ape-current-design.json"
DEST = BASE / "v150/native-paired-contact"
MATRIX = DEST / "matrix.json"


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def write_json(path: Path, value: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2) + "\n")
    os.replace(temporary, path)


def run_pair(case: dict, plan: dict) -> dict:
    source = plan["source_sha256"]
    base = baseline.run_case({**case, "id": case["id"] + "-zero"},
                             {"source_sha256": source}, DEST)
    base_report = json.loads((ROOT / base["result"]).read_text())
    observed = score_runner.observations.observed_features(base_report)
    screen_faults = score_runner.entry_parity(
        case["entry_observed_features"], observed)
    choices = {}
    for label, shift in (("neg2", -.002), ("pos2", .002)):
        result = shifted.run_case(
            {**case, "id": case["id"] + "-" + label,
             "diagnostic_goal_shift_body_y_m": shift},
            {"source_sha256": {**source, "evaluator": source["shift_evaluator"]}},
            DEST)
        report = json.loads((ROOT / result["result"]).read_text())
        faults = score_runner.entry_parity(
            observed, score_runner.observations.observed_features(report))
        if (report["seed"] != base_report["seed"] or
                report["body_sha256"] != base_report["body_sha256"] or
                report["selected_pointer_sha256"] !=
                base_report["selected_pointer_sha256"] or
                report["selection_mode"] != base_report["selection_mode"]):
            faults.append("paired native source or automatic route mismatch")
        choices[label] = {
            "shift_body_y_m": shift,
            "contact_s": result["contact_s"],
            "contact_delta_s": round(result["contact_s"] - base["contact_s"], 3),
            "native_pass": result["pass"],
            "pass": result["pass"] and not faults,
            "margin_1p2_s_met": result["margin_1p2_s_met"],
            "entry_parity_faults": faults,
            "faults": result["faults"] + faults,
            "result": result["result"], "trace": result["trace"],
        }
    return {**case, "selection_mode": base["selection_mode"],
            "full_entry_observed_features": observed,
            "screen_to_full_entry_faults": screen_faults,
            "baseline_contact_s": base["contact_s"],
            "baseline_pass": base["pass"],
            "baseline_margin_1p2_s_met": base["margin_1p2_s_met"],
            "baseline_faults": base["faults"],
            "baseline_result": base["result"],
            "baseline_trace": base["trace"],
            "choices": choices}


def write_matrix(plan: dict, rows: dict[str, dict]) -> dict:
    ordered = [rows[case["id"]] for case in plan["cases"] if case["id"] in rows]
    matrix = {
        "schema": "robot-learning-lab.cove50-v150-paired-contact-training.v1",
        "role": plan["role"], "body_role": plan["body_role"],
        "plan_sha256": sha(PLAN),
        "complete": len(ordered) == plan["case_count"],
        "case_count": len(ordered), "planned_count": plan["case_count"],
        "baseline_pass_count": sum(row["baseline_pass"] for row in ordered),
        "neg2_pass_count": sum(row["choices"]["neg2"]["pass"] for row in ordered),
        "pos2_pass_count": sum(row["choices"]["pos2"]["pass"] for row in ordered),
        "neg2_rescue_count": sum(not row["baseline_pass"] and
                                 row["choices"]["neg2"]["pass"] for row in ordered),
        "pos2_rescue_count": sum(not row["baseline_pass"] and
                                 row["choices"]["pos2"]["pass"] for row in ordered),
        "neg2_regression_count": sum(row["baseline_pass"] and
                                     not row["choices"]["neg2"]["pass"] for row in ordered),
        "pos2_regression_count": sum(row["baseline_pass"] and
                                     not row["choices"]["pos2"]["pass"] for row in ordered),
        "screen_entry_parity_failure_count": sum(bool(row["screen_to_full_entry_faults"])
                                                 for row in ordered),
        "paired_entry_parity_failure_count": sum(bool(choice["entry_parity_faults"])
                                                 for row in ordered
                                                 for choice in row["choices"].values()),
        "cases": ordered,
    }
    write_json(MATRIX, matrix)
    return matrix


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--workers", type=int, default=2)
    args = parser.parse_args()
    if args.workers not in (1, 2):
        raise ValueError("Use one or two native evaluators")
    plan = json.loads(PLAN.read_text())
    if (plan["source_sha256"]["v150_runner"] != sha(Path(__file__)) or
            plan["source_sha256"]["untouched_score_reserve"] != sha(RESERVE) or
            plan["source_sha256"]["v150_selection_method"] != sha(METHOD) or
            plan["source_sha256"]["v149_entry_matrix"] != sha(ENTRY_MATRIX) or
            plan["source_sha256"]["v149_entry_summary"] != sha(ENTRY_SUMMARY) or
            plan["source_sha256"]["pointer"] != sha(POINTER) or
            json.loads(RESERVE.read_text())["status"] != "sealed_no_contact_visits"):
        raise ValueError("V150 source or sealed reserve changed")
    for case in plan["cases"]:
        report = ROOT / case["entry_report"]
        if (sha(report) != case["entry_report_sha256"] or
                sha(report.with_suffix(".npz")) != case["entry_trace_sha256"]):
            raise ValueError(f"Selected V149 entry source changed: {case['id']}")
    DEST.mkdir(parents=True, exist_ok=True)
    old = json.loads(MATRIX.read_text()) if MATRIX.exists() else None
    if old and old["plan_sha256"] != sha(PLAN):
        raise ValueError("Existing V150 matrix belongs to another plan")
    rows = {row["id"]: row for row in old["cases"]} if old else {}
    print(f"V150 full-native training plan sha256={sha(PLAN)}; "
          "48 starts, 144 visits", flush=True)
    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        futures = {pool.submit(run_pair, case, plan): case
                   for case in plan["cases"] if case["id"] not in rows}
        for future in as_completed(futures):
            row = future.result()
            rows[row["id"]] = row
            matrix = write_matrix(plan, rows)
            print(f"V150 {row['id']}: unchanged {row['baseline_contact_s']:.2f}, "
                  f"−2 {row['choices']['neg2']['contact_s']:.2f}, "
                  f"+2 {row['choices']['pos2']['contact_s']:.2f}s; "
                  f"{matrix['case_count']}/48", flush=True)


if __name__ == "__main__":
    main()
