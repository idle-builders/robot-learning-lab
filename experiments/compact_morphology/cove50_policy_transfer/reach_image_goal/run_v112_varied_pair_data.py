"""Collect paired native contact data across varied observed RGB entries."""

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
OUT = BASE / "v112"
PLAN = OUT / "varied-paired-contact-data-plan.json"
DEST = OUT / "native-varied-paired-contact-data"
V105_PLAN = BASE / "v105/varied-entry-exposure-plan.json"
V105_MATRIX = BASE / "v105/native-varied-entry-exposure/matrix.json"
COUNT = 24
SHIFT_M = -.002
ALREADY_OPENED_FULL = {19011, 19020}
VARIANTS = ("center", "pitch-low", "pitch-high", "roll-low", "roll-high")


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def select_rows(rows: list[dict]) -> list[dict]:
    eligible = [row for row in rows if row["seed"] not in ALREADY_OPENED_FULL]
    chosen: list[dict] = []
    seen: set[str] = set()

    def add(row: dict) -> None:
        if row["id"] not in seen and len(chosen) < COUNT:
            chosen.append(row)
            seen.add(row["id"])

    for row in sorted((row for row in eligible if row["natural_specialist"]),
                      key=lambda row: row["seed"]):
        add(row)
    for row in sorted((row for row in eligible
                       if -50 <= row["camera_xy_mm"][0] < 0 and
                       row["camera_xy_mm"][1] < -80),
                      key=lambda row: row["seed"]):
        add(row)
    for row in sorted((row for row in eligible if row["orange_pixels"] < 40),
                      key=lambda row: (row["orange_pixels"], row["seed"]))[:6]:
        add(row)
    while len(chosen) < COUNT:
        before = len(chosen)
        for variant in VARIANTS:
            candidate = next((row for row in eligible
                              if row["id"] not in seen and
                              row["route_variant"] == variant), None)
            if candidate:
                add(candidate)
        if len(chosen) == before:
            raise ValueError("Could not fill varied RGB-entry development selection")
    if len({row["seed"] for row in chosen}) != COUNT or not all(
            any(row["route_variant"] == variant for row in chosen)
            for variant in VARIANTS):
        raise ValueError("Varied paired selection lacks distinct seeds or route coverage")
    return chosen


def freeze() -> dict:
    source_plan = json.loads(V105_PLAN.read_text())
    source_matrix = json.loads(V105_MATRIX.read_text())
    if (not source_matrix["complete"] or source_matrix["case_count"] != 64 or
            source_matrix["entry_available_count"] != 64 or
            source_matrix["plan_sha256"] != sha(V105_PLAN)):
        raise ValueError("V112 waits for complete source-matched V105 RGB entries")
    pointer = json.loads((ROOT / "robot/ape-current-design.json").read_text())
    if pointer["status"] != "rejected_no_selected_mechanical_design":
        raise ValueError("V112 is only a rejected-body development data collection")
    sources = {**source_plan["source_sha256"],
               "evaluator": sha(baseline.EVALUATOR),
               "shift_evaluator": sha(shifted.EVALUATOR),
               "baseline_runner": sha(Path(baseline.__file__)),
               "shift_runner": sha(Path(shifted.__file__)),
               "runner": sha(Path(__file__))}
    original = {case["id"]: case for case in source_plan["cases"]}
    cases = []
    refs = []
    for row in select_rows(source_matrix["cases"]):
        report_path = ROOT / row["result"]
        report = json.loads(report_path.read_text())
        if (report["body_sha256"] != sources["body"] or
                report["selected_pointer_sha256"] != sources["pointer"] or
                report["selection_mode"] != row["selection_mode"]):
            raise ValueError(f"RGB entry source mismatch: {row['id']}")
        entry = original[row["id"]]
        refs.append({"id": row["id"], "path": row["result"],
                     "sha256": sha(report_path)})
        cases.append({**entry, "id": f"paired-{row['seed']}",
                      "source_entry_id": row["id"],
                      "expected_selection_mode": row["selection_mode"],
                      "source_entry_orange_pixels": row["orange_pixels"],
                      "source_entry_camera_xy_mm": row["camera_xy_mm"]})
    coverage = {
        "route_variants": {name: sum(c["route_variant"] == name for c in cases)
                           for name in VARIANTS},
        "motor_v": {str(v): sum(c["rail_v"] == v for c in cases)
                    for v in (10, 11, 12)},
        "delay_ms": {str(v): sum(c["delay_ms"] == v for c in cases)
                     for v in (0, 5, 10)},
        "gaze_rad": {str(v): sum(c["gaze_head_yaw_rad"] == v for c in cases)
                     for v in (-1.3, -1.5)},
        "natural_specialist": sum(row["natural_specialist"]
                                  for row in select_rows(source_matrix["cases"])),
    }
    plan = {
        "schema": "robot-learning-lab.cove50-v112-varied-paired-contact-plan.v1",
        "role": "opened_varied_paired_contact_training_data_not_score",
        "body_role": "rejected_historical_direct_frame_sandbox",
        "source_sha256": sources,
        "v105_plan_sha256": sha(V105_PLAN),
        "v105_matrix_sha256": sha(V105_MATRIX),
        "v105_entry_reports": refs,
        "case_count": COUNT,
        "visit_count": 2 * COUNT,
        "diagnostic_goal_shift_body_y_m": SHIFT_M,
        "selection": ("Before any selected full-native contact visit, take all "
                      "V105 natural specialists except the two already tested "
                      "in V109, then the observed rear camera-region entries, "
                      "six lowest-pixel visible entries and round-robin cases "
                      "from all five route commands until 24 unique entries. "
                      "Selection uses only completed RGB entry features, never "
                      "contact labels or simulator truth at inference."),
        "coverage": coverage,
        "cases": cases,
        "method": ("Replay each exact V105 physical start at zero and −2 mm "
                   "post-route body-Y target offsets with unchanged automatic "
                   "observed-RGB routing. Retain both complete native reports, "
                   "traces, RGB/contact/support/rated-effort/action/collision "
                   "gates and contact deltas. Failures are training evidence, "
                   "not a reason to drop a pair."),
        "decision_gate": ("The complete 24-pair matrix supports fitting an "
                          "observable selective correction alongside V101, V108, "
                          "V109, V110 and V111 evidence. Native held-out "
                          "evaluation on distinct still-untouched starts is "
                          "required before any policy promotion. This does not "
                          "change the failed V101 score or qualify a body."),
    }
    OUT.mkdir(parents=True, exist_ok=True)
    encoded = (json.dumps(plan, indent=2) + "\n").encode()
    if PLAN.exists() and PLAN.read_bytes() != encoded:
        raise ValueError("Frozen V112 source or plan changed")
    PLAN.write_bytes(encoded)
    return plan


def write_matrix(plan: dict, rows: dict[str, dict]) -> dict:
    ordered = [rows[case["id"]] for case in plan["cases"] if case["id"] in rows]
    matrix = {
        "schema": "robot-learning-lab.cove50-v112-varied-paired-contact-data.v1",
        "role": plan["role"], "body_role": plan["body_role"],
        "plan_sha256": sha(PLAN), "complete": len(ordered) == COUNT,
        "case_count": len(ordered), "visit_count": 2 * len(ordered),
        "baseline_pass_count": sum(row["baseline_pass"] for row in ordered),
        "shifted_pass_count": sum(row["shifted_pass"] for row in ordered),
        "rescue_count": sum(not row["baseline_pass"] and row["shifted_pass"]
                            for row in ordered),
        "regression_count": sum(row["baseline_pass"] and not row["shifted_pass"]
                                for row in ordered),
        "shifted_margin_count": sum(row["shifted_margin_1p2_s_met"]
                                    for row in ordered),
        "cases": ordered,
    }
    temporary = DEST / "matrix.json.tmp"
    temporary.write_text(json.dumps(matrix, indent=2) + "\n")
    os.replace(temporary, DEST / "matrix.json")
    return matrix


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--freeze-only", action="store_true")
    parser.add_argument("--workers", type=int, default=2)
    args = parser.parse_args()
    if args.workers not in (1, 2):
        raise ValueError("Use one or two paired native evaluators")
    plan = freeze()
    print(f"frozen {PLAN} sha256={sha(PLAN)}", flush=True)
    if args.freeze_only:
        return
    DEST.mkdir(parents=True, exist_ok=True)
    old = DEST / "matrix.json"
    existing = json.loads(old.read_text()) if old.exists() else None
    if existing and existing["plan_sha256"] != sha(PLAN):
        raise ValueError("Existing V112 matrix belongs to another plan")
    rows = {row["id"]: row for row in existing["cases"]} if existing else {}
    source_hashes = plan["source_sha256"]
    baseline_plan = {"source_sha256": {**source_hashes,
                                       "evaluator": source_hashes["evaluator"]}}
    shifted_plan = {"source_sha256": {**source_hashes,
                                      "evaluator": source_hashes["shift_evaluator"]}}

    def run_pair(case: dict) -> dict:
        base_case = {**case, "id": case["id"] + "-zero"}
        shift_case = {**case, "id": case["id"] + "-neg2",
                      "diagnostic_goal_shift_body_y_m": SHIFT_M}
        base = baseline.run_case(base_case, baseline_plan, DEST)
        shift = shifted.run_case(shift_case, shifted_plan, DEST)
        if (base["selection_mode"] != case["expected_selection_mode"] or
                shift["selection_mode"] != base["selection_mode"]):
            raise ValueError(f"Observed route drift on paired start {case['id']}")
        return {
            "id": case["id"], "seed": case["seed"],
            "route_variant": case["route_variant"],
            "rail_v": case["rail_v"], "delay_ms": case["delay_ms"],
            "gaze_head_yaw_rad": case["gaze_head_yaw_rad"],
            "toy_offset_x_m": case["toy_offset_x_m"],
            "toy_offset_y_m": case["toy_offset_y_m"],
            "selection_mode": base["selection_mode"],
            "source_entry_camera_xy_mm": case["source_entry_camera_xy_mm"],
            "source_entry_orange_pixels": case["source_entry_orange_pixels"],
            "baseline_contact_s": base["contact_s"],
            "shifted_contact_s": shift["contact_s"],
            "contact_delta_s": round(shift["contact_s"] - base["contact_s"], 3),
            "baseline_pass": base["pass"], "shifted_pass": shift["pass"],
            "baseline_margin_1p2_s_met": base["margin_1p2_s_met"],
            "shifted_margin_1p2_s_met": shift["margin_1p2_s_met"],
            "baseline_faults": base["faults"], "shifted_faults": shift["faults"],
            "baseline_result": base["result"], "baseline_trace": base["trace"],
            "shifted_result": shift["result"], "shifted_trace": shift["trace"],
        }

    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        futures = {pool.submit(run_pair, case): case for case in plan["cases"]
                   if case["id"] not in rows}
        for future in as_completed(futures):
            row = future.result()
            rows[row["id"]] = row
            write_matrix(plan, rows)
            print(json.dumps({key: row[key] for key in
                              ("id", "baseline_contact_s", "shifted_contact_s",
                               "contact_delta_s", "baseline_pass", "shifted_pass")}),
                  flush=True)
    matrix = write_matrix(plan, rows)
    print(json.dumps({key: matrix[key] for key in
                      ("case_count", "visit_count", "baseline_pass_count",
                       "shifted_pass_count", "rescue_count", "regression_count")}),
          flush=True)


if __name__ == "__main__":
    main()
