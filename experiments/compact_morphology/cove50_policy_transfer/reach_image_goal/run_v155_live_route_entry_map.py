"""Map RGB route exposure near opened live-route failure 24591.

Every case reuses one simulator seed and stops at the first controlled RGB
reach frame. This is physical-entry calibration, not independent training,
contact evaluation, or a selector score.
"""

from __future__ import annotations

import argparse
from collections import Counter
from concurrent.futures import ThreadPoolExecutor, as_completed
import hashlib
import json
import os
from pathlib import Path

from experiments.compact_morphology.cove50_policy_transfer.reach_image_goal import (
    run_v98_entry_exposure as entry_runner,
    run_v139_reserved_contact as score_runner,
    v118_observable_selector as observations,
)


ROOT = Path(__file__).resolve().parents[4]
BASE = ROOT / "artifacts/ape-cove50-reach-goal-channels-20261010"
ENTRY_PLAN = BASE / "v149/new-seed-route-exposure-plan.json"
ENTRY_MATRIX = BASE / "v149/entry-exposure-inspector-matrix.json"
ENTRY_SUMMARY = BASE / "v149/entry-exposure-summary.json"
V150_MATRIX = BASE / "v150/native-paired-contact/matrix.json"
V150_SUMMARY = BASE / "v150/source-checked-paired-summary.json"
V152_PLAN = BASE / "v152/paired-route-training-plan.json"
V153_PREFLIGHT = BASE / "v153/opportunity-support-preflight.json"
V154_AUDIT = BASE / "v154/remaining-entry-route-coverage.json"
RESERVE = BASE / "v147/untouched-score-reserve-plan.json"
POINTER = ROOT / "robot/ape-current-design.json"
OUT = BASE / "v155"
PLAN = OUT / "opened-live-entry-map-plan.json"
DEST = OUT / "native-entry-only"
MATRIX = DEST / "matrix.json"
AXIS_OFFSETS_M = (-.001, -.0005, 0.0, .0005, .001)
ANCHOR_SEED = 24591


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def write_json(path: Path, value: dict, *, frozen: bool = False) -> None:
    encoded = json.dumps(value, indent=2) + "\n"
    path.parent.mkdir(parents=True, exist_ok=True)
    if frozen and path.exists() and path.read_text() != encoded:
        raise ValueError(f"Frozen V155 entry plan changed: {path}")
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(encoded)
    os.replace(temporary, path)


def freeze() -> dict:
    entry_plan = json.loads(ENTRY_PLAN.read_text())
    entry_matrix = json.loads(ENTRY_MATRIX.read_text())
    entry_summary = json.loads(ENTRY_SUMMARY.read_text())
    v150 = json.loads(V150_MATRIX.read_text())
    v150_summary = json.loads(V150_SUMMARY.read_text())
    v152_plan = json.loads(V152_PLAN.read_text())
    v153 = json.loads(V153_PREFLIGHT.read_text())
    v154 = json.loads(V154_AUDIT.read_text())
    reserve = json.loads(RESERVE.read_text())
    pointer = json.loads(POINTER.read_text())
    entry_source = entry_runner.freeze(persist=False)["source_sha256"]
    prior_source = entry_plan["source_sha256"]
    source_keys = ("evaluator", "original_v95_evaluator", "body",
                   "pointer", "squat", "actor", "alternate_actor",
                   "specialist_actor", "reference", "geometry", "calibration")
    if (any(entry_source[key] != prior_source[key] for key in source_keys) or
            entry_source["runner"] != prior_source["entry_runner"] or
            entry_matrix["plan_sha256"] != sha(ENTRY_PLAN) or
            entry_matrix["case_count"] != 96 or
            entry_summary["inspector_matrix_sha256"] != sha(ENTRY_MATRIX) or
            entry_summary["source_checked_entry_count"] != 96 or
            not entry_summary["complete"] or
            v150["case_count"] != 48 or not v150["complete"] or
            v150_summary["matrix_sha256"] != sha(V150_MATRIX) or
            not v150_summary["complete"] or
            v152_plan["case_count"] != 48 or
            v153["v150_baseline_margin_miss_count"] != 3 or
            v153["frozen_v153_training_gate_mathematically_possible"] or
            v154["case_count"] != 48 or v154["live_route_entry_count"] != 0 or
            reserve["status"] != "sealed_no_contact_visits" or
            pointer["selected_design_id"] is not None or
            ANCHOR_SEED in {case["seed"] for case in reserve["cases"]}):
        raise ValueError("Opened route source, coverage, or sealed reserve changed")
    anchor = next(case for case in entry_plan["cases"]
                  if case["seed"] == ANCHOR_SEED)
    entry = next(row for row in entry_matrix["cases"]
                 if row["seed"] == ANCHOR_SEED)
    baseline = next(row for row in v150["cases"]
                    if row["seed"] == ANCHOR_SEED)
    if (not entry["source_checked"] or not entry["entry_available"] or
            entry["selection_mode"] != "v16_live" or
            entry["observed_features"]["route_goal_source"] !=
            "selected_image_goal" or
            baseline["baseline_margin_1p2_s_met"] or
            baseline["baseline_contact_s"] != 0):
        raise ValueError("Anchor is not the saved live-route contact failure")
    cases = []
    for ix, dx in enumerate(AXIS_OFFSETS_M):
        for iy, dy in enumerate(AXIS_OFFSETS_M):
            cases.append({**anchor,
                          "id": ("live-map-center-24591" if not dx and not dy
                                 else f"live-map-x{ix}-y{iy}-24591"),
                          "toy_offset_x_m": round(anchor["toy_offset_x_m"] + dx, 6),
                          "toy_offset_y_m": round(anchor["toy_offset_y_m"] + dy, 6),
                          "toy_delta_x_m": dx, "toy_delta_y_m": dy,
                          "correlation_family_audit_only": "opened-seed-24591"})
    cases.sort(key=lambda case: (case["id"] != "live-map-center-24591",
                                 case["toy_delta_x_m"], case["toy_delta_y_m"]))
    source = {**entry_source,
              "runner": sha(Path(__file__)),
              "entry_runner": entry_source["runner"],
              "v149_plan": sha(ENTRY_PLAN),
              "v149_matrix": sha(ENTRY_MATRIX),
              "v149_summary": sha(ENTRY_SUMMARY),
              "v150_matrix": sha(V150_MATRIX),
              "v150_summary": sha(V150_SUMMARY),
              "v152_plan": sha(V152_PLAN),
              "v153_support_preflight": sha(V153_PREFLIGHT),
              "v154_remaining_entry_audit": sha(V154_AUDIT),
              "untouched_score_reserve": sha(RESERVE)}
    plan = {
        "schema": "robot-learning-lab.cove50-v155-live-route-entry-map-plan.v1",
        "role": "opened_one_seed_camera_only_route_map_not_contact_or_score",
        "body_role": "rejected_historical_direct_frame_sandbox",
        "source_sha256": source,
        "case_count": len(cases),
        "correlated_anchor_seed_count": 1,
        "contains_contact_visits": False,
        "prior_visibility": (
            "Frozen after V150 complete, V151 opened forced-route outcomes, "
            "and partial V152 route contact including the 24591 source reuse "
            "were visible. No V155 contact or sealed V147 score was opened."),
        "method": (
            "On already opened failing live-route seed 24591, vary toy X and Y "
            "independently by -1, -0.5, 0, +0.5 and +1 mm at the fixed saved "
            "motor, delay, gaze, pitch and roll settings. Stop at the first "
            "used controlled RGB frame. The center repeat must match V149 "
            "native RGB and route. Map automatic route boundaries and toy "
            "visibility for choosing a later, genuinely new-seed entry pool. "
            "Treat all 25 cases as one correlated anchor family, never as "
            "independent failures or contact scores."),
        "cases": cases,
    }
    write_json(PLAN, plan, frozen=True)
    return plan


def run_case(case: dict, plan: dict) -> dict:
    row = entry_runner.run_case(case, plan)
    result = ROOT / row["result"]
    trace = result.with_suffix(".npz")
    report = json.loads(result.read_text())
    observed = observations.observed_features(report) if row["entry_available"] else None
    if row["entry_available"] and observed["selection_mode"] != row["selection_mode"]:
        raise ValueError(f"V155 observed route differs: {case['id']}")
    center_faults = []
    if case["id"] == "live-map-center-24591":
        reference = next(item for item in
                         json.loads(ENTRY_MATRIX.read_text())["cases"]
                         if item["seed"] == ANCHOR_SEED)
        if observed is None:
            center_faults = ["Center repeat has no controlled RGB entry"]
        else:
            center_faults = score_runner.entry_parity(
                reference["observed_features"], observed)
        if center_faults:
            raise ValueError(f"V155 center CUDA RGB parity failed: {center_faults}")
    return {**row, "cohort": case["cohort"],
            "toy_delta_x_m": case["toy_delta_x_m"],
            "toy_delta_y_m": case["toy_delta_y_m"],
            "correlation_family_audit_only": case["correlation_family_audit_only"],
            "observed_features": observed,
            "center_rgb_parity_faults": center_faults,
            "result_sha256": sha(result), "trace_sha256": sha(trace)}


def write_matrix(plan: dict, rows: dict[str, dict]) -> dict:
    ordered = [rows[case["id"]] for case in plan["cases"] if case["id"] in rows]
    matrix = {
        "schema": "robot-learning-lab.cove50-v155-live-route-entry-map.v1",
        "role": plan["role"], "body_role": plan["body_role"],
        "plan_sha256": sha(PLAN),
        "case_count": len(ordered), "planned_count": plan["case_count"],
        "complete": len(ordered) == plan["case_count"],
        "entry_available_count": sum(row["entry_available"] for row in ordered),
        "observed_route_counts": dict(sorted(Counter(
            row["selection_mode"] for row in ordered).items())),
        "correlated_anchor_seed_count": 1,
        "contains_contact_outcomes": False,
        "cases": ordered,
    }
    write_json(MATRIX, matrix)
    return matrix


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--freeze-only", action="store_true")
    parser.add_argument("--workers", type=int, default=2)
    args = parser.parse_args()
    if args.workers not in (1, 2):
        raise ValueError("Use one or two CUDA entry evaluators")
    plan = freeze()
    if args.freeze_only:
        print(f"V155 froze 25 correlated camera-only entries; sha256={sha(PLAN)}")
        return
    refs = plan["source_sha256"]
    if (refs["runner"] != sha(Path(__file__)) or
            refs["entry_runner"] != sha(Path(entry_runner.__file__)) or
            refs["v149_matrix"] != sha(ENTRY_MATRIX) or
            refs["v153_support_preflight"] != sha(V153_PREFLIGHT) or
            refs["v154_remaining_entry_audit"] != sha(V154_AUDIT) or
            refs["untouched_score_reserve"] != sha(RESERVE)):
        raise ValueError("Frozen V155 source changed")
    DEST.mkdir(parents=True, exist_ok=True)
    old = json.loads(MATRIX.read_text()) if MATRIX.exists() else None
    if old and old["plan_sha256"] != sha(PLAN):
        raise ValueError("V155 existing matrix belongs to another plan")
    rows = {row["id"]: row for row in old["cases"]} if old else {}
    entry_runner.DEST = DEST
    center = plan["cases"][0]
    if center["id"] not in rows:
        rows[center["id"]] = run_case(center, plan)
        write_matrix(plan, rows)
        print("V155 center repeat matches saved controlled RGB", flush=True)
    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        futures = {pool.submit(run_case, case, plan): case
                   for case in plan["cases"][1:] if case["id"] not in rows}
        for future in as_completed(futures):
            row = future.result()
            rows[row["id"]] = row
            matrix = write_matrix(plan, rows)
            print(f"V155 {row['id']}: {row['selection_mode']} "
                  f"{matrix['case_count']}/25 entries", flush=True)


if __name__ == "__main__":
    main()
