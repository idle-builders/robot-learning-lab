"""Source-check the new-seed RGB entry screen before selecting contact training."""

from __future__ import annotations

import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path

import numpy as np

from experiments.compact_morphology.cove50_policy_transfer.reach_image_goal import (
    v118_observable_selector as observations,
)


ROOT = Path(__file__).resolve().parents[4]
BASE = ROOT / "artifacts/ape-cove50-reach-goal-channels-20261010"
PLAN = BASE / "v149/new-seed-route-exposure-plan.json"
MATRIX = BASE / "v149/native-entry-only/matrix.json"
RESERVE = BASE / "v147/untouched-score-reserve-plan.json"
OUT = BASE / "v149/entry-exposure-summary.json"
INSPECTOR = BASE / "v149/entry-exposure-inspector-matrix.json"
PARTIAL_OUT = BASE / "v149/partial-entry-exposure-summary.json"
PARTIAL_INSPECTOR = BASE / "v149/partial-entry-inspector-matrix.json"
RUNNER = Path(__file__).with_name("run_v149_route_exposure_screen.py")
PHYSICAL_FIELDS = (
    "seed", "rail_v", "delay_ms", "toy_offset_x_m", "toy_offset_y_m",
    "pitch_offset_rad", "roll_clearance_rad", "gaze_head_yaw_rad",
)


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--verify-partial", action="store_true")
    args = parser.parse_args()
    plan = json.loads(PLAN.read_text())
    matrix = json.loads(MATRIX.read_text())
    reserve = json.loads(RESERVE.read_text())
    planned = plan["cases"]
    rows = matrix["cases"]
    by_id = {row["id"]: row for row in rows}
    included = [(case, by_id[case["id"]]) for case in planned
                if case["id"] in by_id]
    if ((not args.verify_partial and not matrix["complete"]) or
            len(rows) > 96 or matrix["case_count"] != len(rows) or
            matrix["complete"] != (len(rows) == 96) or
            matrix["planned_count"] != 96 or
            matrix["plan_sha256"] != sha(PLAN) or
            matrix["contains_contact_outcomes"] is not False or
            reserve["status"] != "sealed_no_contact_visits" or
            plan["source_sha256"]["v147_unopened_score_reserve"] != sha(RESERVE) or
            plan["source_sha256"]["runner"] != sha(RUNNER) or
            len(by_id) != len(rows) or
            [row["id"] for row in rows] != [case["id"] for case, _ in included] or
            {row["seed"] for row in rows} &
            {case["seed"] for case in reserve["cases"]}):
        raise ValueError("Source-matched entry-only matrix and sealed reserve required")
    by_cohort: dict[str, list[tuple[dict, dict]]] = {}
    all_routes = Counter()
    entry_images: dict[str, dict | None] = {}
    for case, row in included:
        if (row["seed"] != case["seed"] or row["cohort"] != case["cohort"] or
                row["route_variant"] != case["route_variant"]):
            raise ValueError(f"Physical start changed: {case['id']}")
        report_path = ROOT / row["result"]
        trace_path = report_path.with_suffix(".npz")
        if (sha(report_path) != row["result_sha256"] or
                sha(trace_path) != row["trace_sha256"]):
            raise ValueError(f"Saved entry source changed: {case['id']}")
        report = json.loads(report_path.read_text())
        source = plan["source_sha256"]
        if (report["preflight_only"] is not True or
                report["reach_duration_s"] != .02 or
                report["evaluator_sha256"] != source["evaluator"] or
                report["selected_pointer_sha256"] != source["pointer"] or
                report["body_sha256"] != source["body"] or
                report["reach_checkpoint_sha256"] != source["actor"] or
                report["seed"] != case["seed"] or
                report["selection_mode"] != row["selection_mode"]):
            raise ValueError(f"Entry evaluator or body differs: {case['id']}")
        for field in PHYSICAL_FIELDS:
            report_field = "toy_offset_b_m" if field.startswith("toy_offset_") else field
            if field.startswith("toy_offset_"):
                index = 0 if field == "toy_offset_x_m" else 1
                actual = report[report_field][index]
            else:
                actual = report[report_field]
            if abs(actual - case[field]) > 1e-8:
                raise ValueError(f"Report physical start differs: {case['id']} {field}")
        with np.load(trace_path, allow_pickle=False) as trace:
            count = len(trace["qpos"])
            if (trace["qpos"].shape != (count, 35) or
                    trace["action"].shape != (count, 21) or
                    trace["phase"].shape != (count,) or
                    abs(float(trace["step_dt_s"]) - .02) > 1e-9 or
                    not np.isfinite(trace["qpos"]).all() or
                    not np.isfinite(trace["action"]).all()):
                raise ValueError(f"Entry trace layout differs: {case['id']}")
        features = observations.observed_features(report) if row["entry_available"] else None
        if features != row["observed_features"]:
            raise ValueError(f"Observed RGB entry differs: {case['id']}")
        image_path = report_path.with_name(report_path.stem + "-entry.png")
        if row["entry_available"] and not image_path.is_file():
            raise ValueError(f"Used RGB entry frame is missing: {case['id']}")
        entry_images[case["id"]] = (
            {"path": str(image_path.relative_to(ROOT)), "sha256": sha(image_path)}
            if row["entry_available"] else None)
        by_cohort.setdefault(case["cohort"], []).append((case, row))
        all_routes[row["selection_mode"]] += 1
    if (matrix["entry_available_count"] !=
            sum(row["entry_available"] for row in rows) or
            matrix["observed_route_counts"] != dict(sorted(all_routes.items()))):
        raise ValueError("V149 matrix aggregate differs from source-checked entries")
    cohort_summary = {}
    for cohort, group in sorted(by_cohort.items()):
        intended = ("v16_specialist_locked" if "pitch_low" in cohort else
                    "v21_scoped_y_locked")
        exposed = [(case, row) for case, row in group
                   if row["selection_mode"] == intended and row["entry_available"]]
        cohort_summary[cohort] = {
            "entry_count": len(group),
            "entry_available_count": sum(row["entry_available"]
                                         for _, row in group),
            "observed_route_counts": dict(sorted(Counter(
                row["selection_mode"] for _, row in group).items())),
            "opened_failure_route": intended,
            "opened_failure_route_exposure_count": len(exposed),
            "exposed_seed_ids_audit_only": [case["seed"] for case, _ in exposed],
            "exposed_motor_voltage_values": sorted({case["rail_v"]
                                                    for case, _ in exposed}),
            "exposed_delay_values_ms": sorted({case["delay_ms"]
                                               for case, _ in exposed}),
            "exposed_gaze_values_rad": sorted({case["gaze_head_yaw_rad"]
                                               for case, _ in exposed}),
        }
    inspector = {
        "schema": "robot-learning-lab.cove50-v149-entry-inspector-matrix.v1",
        "role": "source_checked_entry_only_training_pool_no_contact_score",
        "body_role": plan["body_role"],
        "plan_sha256": sha(PLAN), "source_matrix_sha256": sha(MATRIX),
        "case_count": len(rows),
        "cases": [{**case, **row, "source_checked": True,
                   "entry_image": (entry_images[case["id"]]["path"]
                                   if entry_images[case["id"]] else None),
                   "entry_image_sha256": (entry_images[case["id"]]["sha256"]
                                          if entry_images[case["id"]] else None),
                   "opened_failure_route": (
                       "v16_specialist_locked" if "pitch_low" in case["cohort"]
                       else "v21_scoped_y_locked"),
                   "matches_opened_failure_route": row["selection_mode"] == (
                       "v16_specialist_locked" if "pitch_low" in case["cohort"]
                       else "v21_scoped_y_locked")}
                  for case, row in included],
    }
    inspector_path = PARTIAL_INSPECTOR if args.verify_partial else INSPECTOR
    inspector_encoded = json.dumps(inspector, indent=2) + "\n"
    if not args.verify_partial and INSPECTOR.exists() and INSPECTOR.read_text() != inspector_encoded:
        raise ValueError("Saved V149 inspector entry matrix changed")
    inspector_path.write_text(inspector_encoded)
    summary = {
        "schema": "robot-learning-lab.cove50-v149-source-checked-entry-summary.v1",
        "role": "entry_only_training_pool_no_contact_or_score",
        "body_role": plan["body_role"],
        "plan_sha256": sha(PLAN), "matrix_sha256": sha(MATRIX),
        "reserve_sha256": sha(RESERVE),
        "inspector_matrix": str(inspector_path.relative_to(ROOT)),
        "inspector_matrix_sha256": sha(inspector_path),
        "source_checked_entry_count": len(rows),
        "complete": matrix["complete"],
        "observed_route_counts": dict(sorted(all_routes.items())),
        "cohorts": cohort_summary,
        "interpretation": ("Only first-control-frame RGB entries were simulated. "
                           "Source checks cover each report, 50 Hz trace, "
                           "physical start, actor, evaluator and observed route. "
                           "Route exposure can guide a separately frozen "
                           "contact-training plan; it is neither contact "
                           "performance nor an independent policy score."),
    }
    output_path = PARTIAL_OUT if args.verify_partial else OUT
    encoded = json.dumps(summary, indent=2) + "\n"
    if not args.verify_partial and OUT.exists() and OUT.read_text() != encoded:
        raise ValueError("Saved V149 source-checked summary changed")
    output_path.write_text(encoded)
    print(f"V149 source-checked {len(rows)} entry-only visits; "
          f"routes={summary['observed_route_counts']}")


if __name__ == "__main__":
    main()
