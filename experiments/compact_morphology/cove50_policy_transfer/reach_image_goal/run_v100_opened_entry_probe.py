"""Probe RGB route exposure by varying gaze and toy pose on opened V98 seeds."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path

from experiments.compact_morphology.cove50_policy_transfer.reach_image_goal import (
    run_v98_entry_exposure as v98,
)


ROOT = Path(__file__).resolve().parents[4]
OUT = ROOT / "artifacts/ape-cove50-reach-goal-channels-20261010/v100"
PLAN = OUT / "opened-entry-probe-plan.json"
DEST = OUT / "native-opened-entry-probe"
MATRIX = DEST / "matrix.json"
V98_MATRIX = v98.DEST / "matrix.json"


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def freeze() -> tuple[dict, dict]:
    v98_plan = v98.freeze(persist=False)
    opened = json.loads(V98_MATRIX.read_text())
    by_seed = {row["seed"]: row for row in opened["cases"]}
    base_by_seed = {case["seed"]: case for case in v98_plan["cases"]}
    hypotheses = (
        # Seed 18023's original camera Y is -106 mm; lower toy X/Y aims
        # to bring its measured 3.0/8.7 mm goal error into the route window.
        (18023, .0049, .0023),
        # Seed 18018's original goal Y is near the window, but its orange
        # count is low; gaze variation tests visibility and camera X together.
        (18018, .0047, .0066),
    )
    cases = []
    opened_sources = {}
    for seed, toy_x, toy_y in hypotheses:
        row = by_seed.get(seed)
        if not row or not row["entry_available"]:
            raise ValueError(f"V100 seed {seed} has no opened V98 RGB entry")
        report_path = ROOT / row["result"]
        opened_sources[str(seed)] = {"result": row["result"], "sha256": sha(report_path)}
        for gaze in (-1.3, -1.5):
            cases.append({**base_by_seed[seed],
                          "id": f"opened-{seed}-gaze-{str(gaze).replace('.', 'p').replace('-', 'm')}",
                          "toy_offset_x_m": toy_x,
                          "toy_offset_y_m": toy_y,
                          "gaze_head_yaw_rad": gaze})
    plan = {
        "schema": "robot-learning-lab.cove50-v100-opened-entry-probe-plan.v1",
        "role": "opened_seed_camera_only_route_exposure_diagnostic",
        "body_role": "rejected_historical_direct_frame_sandbox",
        "source_sha256": {**v98_plan["source_sha256"],
                          "probe_runner": sha(Path(__file__)),
                          "v98_plan": sha(v98.PLAN)},
        "opened_v98_source_reports": opened_sources,
        "case_count": len(cases), "cases": cases,
        "hypothesis": ("Move the toy by millimeters and vary attained physical gaze on "
                       "two already opened starts to see whether actual first-frame "
                       "image-goal and camera predicates overlap. Each visit ends "
                       "before contact; this is route-exposure diagnosis, not a reach score."),
    }
    OUT.mkdir(parents=True, exist_ok=True)
    encoded = (json.dumps(plan, indent=2) + "\n").encode()
    if PLAN.exists() and PLAN.read_bytes() != encoded:
        raise ValueError("Frozen V100 probe plan or sources changed")
    PLAN.write_bytes(encoded)
    return plan, v98_plan


def predicate_row(row: dict) -> dict:
    report = json.loads((ROOT / row["result"]).read_text())
    entries = [event for event in report["image_goal_capture_events"]
               if event["used_for_control"]]
    if len(entries) != 1:
        raise ValueError(f"V100 expected one used-for-control entry: {row['id']}")
    event = entries[0]
    goal_x, goal_y = report["selection_features"]["entry_error_xy_m"]
    camera_x, camera_y = event["camera_world_xyz_m"][:2]
    limits = report["alternate_gaze_specialist_observable_thresholds"]
    predicates = {
        "goal_x": limits["entry_error_x_range_m"][0] <= goal_x < limits["entry_error_x_range_m"][1],
        "goal_y": limits["entry_error_y_range_m"][0] <= goal_y < limits["entry_error_y_range_m"][1],
        "camera_x": limits["camera_world_x_range_m"][0] <= camera_x < limits["camera_world_x_range_m"][1],
        "camera_y": camera_y < limits["camera_world_y_max_m"],
        "toy_pixels": event["orange_pixels"] >= limits["entry_orange_pixel_min"],
    }
    missing = [name for name, met in predicates.items() if not met]
    if row["new_specialist_applied"] and missing:
        raise ValueError(f"Specialist activated without its visible predicates: {row['id']}")
    return {**row,
            "entry_error_xy_mm": [round(goal_x * 1000, 3), round(goal_y * 1000, 3)],
            "camera_xy_mm": [round(camera_x * 1000, 3), round(camera_y * 1000, 3)],
            "orange_pixels": event["orange_pixels"],
            "predicates": predicates,
            "all_predicates_met": not missing,
            "route_blockers": missing if missing else
                              ([] if row["new_specialist_applied"] else ["earlier route selected"])}


def write_matrix(plan: dict, rows: dict[str, dict]) -> None:
    ordered = [rows[case["id"]] for case in plan["cases"] if case["id"] in rows]
    matrix = {"schema": "robot-learning-lab.cove50-v100-opened-entry-probe.v1",
              "role": plan["role"], "body_role": plan["body_role"],
              "plan_sha256": sha(PLAN), "complete": len(ordered) == len(plan["cases"]),
              "case_count": len(ordered),
              "new_specialist_count": sum(row["new_specialist_applied"] for row in ordered),
              "all_predicates_met_count": sum(row["all_predicates_met"] for row in ordered),
              "cases": ordered}
    temp = MATRIX.with_suffix(".json.tmp")
    temp.write_text(json.dumps(matrix, indent=2) + "\n")
    os.replace(temp, MATRIX)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--freeze-only", action="store_true")
    args = parser.parse_args()
    plan, v98_plan = freeze()
    if args.freeze_only:
        print(f"frozen {PLAN} sha256={sha(PLAN)}")
        return
    DEST.mkdir(parents=True, exist_ok=True)
    v98.DEST = DEST
    rows = {}
    for case in plan["cases"]:
        row = predicate_row(v98.run_case(case, v98_plan))
        rows[case["id"]] = row
        write_matrix(plan, rows)
        print(json.dumps({key: row[key] for key in
                          ("id", "entry_error_xy_mm", "camera_xy_mm", "orange_pixels",
                           "route_blockers", "new_specialist_applied")}), flush=True)


if __name__ == "__main__":
    main()
