"""V65: unchanged low hits with the frozen 2-D selected-body band proxy."""

from __future__ import annotations

import json
import math
from pathlib import Path

import numpy as np

from audit_3d_clearance_layout_v53 import FRACTIONS, FLAT_MARGIN_M
from audit_near_floor_range_v54 import xyz_points
from goal_map_route import footprint_distance
from robot_geometry import RobotGeometry
from run import ROOT, Room, sha


SOURCE = Path(__file__)
OUT = ROOT / "artifacts/ape-closed-loop-cove50-v65"
COHORTS = (
    ("varied_crate", ROOT / "artifacts/ape-closed-loop-cove50-v56",
     "low-obstacle-transfer.json"),
    ("side_rear", ROOT / "artifacts/ape-closed-loop-cove50-v64",
     "low-range-side-back-audit.json"),
)


def score(rows: list[dict], key: str) -> dict:
    return {"passes": sum(row[key] for row in rows),
            "false_safe": sum(row[key] and not row["evaluator_safe_5cm"]
                              for row in rows),
            "false_block": sum(not row[key] and row["evaluator_safe_5cm"]
                               for row in rows)}


def evaluate_cohort(label: str, directory: Path, audit_name: str,
                    robot: RobotGeometry) -> dict:
    audit_path = directory / audit_name
    reference = json.loads(audit_path.read_text())
    by_case: dict[str, list[dict]] = {}
    for row in reference["rows"]:
        by_case.setdefault(row["case"], []).append(row)
    cases = []
    rows = []
    for case_id, source_rows in by_case.items():
        saved_path = directory / f"{case_id}-actor-input.npz"
        with np.load(saved_path) as saved:
            ranges = [None if np.isnan(value) else float(value)
                      for value in saved["range_4_72_beams_m"]]
            pose = saved["estimated_pose_xyyaw"].copy()
            actor_qpos = saved["actor_qpos"].copy()
        robot.set_state(actor_qpos)
        current = robot.primitives()
        angles = np.linspace(-math.pi, math.pi, 72, endpoint=False)
        low_xy = {}
        for field in (90, 360):
            half_angle = math.radians(field / 2)
            masked = [value if field == 360 or abs(float(angle)) <= half_angle
                      else None for angle, value in zip(angles, ranges)]
            low_xy[field] = xyz_points(masked, pose, .04, current)[:, :2]
        cases.append({"case": case_id,
                      "saved_actor_input_sha256": sha(saved_path),
                      "self_filtered_low_points_90": len(low_xy[90]),
                      "self_filtered_low_points_360": len(low_xy[360])})
        for original in source_rows:
            candidate = np.asarray(original["candidate_xy_m"])
            delta = candidate - pose[:2]
            low_gap = {field: min(footprint_distance(
                low_xy[field], pose[:2] + fraction * delta, pose[2])
                for fraction in FRACTIONS) for field in (90, 360)}
            baseline_3d = (original["both_head_pass"] if label == "varied_crate"
                           else original["field_pass"]["360"])
            rows.append({"case": case_id,
                         "direction_cell": original["direction_cell"],
                         "evaluator_model_clearance_m": original["evaluator_model_clearance_m"],
                         "evaluator_safe_5cm": original["evaluator_safe_5cm"],
                         "flat_11_pass": original["flat_2d_pass"],
                         "baseline_3d_360_pass": baseline_3d,
                         "flat_11_low90_pass": (original["flat_2d_pass"] and
                                                low_gap[90] >= FLAT_MARGIN_M),
                         "flat_11_low360_pass": (original["flat_2d_pass"] and
                                                 low_gap[360] >= FLAT_MARGIN_M),
                         "low_90_flat_gap_m": round(low_gap[90], 4),
                         "low_360_flat_gap_m": round(low_gap[360], 4)})
    assert len(rows) == reference["valid_moves"]
    scores = {name: score(rows, key) for name, key in (
        ("flat_11", "flat_11_pass"),
        ("baseline_3d_360", "baseline_3d_360_pass"),
        ("flat_11_low90", "flat_11_low90_pass"),
        ("flat_11_low360", "flat_11_low360_pass"))}
    assert scores["flat_11"] == reference["scores"]["flat_2d"]
    assert scores["baseline_3d_360"] == (
        reference["scores"]["both_head"] if label == "varied_crate"
        else reference["scores"]["360"])
    return {"label": label, "prior_audit_sha256": sha(audit_path),
            "cases": cases, "valid_moves": len(rows),
            "evaluator_safe_moves": sum(row["evaluator_safe_5cm"] for row in rows),
            "scores": scores,
            "prior_3d_360_false_safe_cases": [row for row in rows
                if row["baseline_3d_360_pass"] and not row["evaluator_safe_5cm"]],
            "rows": rows}


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    room = Room()
    try:
        robot = RobotGeometry(room.model)
        cohorts = {label: evaluate_cohort(label, directory, audit_name, robot)
                   for label, directory, audit_name in COHORTS}
    finally:
        room.close()
    gate = (all(cohorts[label]["scores"]["flat_11_low360"]["false_safe"] == 0
                for label in cohorts) and
            cohorts["varied_crate"]["scores"]["flat_11_low360"]["false_block"] <= 13 and
            cohorts["side_rear"]["scores"]["flat_11_low360"]["false_block"] <= 32)
    output = {"scope": "Frozen .055m 2-D body-band footprint with exact saved self-filtered low 4cm returns appended to prior 11cm/head upper decision on V56 and V64; evaluator labels offline only. Diagnostic, no motion authority.",
              "source_sha256": sha(SOURCE),
              "selected_design_sha256": sha(ROOT / "robot/ape-current-design.json"),
              "flat_margin_m": FLAT_MARGIN_M,
              "cohorts": cohorts,
              "flat_low360_diagnostic_gate_pass": gate}
    (OUT / "low-range-flat-fusion-audit.json").write_text(
        json.dumps(output, indent=2) + "\n")
    (OUT / "audit_low_range_flat_fusion_v65_source.py").write_bytes(
        SOURCE.read_bytes())
    for label, cohort in cohorts.items():
        print(label, cohort["valid_moves"], cohort["scores"])
    print("flat low360 diagnostic gate:", gate)


if __name__ == "__main__":
    main()
