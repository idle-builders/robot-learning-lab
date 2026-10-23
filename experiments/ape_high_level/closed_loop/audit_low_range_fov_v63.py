"""V63: body-forward angular masks on saved 4 cm range scans."""

from __future__ import annotations

import json
import math
from pathlib import Path

import numpy as np

from audit_3d_clearance_layout_v53 import FRACTIONS
from audit_near_floor_range_v54 import xyz_points
from robot_geometry import RobotGeometry
from run import ROOT, Room, sha


SOURCE = Path(__file__)
BASE = ROOT / "artifacts/ape-closed-loop-cove50-v56"
HEAD = ROOT / "artifacts/ape-closed-loop-cove50-v61"
OUT = ROOT / "artifacts/ape-closed-loop-cove50-v63"
MARGIN_M = .065
FIELDS_DEG = (90, 180, 270, 360)


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    head_path = HEAD / "head-depth-low-audit.json"
    reference = json.loads(head_path.read_text())
    indexed = {(row["case"], tuple(row["direction_cell"])): row
               for row in reference["rows"]}
    rows = []
    cases = []
    room = Room()
    try:
        for case in reference["cases"]:
            case_id = case["case"]
            saved_path = BASE / f"{case_id}-actor-input.npz"
            with np.load(saved_path) as saved:
                ranges = [None if np.isnan(value) else float(value)
                          for value in saved["range_4_72_beams_m"]]
                actor_qpos = saved["actor_qpos"].copy()
                pose = saved["estimated_pose_xyyaw"].copy()
            robot = RobotGeometry(room.model)
            robot.set_state(actor_qpos)
            current = robot.primitives()
            angle = np.linspace(-math.pi, math.pi, len(ranges), endpoint=False)
            source_points = {}
            for field in FIELDS_DEG:
                half_angle = math.radians(field / 2)
                masked = [value if field == 360 or abs(float(a)) <= half_angle
                          else None for a, value in zip(angle, ranges)]
                source_points[field] = xyz_points(masked, pose, .04, current)
            cases.append({"case": case_id,
                          "v56_actor_input_sha256": sha(saved_path),
                          "low_points_by_field_deg": {
                              str(field): len(points)
                              for field, points in source_points.items()}})
            for direction in ((1, 0), (1, 1), (0, 1), (-1, 1),
                              (-1, 0), (-1, -1), (0, -1), (1, -1)):
                old = indexed[(case_id, direction)]
                candidate = np.asarray(old["candidate_xy_m"])
                delta = candidate - pose[:2]
                low_gaps = {field: math.inf for field in FIELDS_DEG}
                for fraction in FRACTIONS:
                    future = pose.copy()
                    future[:2] += fraction * delta
                    robot.set_state(robot.pose_qpos(actor_qpos, future))
                    solids = robot.primitives()
                    for field, points in source_points.items():
                        if len(points):
                            low_gaps[field] = min(low_gaps[field], float(np.min(
                                RobotGeometry.solid_distances(points, solids))))
                base_gap = min(old["gaps_m"]["range_11"],
                               old["gaps_m"]["head_upper"])
                decisions = {field: min(base_gap, low_gaps[field]) >= MARGIN_M
                             for field in FIELDS_DEG}
                if decisions[360] != old["both_head_upper_pass"]:
                    raise AssertionError(f"V56 full-range replay mismatch: {case_id} {direction}")
                rows.append({"case": case_id, "seed": int(case_id[4:7]),
                             "direction_cell": list(direction),
                             "evaluator_model_clearance_m": old["evaluator_model_clearance_m"],
                             "evaluator_safe_5cm": old["evaluator_safe_5cm"],
                             "flat_2d_pass": old["flat_2d_pass"],
                             "field_pass": {str(field): decisions[field]
                                            for field in FIELDS_DEG},
                             "low_gap_m": {str(field): round(low_gaps[field], 4)
                                           for field in FIELDS_DEG},
                             "saved_11_head_gap_m": round(base_gap, 4)})
    finally:
        room.close()
    assert len(rows) == reference["valid_moves"] == 312
    scores = {str(field): {"passes": sum(row["field_pass"][str(field)]
                                          for row in rows),
                           "false_safe": sum(row["field_pass"][str(field)] and
                                             not row["evaluator_safe_5cm"]
                                             for row in rows),
                           "false_block": sum(not row["field_pass"][str(field)] and
                                              row["evaluator_safe_5cm"]
                                              for row in rows)}
              for field in FIELDS_DEG}
    assert scores["360"] == reference["scores"]["both_head_upper"]
    flat_misses = [row for row in rows if row["flat_2d_pass"]
                   and not row["evaluator_safe_5cm"]]
    assert len(flat_misses) == 5
    qualifying = [field for field in FIELDS_DEG if field < 360 and
                  scores[str(field)]["false_safe"] == 0 and
                  scores[str(field)]["false_block"] <= 11]
    output = {"scope": "Body-forward angular masks of exact saved noisy 4cm V56 scan; frozen .065m selected-body point rule with saved 11cm/head upper gap, 312 evaluator-labeled actions. Diagnostic only, no mount or motion authority.",
              "source_sha256": sha(SOURCE),
              "v61_audit_sha256": sha(head_path),
              "selected_design_sha256": sha(ROOT / "robot/ape-current-design.json"),
              "point_margin_m": MARGIN_M,
              "fields_deg": FIELDS_DEG,
              "cases": cases, "valid_moves": len(rows),
              "evaluator_safe_moves": sum(row["evaluator_safe_5cm"] for row in rows),
              "scores": scores,
              "five_low_crate_flat_false_safe_cases": flat_misses,
              "narrow_fields_meeting_diagnostic_gate_deg": qualifying,
              "rows": rows}
    (OUT / "low-range-fov-audit.json").write_text(
        json.dumps(output, indent=2) + "\n")
    (OUT / "audit_low_range_fov_v63_source.py").write_bytes(SOURCE.read_bytes())
    print("cases", len(cases), "moves", len(rows),
          "safe", output["evaluator_safe_moves"])
    print(scores)
    print("narrow fields meeting diagnostic:", qualifying)


if __name__ == "__main__":
    main()
