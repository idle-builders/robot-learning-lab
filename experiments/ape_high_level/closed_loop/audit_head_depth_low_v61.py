"""V61: saved-frame low head-depth returns versus an added 4 cm range plane."""

from __future__ import annotations

from collections import Counter
import json
import math
from pathlib import Path

import numpy as np

from audit_3d_clearance_layout_v53 import FRACTIONS, point_gap
from audit_near_floor_range_v54 import xyz_points
from depth_voxel import DepthVoxelMap, VOXEL_M
from robot_geometry import RobotGeometry
from run import ROOT, Room, sha


SOURCE = Path(__file__)
BASE = ROOT / "artifacts/ape-closed-loop-cove50-v56"
OUT = ROOT / "artifacts/ape-closed-loop-cove50-v61"
POINT_MARGIN_M = .065


def score(rows: list[dict], key: str) -> dict:
    return {"passes": sum(row[key] for row in rows),
            "false_safe": sum(row[key] and not row["evaluator_safe_5cm"]
                              for row in rows),
            "false_block": sum(not row[key] and row["evaluator_safe_5cm"]
                               for row in rows)}


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    base_path = BASE / "low-obstacle-transfer.json"
    baseline = json.loads(base_path.read_text())
    indexed = {(row["case"], tuple(row["direction_cell"])): row
               for row in baseline["rows"]}
    rows = []
    cases = []
    room = Room()
    try:
        for case in baseline["cases"]:
            if not case.get("valid_start", False):
                continue
            case_id = case["case"]
            saved_path = BASE / f"{case_id}-actor-input.npz"
            with np.load(saved_path) as saved:
                range_11 = [None if np.isnan(value) else float(value)
                            for value in saved["range_11_72_beams_m"]]
                range_4 = [None if np.isnan(value) else float(value)
                           for value in saved["range_4_72_beams_m"]]
                depth = saved["depth_m"].copy()
                camera_pose = (saved["camera_origin_m"].copy(),
                               saved["camera_rotation"].copy())
                actor_qpos = saved["actor_qpos"].copy()
                pose = saved["estimated_pose_xyyaw"].copy()
            robot = RobotGeometry(room.model)
            robot.set_state(actor_qpos)
            current_solids = robot.primitives()
            voxel = DepthVoxelMap()
            voxel.observe(
                depth, camera_pose, 0.,
                fovy_deg=float(room.model.cam_fovy[
                    room.model.camera("lower_rgb").id]),
                self_hit=lambda point: RobotGeometry.solid_distance(
                    point, current_solids) < 0.)
            points_11 = xyz_points(range_11, pose, .11, current_solids)
            points_4 = xyz_points(range_4, pose, .04, current_solids)
            upper = []
            low = []
            for (ix, iy, iz), hit_s in voxel.occupied.items():
                if hit_s < voxel.free.get((ix, iy, iz), -math.inf):
                    continue
                point = np.asarray([(ix + .5) * VOXEL_M,
                                    (iy + .5) * VOXEL_M,
                                    (iz + .5) * VOXEL_M])
                if .035 <= point[2] < .18:
                    low.append(point)
                if .18 <= point[2] <= .43:
                    upper.append(point)
            low = np.asarray(low, dtype=float).reshape(-1, 3)
            upper = np.asarray(upper, dtype=float).reshape(-1, 3)
            all_head = np.concatenate((low, upper))
            cases.append({"case": case_id, "saved_input_sha256": sha(saved_path),
                          "range_11_points": len(points_11),
                          "range_4_points": len(points_4),
                          "head_low_points": len(low),
                          "head_upper_points": len(upper)})
            for direction in ((1, 0), (1, 1), (0, 1), (-1, 1),
                              (-1, 0), (-1, -1), (0, -1), (1, -1)):
                old = indexed[(case_id, direction)]
                candidate = np.asarray(old["candidate_xy_m"])
                delta = candidate - pose[:2]
                gaps = {"range_11": math.inf, "range_4": math.inf,
                        "head_low": math.inf, "head_upper": math.inf}
                for fraction in FRACTIONS:
                    future = pose.copy()
                    future[:2] += fraction * delta
                    robot.set_state(robot.pose_qpos(actor_qpos, future))
                    solids = robot.primitives()
                    for name, points in (("range_11", points_11),
                                         ("range_4", points_4),
                                         ("head_low", low),
                                         ("head_upper", upper)):
                        gaps[name] = min(gaps[name], point_gap(points, solids))
                original_11 = min(gaps["range_11"], gaps["head_upper"]) >= POINT_MARGIN_M
                original_both = min(gaps["range_11"], gaps["range_4"],
                                    gaps["head_upper"]) >= POINT_MARGIN_M
                if original_11 != old["range_11_head_pass"] or original_both != old["both_head_pass"]:
                    raise AssertionError(f"V56 decision replay mismatch: {case_id} {direction}")
                rows.append({"case": case_id, "seed": case["seed"],
                             "direction_cell": list(direction),
                             "candidate_xy_m": candidate.tolist(),
                             "evaluator_model_clearance_m": old["evaluator_model_clearance_m"],
                             "evaluator_safe_5cm": old["evaluator_safe_5cm"],
                             "flat_2d_pass": old["flat_2d_pass"],
                             "range_11_head_upper_pass": original_11,
                             "range_11_head_all_pass": min(
                                 gaps["range_11"], gaps["head_low"],
                                 gaps["head_upper"]) >= POINT_MARGIN_M,
                             "both_head_upper_pass": original_both,
                             "both_head_all_pass": min(gaps.values()) >= POINT_MARGIN_M,
                             "head_all_only_pass": min(gaps["head_low"],
                                                       gaps["head_upper"]) >= POINT_MARGIN_M,
                             "gaps_m": {name: round(value, 4)
                                        for name, value in gaps.items()}})
    finally:
        room.close()
    assert len(rows) == baseline["valid_moves"] == 312
    scores = {name: score(rows, key) for name, key in (
        ("flat_2d", "flat_2d_pass"),
        ("range_11_head_upper", "range_11_head_upper_pass"),
        ("range_11_head_all", "range_11_head_all_pass"),
        ("both_head_upper", "both_head_upper_pass"),
        ("both_head_all", "both_head_all_pass"),
        ("head_all_only", "head_all_only_pass"))}
    assert scores["flat_2d"] == baseline["scores"]["flat_2d"]
    assert scores["range_11_head_upper"] == baseline["scores"]["range_11_head"]
    assert scores["both_head_upper"] == baseline["scores"]["both_head"]
    low_crate_misses = [row for row in rows if row["flat_2d_pass"]
                        and not row["evaluator_safe_5cm"]]
    assert len(low_crate_misses) == 5
    criterion = (scores["range_11_head_all"]["false_safe"] == 0 and
                 scores["range_11_head_all"]["false_block"] <= 11)
    output = {"scope": "Same-pose V56 saved depth/range replay with head occupied voxels extended below 18cm; frozen .065m selected-body point rule; evaluator labels inherited unchanged. Diagnostic only, no motion authority.",
              "source_sha256": sha(SOURCE),
              "v56_audit_sha256": sha(base_path),
              "selected_design_sha256": sha(ROOT / "robot/ape-current-design.json"),
              "point_margin_m": POINT_MARGIN_M,
              "cases": cases, "valid_moves": len(rows),
              "evaluator_safe_moves": sum(row["evaluator_safe_5cm"] for row in rows),
              "scores": scores,
              "low_crate_flat_false_safe_cases": low_crate_misses,
              "depth_replacement_diagnostic_criterion_pass": criterion,
              "rows": rows}
    (OUT / "head-depth-low-audit.json").write_text(
        json.dumps(output, indent=2) + "\n")
    (OUT / "audit_head_depth_low_v61_source.py").write_bytes(SOURCE.read_bytes())
    print("cases", len(cases), "moves", len(rows),
          "safe", output["evaluator_safe_moves"])
    print(scores)
    print("depth replacement diagnostic:", criterion)


if __name__ == "__main__":
    main()
