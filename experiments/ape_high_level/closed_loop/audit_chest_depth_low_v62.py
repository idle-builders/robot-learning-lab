"""V62: new chest RGB-D frame versus saved near-floor range observations."""

from __future__ import annotations

import json
import math
from pathlib import Path

import numpy as np

from audit_3d_clearance_layout_v53 import FRACTIONS
from depth_voxel import DepthVoxelMap, VOXEL_M
from robot_geometry import RobotGeometry
from run import ROOT, Room, sha


SOURCE = Path(__file__)
BASE = ROOT / "artifacts/ape-closed-loop-cove50-v56"
HEAD = ROOT / "artifacts/ape-closed-loop-cove50-v61"
OUT = ROOT / "artifacts/ape-closed-loop-cove50-v62"
MARGIN_M = .065


def score(rows: list[dict], key: str) -> dict:
    return {"passes": sum(row[key] for row in rows),
            "false_safe": sum(row[key] and not row["evaluator_safe_5cm"]
                              for row in rows),
            "false_block": sum(not row[key] and row["evaluator_safe_5cm"]
                               for row in rows)}


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    head_path = HEAD / "head-depth-low-audit.json"
    prior = json.loads(head_path.read_text())
    reference = json.loads((BASE / "low-obstacle-transfer.json").read_text())
    old = {(row["case"], tuple(row["direction_cell"])): row
           for row in prior["rows"]}
    rows = []
    cases = []
    room = Room()
    try:
        for case in prior["cases"]:
            case_id = case["case"]
            seed = int(case_id.split("-pose")[0].replace("seed", ""))
            geometry = next(item for item in reference["cases"]
                            if item.get("case") == case_id)
            room.reset()
            room.layout(seed, absent=False, furnished=True)
            crate_id = room.model.geom("loop_crate_geom").id
            half = geometry["crate_half_size_m"]
            room.model.geom_size[crate_id] = half
            crate_xy = geometry["crate_center_xy_m"]
            room.model.body_pos[room.model.body("loop_crate").id] = (
                crate_xy[0], crate_xy[1], half[2])
            saved_path = BASE / f"{case_id}-actor-input.npz"
            with np.load(saved_path) as saved:
                actor_qpos = saved["actor_qpos"].copy()
                pose = saved["estimated_pose_xyyaw"].copy()
            room.actual[:] = pose
            room.estimated[:] = pose
            room.head = 0.
            room._forward()
            rgb, depth = room.capture("chest_low50_x45", depth=True)
            camera_pose = room.camera_pose_estimate("chest_low50_x45")
            if not np.allclose(room.pose_data.qpos, actor_qpos, atol=1e-8):
                raise AssertionError(f"Actor qpos mismatch in {case_id}")
            rgb.save(OUT / f"{case_id}-chest-rgb.png")
            np.savez_compressed(
                OUT / f"{case_id}-chest-depth.npz",
                depth_m=depth, camera_origin_m=camera_pose[0],
                camera_rotation=camera_pose[1], estimated_pose_xyyaw=pose,
                actor_qpos=actor_qpos)
            robot = RobotGeometry(room.model)
            robot.set_state(actor_qpos)
            current = robot.primitives()
            voxel = DepthVoxelMap()
            info = voxel.observe(
                depth, camera_pose, 0.,
                fovy_deg=float(room.model.cam_fovy[
                    room.model.camera("chest_low50_x45").id]),
                self_hit=lambda point: RobotGeometry.solid_distance(
                    point, current) < 0.)
            chest_low = []
            chest_upper = []
            for (ix, iy, iz), hit_s in voxel.occupied.items():
                if hit_s < voxel.free.get((ix, iy, iz), -math.inf):
                    continue
                point = np.asarray([(ix + .5) * VOXEL_M,
                                    (iy + .5) * VOXEL_M,
                                    (iz + .5) * VOXEL_M])
                if .035 <= point[2] < .18:
                    chest_low.append(point)
                if .18 <= point[2] <= .43:
                    chest_upper.append(point)
            chest_low = np.asarray(chest_low, dtype=float).reshape(-1, 3)
            chest_upper = np.asarray(chest_upper, dtype=float).reshape(-1, 3)
            chest_all = np.concatenate((chest_low, chest_upper))
            cases.append({"case": case_id,
                          "v56_actor_input_sha256": sha(saved_path),
                          "chest_low_points": len(chest_low),
                          "chest_upper_points": len(chest_upper),
                          "chest_depth_rays_used": info["depth_rays_used"]})
            for direction in ((1, 0), (1, 1), (0, 1), (-1, 1),
                              (-1, 0), (-1, -1), (0, -1), (1, -1)):
                baseline = old[(case_id, direction)]
                candidate = np.asarray(baseline["candidate_xy_m"])
                delta = candidate - pose[:2]
                chest_gap = math.inf
                for fraction in FRACTIONS:
                    future = pose.copy()
                    future[:2] += fraction * delta
                    robot.set_state(robot.pose_qpos(actor_qpos, future))
                    if len(chest_all):
                        chest_gap = min(chest_gap, float(np.min(
                            RobotGeometry.solid_distances(
                                chest_all, robot.primitives()))))
                gaps = baseline["gaps_m"]
                head_all = min(gaps["head_low"], gaps["head_upper"])
                row = {"case": case_id, "seed": seed,
                       "direction_cell": list(direction),
                       "evaluator_model_clearance_m": baseline["evaluator_model_clearance_m"],
                       "evaluator_safe_5cm": baseline["evaluator_safe_5cm"],
                       "flat_2d_pass": baseline["flat_2d_pass"],
                       "range_11_head_all_pass": baseline["range_11_head_all_pass"],
                       "range_11_chest_all_pass": min(
                           gaps["range_11"], chest_gap) >= MARGIN_M,
                       "range_11_both_camera_all_pass": min(
                           gaps["range_11"], head_all, chest_gap) >= MARGIN_M,
                       "both_range_head_upper_pass": baseline["both_head_upper_pass"],
                       "both_range_both_camera_all_pass": min(
                           gaps["range_11"], gaps["range_4"], head_all,
                           chest_gap) >= MARGIN_M,
                       "chest_gap_m": round(chest_gap, 4),
                       "saved_head_and_range_gaps_m": gaps}
                rows.append(row)
    finally:
        room.close()
    assert len(rows) == prior["valid_moves"] == 312
    scores = {name: score(rows, key) for name, key in (
        ("flat_2d", "flat_2d_pass"),
        ("range_11_head_all", "range_11_head_all_pass"),
        ("range_11_chest_all", "range_11_chest_all_pass"),
        ("range_11_both_camera_all", "range_11_both_camera_all_pass"),
        ("both_range_head_upper", "both_range_head_upper_pass"),
        ("both_range_both_camera_all", "both_range_both_camera_all_pass"))}
    assert scores["flat_2d"] == prior["scores"]["flat_2d"]
    assert scores["range_11_head_all"] == prior["scores"]["range_11_head_all"]
    assert scores["both_range_head_upper"] == prior["scores"]["both_head_upper"]
    misses = [row for row in rows if row["flat_2d_pass"]
              and not row["evaluator_safe_5cm"]]
    assert len(misses) == 5
    criterion = (scores["range_11_chest_all"]["false_safe"] == 0 and
                 scores["range_11_chest_all"]["false_block"] <= 11)
    output = {"scope": "One new existing-mount chest RGB-D frame at exact V56 varied-crate actor poses, frozen .065m selected-body point rule and saved scans/head gaps; evaluator labels inherited. Diagnostic only; no physical mount or motion authority.",
              "source_sha256": sha(SOURCE),
              "v61_audit_sha256": sha(head_path),
              "selected_design_sha256": sha(ROOT / "robot/ape-current-design.json"),
              "point_margin_m": MARGIN_M,
              "cases": cases, "valid_moves": len(rows),
              "evaluator_safe_moves": sum(row["evaluator_safe_5cm"] for row in rows),
              "scores": scores,
              "five_low_crate_flat_false_safe_cases": misses,
              "chest_replacement_diagnostic_criterion_pass": criterion,
              "rows": rows}
    (OUT / "chest-depth-low-audit.json").write_text(
        json.dumps(output, indent=2) + "\n")
    (OUT / "audit_chest_depth_low_v62_source.py").write_bytes(SOURCE.read_bytes())
    print("cases", len(cases), "moves", len(rows),
          "safe", output["evaluator_safe_moves"])
    print(scores)
    print("chest replacement diagnostic:", criterion)


if __name__ == "__main__":
    main()
