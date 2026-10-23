"""V56: frozen 6.5 cm two-height point rule on changed low obstacles."""

from __future__ import annotations

from collections import Counter
import json
import math
from pathlib import Path

import mujoco
import numpy as np

from audit_3d_clearance_layout_v53 import (DELTAS, FRACTIONS, FLAT_MARGIN_M,
                                             STARTS, point_gap, scene_solid_gaps)
from audit_near_floor_range_v54 import near_floor_scan, xyz_points
from depth_voxel import DepthVoxelMap, VOXEL_M
from footprint_route import scan_points
from goal_map_route import GoalMapRoute, footprint_distance
from robot_geometry import RobotGeometry
from run import ROOT, Room, sensed_range, sha


SOURCE = Path(__file__)
OUT = ROOT / "artifacts/ape-closed-loop-cove50-v56"
SCENES = ((201, .36, -.02, .07, .07, .045),
          (202, .40, -.08, .11, .06, .070),
          (203, .44, -.12, .06, .11, .110),
          (204, .48, -.16, .10, .08, .050),
          (205, .36, +.06, .08, .10, .090),
          (206, .40, +.02, .12, .07, .120),
          (207, .44, -.02, .07, .12, .060),
          (208, .48, -.06, .10, .10, .140))
POINT_MARGIN_M = .065


def score(rows: list[dict], key: str) -> dict:
    return {"passes": sum(row[key] for row in rows),
            "false_safe": sum(row[key] and not row["evaluator_safe_5cm"]
                              for row in rows),
            "false_block": sum(not row[key] and row["evaluator_safe_5cm"]
                               for row in rows)}


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    cases = []
    rows = []
    room = Room()
    try:
        for seed, crate_x, crate_y_rel, half_x, half_y, half_height in SCENES:
            side = 1 if seed % 2 else -1
            room.reset()
            room.layout(seed, absent=False, furnished=True)
            crate_id = room.model.geom("loop_crate_geom").id
            room.model.geom_size[crate_id] = (half_x, half_y, half_height)
            room.model.body_pos[room.model.body("loop_crate").id] = (
                crate_x, side * crate_y_rel, half_height)
            room._forward()
            scene_gaps = scene_solid_gaps(room)
            scene_gaps.update({
                "crate_" + name: float(mujoco.mj_geomDistance(
                    room.model, room.data, crate_id,
                    room.model.geom(f"loop_{name}_geom").id, 10., None))
                for name in ("TomatoSauce", "box", "shelf", "stool")})
            scene_valid = all(gap >= 0. for gap in scene_gaps.values())
            if not scene_valid:
                cases.append({"seed": seed, "crate_center_xy_m": [
                    crate_x, side * crate_y_rel],
                    "crate_half_size_m": [half_x, half_y, half_height],
                    "valid_scene": False,
                    "scene_solid_gaps_m": scene_gaps})
                continue
            for pose_index, (x, y_rel, yaw_rel) in enumerate(STARTS):
                pose = np.asarray([x + crate_x - .36,
                                   side * (y_rel + crate_y_rel + .02),
                                   side * yaw_rel])
                room.actual[:] = pose
                room.estimated[:] = pose
                room.head = 0.
                room._forward()
                start_clear = room._current_obstacle_clearance()[0]
                case_id = f"seed{seed}-pose{pose_index}"
                case = {"case": case_id, "seed": seed,
                        "crate_center_xy_m": [crate_x, side * crate_y_rel],
                        "crate_half_size_m": [half_x, half_y, half_height],
                        "pose_xyyaw": pose.tolist(),
                        "valid_scene": True,
                        "scene_solid_gaps_m": scene_gaps,
                        "start_model_clearance_m": round(start_clear, 4),
                        "valid_start": start_clear >= .05}
                if start_clear < .05:
                    cases.append(case)
                    continue

                ranges_11 = sensed_range(room, seed, 100 + pose_index)
                ranges_4 = near_floor_scan(room, seed, pose_index)
                head_rgb, depth = room.capture("lower_rgb", depth=True)
                camera_pose = room.camera_pose_estimate("lower_rgb")
                actor_qpos = room.pose_data.qpos.copy()
                robot = RobotGeometry(room.model)
                robot.set_state(actor_qpos)
                current_solids = robot.primitives()
                voxel = DepthVoxelMap()
                sensor_info = voxel.observe(
                    depth, camera_pose, 0.,
                    fovy_deg=float(room.model.cam_fovy[
                        room.model.camera("lower_rgb").id]),
                    self_hit=lambda point: RobotGeometry.solid_distance(
                        point, current_solids) < 0.)
                low_xy = scan_points(ranges_11, pose)
                upper_xy = voxel.upper_occupied_xy(0.)
                points_11 = xyz_points(ranges_11, pose, .11, current_solids)
                points_4 = xyz_points(ranges_4, pose, .04, current_solids)
                upper_xyz = np.asarray([
                    [(ix + .5) * VOXEL_M, (iy + .5) * VOXEL_M,
                     (iz + .5) * VOXEL_M]
                    for (ix, iy, iz), hit_s in voxel.occupied.items()
                    if hit_s >= voxel.free.get((ix, iy, iz), -math.inf) and
                    .18 <= (iz + .5) * VOXEL_M <= .43
                ], dtype=float).reshape(-1, 3)
                case.update({"range_11_points": len(points_11),
                             "range_4_points": len(points_4),
                             "head_depth_rays_used": sensor_info["depth_rays_used"],
                             "upper_occupied_3d": len(upper_xyz),
                             "upper_projected_2d": len(upper_xy)})
                np.savez_compressed(
                    OUT / f"{case_id}-actor-input.npz",
                    range_11_72_beams_m=np.asarray([
                        np.nan if item is None else item for item in ranges_11]),
                    range_4_72_beams_m=np.asarray([
                        np.nan if item is None else item for item in ranges_4]),
                    depth_m=depth, camera_origin_m=camera_pose[0],
                    camera_rotation=camera_pose[1], actor_qpos=actor_qpos,
                    estimated_pose_xyyaw=pose)
                head_rgb.save(OUT / f"{case_id}-head-rgb.png")
                origin_cell = GoalMapRoute.cell(pose[:2])
                for dx, dy in DELTAS:
                    next_cell = origin_cell[0] + dx, origin_cell[1] + dy
                    candidate = GoalMapRoute.xy(next_cell)
                    delta = candidate - pose[:2]
                    flat_low = min(footprint_distance(
                        low_xy, pose[:2] + fraction * delta, pose[2])
                        for fraction in FRACTIONS)
                    flat_upper = min(footprint_distance(
                        upper_xy, pose[:2] + fraction * delta, pose[2])
                        for fraction in FRACTIONS)
                    point_gaps = {"range_11": math.inf, "range_4": math.inf,
                                  "head_upper": math.inf}
                    truth = math.inf
                    for fraction in FRACTIONS:
                        future = pose.copy()
                        future[:2] += fraction * delta
                        robot.set_state(robot.pose_qpos(actor_qpos, future))
                        solids = robot.primitives()
                        for key, points in (("range_11", points_11),
                                            ("range_4", points_4),
                                            ("head_upper", upper_xyz)):
                            point_gaps[key] = min(point_gaps[key],
                                                  point_gap(points, solids))
                        room.actual[:] = future
                        room._forward()
                        truth = min(truth,
                                    room._current_obstacle_clearance()[0])
                    coverage = voxel.buffered_upper_coverage(
                        np.array([candidate[0], candidate[1], pose[2]]), 0.)
                    row = {"case": case_id, "seed": seed,
                           "crate_half_size_m": [half_x, half_y, half_height],
                           "direction_cell": [dx, dy],
                           "candidate_xy_m": candidate.tolist(),
                           "flat_low_gap_m": round(flat_low, 4),
                           "flat_upper_gap_m": round(flat_upper, 4),
                           "point_gaps_m": {key: round(value, 4)
                                            for key, value in point_gaps.items()},
                           "flat_2d_pass": min(flat_low, flat_upper) >= FLAT_MARGIN_M,
                           "range_11_head_pass": min(point_gaps["range_11"],
                                                     point_gaps["head_upper"]) >= POINT_MARGIN_M,
                           "range_4_head_pass": min(point_gaps["range_4"],
                                                    point_gaps["head_upper"]) >= POINT_MARGIN_M,
                           "both_head_pass": min(point_gaps.values()) >= POINT_MARGIN_M,
                           "evaluator_model_clearance_m": round(truth, 4),
                           "evaluator_safe_5cm": truth >= .05,
                           "buffered_upper_coverage": coverage}
                    rows.append(row)
                cases.append(case)
    finally:
        room.close()

    scores = {name: score(rows, key) for name, key in (
        ("flat_2d", "flat_2d_pass"),
        ("range_11_head", "range_11_head_pass"),
        ("range_4_head", "range_4_head_pass"),
        ("both_head", "both_head_pass"))}
    counts = Counter(row["evaluator_safe_5cm"] for row in rows)
    transfer_pass = (len(rows) >= 100 and scores["both_head"]["false_safe"] == 0
                     and scores["both_head"]["false_block"] <
                     scores["flat_2d"]["false_block"])
    output = {"scope": "Frozen .065m combined 4cm+11cm+head selected-body point SDF vs .055m flat 2D proxy on changed low-obstacle position/width/height. Actor sees independent noisy scans/depth only. Evaluator labels offline. No motion authority.",
              "source_sha256": sha(SOURCE),
              "selected_design_sha256": sha(ROOT / "robot/ape-current-design.json"),
              "point_margin_m": POINT_MARGIN_M,
              "flat_margin_m": FLAT_MARGIN_M,
              "evaluator_safe_margin_m": .05,
              "scene_specs": SCENES, "start_specs": STARTS,
              "cases": cases, "valid_moves": len(rows),
              "evaluator_safe_moves": counts[True],
              "scores": scores, "transfer_gate_pass": transfer_pass,
              "rows": rows}
    (OUT / "low-obstacle-transfer.json").write_text(
        json.dumps(output, indent=2) + "\n")
    (OUT / "audit_low_obstacle_transfer_v56_source.py").write_bytes(
        SOURCE.read_bytes())
    print("scenes", len(SCENES), "valid starts",
          sum(case.get("valid_start", False) for case in cases),
          "moves", len(rows), "safe", counts[True])
    print(scores)
    print("transfer gate:", transfer_pass)


if __name__ == "__main__":
    main()
