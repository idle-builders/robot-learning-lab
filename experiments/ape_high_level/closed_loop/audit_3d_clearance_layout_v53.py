"""Frozen selected-body point margin on new furnished table layouts."""

from __future__ import annotations

from collections import Counter
import json
import math
from pathlib import Path

import mujoco
import numpy as np

from depth_voxel import DepthVoxelMap, VOXEL_M
from footprint_route import scan_points
from goal_map_route import GoalMapRoute, footprint_distance
from robot_geometry import RobotGeometry
from run import ROOT, Room, sensed_range, sha


SOURCE = Path(__file__)
OUT = ROOT / "artifacts/ape-closed-loop-cove50-v53"
SCENES = ((101, .32, .42, .27),
          (102, .48, .42, .31),
          (103, .32, .56, .35),
          (104, .48, .56, .39))
STARTS = ((0., 0., 0.), (.08, 0., 0.), (.16, 0., 0.),
          (0., .12, -.45), (.08, .12, -.45), (.16, .12, -.45))
DELTAS = ((1, 0), (1, 1), (0, 1), (-1, 1), (-1, 0),
          (-1, -1), (0, -1), (1, -1))
FRACTIONS = (.25, .5, .75, 1.)
FLAT_MARGIN_M = .055
FROZEN_3D_MARGIN_M = .075


def point_gap(points: np.ndarray, solids: tuple) -> float:
    return min((RobotGeometry.solid_distance(point, solids)
                for point in points), default=math.inf)


def scene_solid_gaps(room: Room) -> dict:
    table_ids = [gid for gid in range(room.model.ngeom)
                 if room.model.geom(gid).name.startswith("loop_table_")]
    result = {}
    for name in ("TomatoSauce", "crate"):
        object_id = room.model.geom(f"loop_{name}_geom").id
        result[name] = min(float(mujoco.mj_geomDistance(
            room.model, room.data, object_id, table_id, 10., None))
            for table_id in table_ids)
    return result


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    cases = []
    rows = []
    room = Room()
    try:
        for seed, table_x, table_y_rel, height in SCENES:
            side = 1 if seed % 2 else -1
            room.reset()
            room.layout(seed, absent=False, furnished=True)
            room.model.body_pos[room.model.body("loop_table").id] = (
                table_x, side * table_y_rel, 0.)
            room.set_table_height(height)
            room._forward()
            scene_gaps = scene_solid_gaps(room)
            scene_valid = all(gap >= 0. for gap in scene_gaps.values())
            if not scene_valid:
                cases.append({"seed": seed, "table_center_xy_m": [table_x,
                               side * table_y_rel], "table_height_m": height,
                              "valid_scene": False,
                              "target_crate_table_solid_gaps_m": scene_gaps})
                continue
            for pose_index, (x, y_rel, yaw_rel) in enumerate(STARTS):
                pose = np.asarray([x, side * y_rel, side * yaw_rel])
                room.actual[:] = pose
                room.estimated[:] = pose
                room.head = 0.
                room._forward()
                start_clear = room._current_obstacle_clearance()[0]
                case_id = f"seed{seed}-pose{pose_index}"
                case = {"case": case_id, "seed": seed,
                        "table_center_xy_m": [table_x, side * table_y_rel],
                        "table_height_m": height,
                        "pose_xyyaw": pose.tolist(),
                        "valid_scene": True,
                        "target_crate_table_solid_gaps_m": scene_gaps,
                        "start_model_clearance_m": round(start_clear, 4),
                        "valid_start": start_clear >= .05}
                if start_clear < .05:
                    cases.append(case)
                    continue
                ranges = sensed_range(room, seed, 100 + pose_index)
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
                low_xy = scan_points(ranges, pose)
                upper_xy = voxel.upper_occupied_xy(0.)
                low_xyz = np.column_stack((low_xy, np.full(len(low_xy), .11)))
                low_xyz = np.asarray([
                    point for point in low_xyz
                    if RobotGeometry.solid_distance(point, current_solids) >= -.005
                ], dtype=float).reshape(-1, 3)
                upper_xyz = np.asarray([
                    [(ix + .5) * VOXEL_M, (iy + .5) * VOXEL_M,
                     (iz + .5) * VOXEL_M]
                    for (ix, iy, iz), hit_s in voxel.occupied.items()
                    if hit_s >= voxel.free.get((ix, iy, iz), -math.inf) and
                    .18 <= (iz + .5) * VOXEL_M <= .43
                ], dtype=float).reshape(-1, 3)
                case.update({"range_hit_points": len(low_xyz),
                             "head_depth_rays_used": sensor_info["depth_rays_used"],
                             "upper_occupied_3d": len(upper_xyz),
                             "upper_projected_2d": len(upper_xy)})
                np.savez_compressed(
                    OUT / f"{case_id}-actor-input.npz",
                    range_72_beams_m=np.asarray([
                        np.nan if item is None else item for item in ranges]),
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
                    three_low = math.inf
                    three_upper = math.inf
                    truth = math.inf
                    for fraction in FRACTIONS:
                        future = pose.copy()
                        future[:2] += fraction * delta
                        robot.set_state(robot.pose_qpos(actor_qpos, future))
                        solids = robot.primitives()
                        three_low = min(three_low, point_gap(low_xyz, solids))
                        three_upper = min(three_upper,
                                          point_gap(upper_xyz, solids))
                        room.actual[:] = future
                        room._forward()
                        truth = min(truth,
                                    room._current_obstacle_clearance()[0])
                    coverage = voxel.buffered_upper_coverage(
                        np.array([candidate[0], candidate[1], pose[2]]), 0.)
                    rows.append({"case": case_id, "seed": seed,
                                 "table_height_m": height,
                                 "direction_cell": [dx, dy],
                                 "candidate_xy_m": candidate.tolist(),
                                 "flat_low_gap_m": round(flat_low, 4),
                                 "flat_upper_gap_m": round(flat_upper, 4),
                                 "three_low_gap_m": round(three_low, 4),
                                 "three_upper_gap_m": round(three_upper, 4),
                                 "flat_2d_pass": min(flat_low, flat_upper) >= FLAT_MARGIN_M,
                                 "frozen_3d_point_pass": min(three_low, three_upper) >= FROZEN_3D_MARGIN_M,
                                 "evaluator_model_clearance_m": round(truth, 4),
                                 "evaluator_safe_5cm": truth >= .05,
                                 "buffered_upper_coverage": coverage})
                cases.append(case)
    finally:
        room.close()
    (OUT / "audit_3d_clearance_layout_v53_source.py").write_bytes(
        SOURCE.read_bytes())
    scores = {}
    for label, key in (("flat_2d", "flat_2d_pass"),
                       ("frozen_3d_point", "frozen_3d_point_pass")):
        scores[label] = {"passes": sum(row[key] for row in rows),
                         "false_safe": sum(row[key] and not
                                           row["evaluator_safe_5cm"]
                                           for row in rows),
                         "false_block": sum(not row[key] and
                                            row["evaluator_safe_5cm"]
                                            for row in rows)}
    counts = Counter(row["evaluator_safe_5cm"] for row in rows)
    output = {"scope": "Frozen .075m selected-body point SDF vs .055m 2-D footprint on new table positions/heights and fixed start poses; fresh noisy low range and one self-filtered lower-head depth frame. Evaluator truth offline only; unknown does not authorize motion.",
              "source_sha256": sha(SOURCE),
              "selected_design_sha256": sha(ROOT / "robot/ape-current-design.json"),
              "thresholds_m": {"flat_2d": FLAT_MARGIN_M,
                               "frozen_3d_point": FROZEN_3D_MARGIN_M,
                               "evaluator_safe": .05},
              "scene_specs": SCENES, "start_specs": STARTS,
              "cases": cases, "valid_moves": len(rows),
              "evaluator_safe_moves": counts[True],
              "scores": scores, "rows": rows}
    (OUT / "layout-transfer-audit.json").write_text(
        json.dumps(output, indent=2) + "\n")
    print("cases", len(cases), "valid_starts",
          sum(case.get("valid_start", False) for case in cases),
          "moves", len(rows), "evaluator_safe", counts[True])
    print(scores)


if __name__ == "__main__":
    main()
