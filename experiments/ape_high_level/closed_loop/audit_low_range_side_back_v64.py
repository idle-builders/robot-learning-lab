"""V64: new side/rear body headings for frozen low-range FOV masks."""

from __future__ import annotations

import json
import math
from pathlib import Path

import mujoco
import numpy as np

from audit_3d_clearance_layout_v53 import (DELTAS, FRACTIONS,
                                             FLAT_MARGIN_M, STARTS,
                                             scene_solid_gaps)
from audit_low_obstacle_transfer_v56 import SCENES as CRATE_SCENES
from audit_near_floor_range_v54 import xyz_points
from depth_voxel import DepthVoxelMap, VOXEL_M
from footprint_route import scan_points
from goal_map_route import GoalMapRoute, footprint_distance
from robot_geometry import RobotGeometry
from run import ROOT, Room, sensed_near_floor_range, sensed_range, sha


SOURCE = Path(__file__)
OUT = ROOT / "artifacts/ape-closed-loop-cove50-v64"
SCENES = tuple(scene for scene in CRATE_SCENES
               if scene[0] in (201, 203, 205, 207))
YAW_OFFSETS = ((90, math.pi / 2), (180, math.pi))
FIELDS_DEG = (90, 180, 270, 360)
POINT_MARGIN_M = .065


def score(rows: list[dict], key: str) -> dict:
    return {"passes": sum(row[key] for row in rows),
            "false_safe": sum(row[key] and not row["evaluator_safe_5cm"]
                              for row in rows),
            "false_block": sum(not row[key] and row["evaluator_safe_5cm"]
                               for row in rows)}


def gap(points: np.ndarray, solids: tuple) -> float:
    return (float(np.min(RobotGeometry.solid_distances(points, solids)))
            if len(points) else math.inf)


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    room = Room()
    cases = []
    rows = []
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
            scene_gaps["crate_TomatoSauce"] = float(mujoco.mj_geomDistance(
                room.model, room.data, crate_id,
                room.model.geom("loop_TomatoSauce_geom").id, 10., None))
            valid_scene = all(value >= 0. for value in scene_gaps.values())
            if not valid_scene:
                cases.append({"seed": seed, "valid_scene": False,
                              "scene_solid_gaps_m": scene_gaps})
                continue
            for pose_index, (x, y_rel, yaw_rel) in enumerate(STARTS):
                for yaw_deg, yaw_offset in YAW_OFFSETS:
                    yaw = side * yaw_rel + yaw_offset
                    yaw = math.atan2(math.sin(yaw), math.cos(yaw))
                    pose = np.asarray([x + crate_x - .36,
                                       side * (y_rel + crate_y_rel + .02), yaw])
                    room.actual[:] = pose
                    room.estimated[:] = pose
                    room.head = 0.
                    room._forward()
                    start_clear = room._current_obstacle_clearance()[0]
                    case_id = f"seed{seed}-pose{pose_index}-yaw{yaw_deg}"
                    case = {"case": case_id, "seed": seed,
                            "pose_xyyaw": pose.tolist(),
                            "yaw_offset_deg": yaw_deg,
                            "crate_center_xy_m": [crate_x, side * crate_y_rel],
                            "crate_half_size_m": [half_x, half_y, half_height],
                            "scene_solid_gaps_m": scene_gaps,
                            "valid_scene": True,
                            "start_model_clearance_m": round(start_clear, 4),
                            "valid_start": start_clear >= .05}
                    if start_clear < .05:
                        cases.append(case)
                        continue

                    scan_step = 100 + pose_index + 100 * (yaw_deg // 90)
                    range_11 = sensed_range(room, seed, scan_step)
                    range_4 = sensed_near_floor_range(room, seed, scan_step)
                    rgb, depth = room.capture("lower_rgb", depth=True)
                    camera_pose = room.camera_pose_estimate("lower_rgb")
                    actor_qpos = room.pose_data.qpos.copy()
                    robot = RobotGeometry(room.model)
                    robot.set_state(actor_qpos)
                    current = robot.primitives()
                    voxel = DepthVoxelMap()
                    info = voxel.observe(
                        depth, camera_pose, 0.,
                        fovy_deg=float(room.model.cam_fovy[
                            room.model.camera("lower_rgb").id]),
                        self_hit=lambda point: RobotGeometry.solid_distance(
                            point, current) < 0.)
                    head_upper = np.asarray([
                        [(ix + .5) * VOXEL_M, (iy + .5) * VOXEL_M,
                         (iz + .5) * VOXEL_M]
                        for (ix, iy, iz), hit_s in voxel.occupied.items()
                        if hit_s >= voxel.free.get((ix, iy, iz), -math.inf)
                        and .18 <= (iz + .5) * VOXEL_M <= .43
                    ], dtype=float).reshape(-1, 3)
                    upper_xy = voxel.upper_occupied_xy(0.)
                    range_11_xy = scan_points(range_11, pose)
                    points_11 = xyz_points(range_11, pose, .11, current)
                    angles = np.linspace(-math.pi, math.pi, 72, endpoint=False)
                    low_points = {}
                    for field in FIELDS_DEG:
                        half_angle = math.radians(field / 2)
                        masked = [value if field == 360 or abs(float(angle)) <= half_angle
                                  else None for angle, value in zip(angles, range_4)]
                        low_points[field] = xyz_points(masked, pose, .04, current)
                    case.update({"range_11_points": len(points_11),
                                 "low_points_by_field_deg": {
                                     str(field): len(points)
                                     for field, points in low_points.items()},
                                 "head_upper_points": len(head_upper),
                                 "head_depth_rays_used": info["depth_rays_used"]})
                    rgb.save(OUT / f"{case_id}-head-rgb.png")
                    np.savez_compressed(
                        OUT / f"{case_id}-actor-input.npz",
                        range_11_72_beams_m=np.asarray([
                            np.nan if value is None else value for value in range_11]),
                        range_4_72_beams_m=np.asarray([
                            np.nan if value is None else value for value in range_4]),
                        depth_m=depth, camera_origin_m=camera_pose[0],
                        camera_rotation=camera_pose[1], actor_qpos=actor_qpos,
                        estimated_pose_xyyaw=pose)
                    origin_cell = GoalMapRoute.cell(pose[:2])
                    for dx, dy in DELTAS:
                        candidate = GoalMapRoute.xy(
                            (origin_cell[0] + dx, origin_cell[1] + dy))
                        delta = candidate - pose[:2]
                        displacement_angle = math.atan2(delta[1], delta[0])
                        body_relative = math.atan2(
                            math.sin(displacement_angle - pose[2]),
                            math.cos(displacement_angle - pose[2]))
                        flat_low = min(footprint_distance(
                            range_11_xy, pose[:2] + fraction * delta, pose[2])
                            for fraction in FRACTIONS)
                        flat_upper = min(footprint_distance(
                            upper_xy, pose[:2] + fraction * delta, pose[2])
                            for fraction in FRACTIONS)
                        point_gaps = {"range_11": math.inf,
                                      "head_upper": math.inf,
                                      **{str(field): math.inf
                                         for field in FIELDS_DEG}}
                        truth = math.inf
                        for fraction in FRACTIONS:
                            future = pose.copy()
                            future[:2] += fraction * delta
                            robot.set_state(robot.pose_qpos(actor_qpos, future))
                            solids = robot.primitives()
                            point_gaps["range_11"] = min(
                                point_gaps["range_11"], gap(points_11, solids))
                            point_gaps["head_upper"] = min(
                                point_gaps["head_upper"], gap(head_upper, solids))
                            for field in FIELDS_DEG:
                                point_gaps[str(field)] = min(
                                    point_gaps[str(field)],
                                    gap(low_points[field], solids))
                            room.actual[:] = future
                            room._forward()
                            truth = min(truth,
                                        room._current_obstacle_clearance()[0])
                        rows.append({"case": case_id, "seed": seed,
                                     "yaw_offset_deg": yaw_deg,
                                     "direction_cell": [dx, dy],
                                     "body_relative_move_rad": round(body_relative, 4),
                                     "move_outside_forward_90": abs(body_relative) > math.pi / 4,
                                     "candidate_xy_m": candidate.tolist(),
                                     "flat_2d_pass": min(flat_low, flat_upper) >= FLAT_MARGIN_M,
                                     "field_pass": {str(field): min(
                                         point_gaps["range_11"],
                                         point_gaps["head_upper"],
                                         point_gaps[str(field)]) >= POINT_MARGIN_M
                                         for field in FIELDS_DEG},
                                     "point_gaps_m": {key: round(value, 4)
                                                      for key, value in point_gaps.items()},
                                     "evaluator_model_clearance_m": round(truth, 4),
                                     "evaluator_safe_5cm": truth >= .05})
                    cases.append(case)
    finally:
        room.close()

    scores = {str(field): {"passes": sum(row["field_pass"][str(field)]
                                          for row in rows),
                           "false_safe": sum(row["field_pass"][str(field)] and
                                             not row["evaluator_safe_5cm"]
                                             for row in rows),
                           "false_block": sum(not row["field_pass"][str(field)] and
                                              row["evaluator_safe_5cm"]
                                              for row in rows)}
              for field in FIELDS_DEG}
    scores["flat_2d"] = score(rows, "flat_2d_pass")
    outside_unsafe = sum(not row["evaluator_safe_5cm"] and
                         row["move_outside_forward_90"] for row in rows)
    interpretable = outside_unsafe >= 20
    qualifiers = [field for field in FIELDS_DEG if field < 360 and
                  interpretable and scores["360"]["false_safe"] == 0 and
                  scores[str(field)]["false_safe"] == 0 and
                  scores[str(field)]["false_block"] <=
                  scores["360"]["false_block"] + 2]
    output = {"scope": "Fresh noisy two-height range/head depth at rotated body headings on four changed low-crate scenes; frozen .065m selected-body point rule, independent offline >=.05m evaluator clearance. No motion authority.",
              "source_sha256": sha(SOURCE),
              "selected_design_sha256": sha(ROOT / "robot/ape-current-design.json"),
              "scene_specs": SCENES, "start_specs": STARTS,
              "yaw_offsets_deg": [item[0] for item in YAW_OFFSETS],
              "fields_deg": FIELDS_DEG, "point_margin_m": POINT_MARGIN_M,
              "cases": cases, "valid_moves": len(rows),
              "evaluator_safe_moves": sum(row["evaluator_safe_5cm"] for row in rows),
              "unsafe_moves_outside_forward_90": outside_unsafe,
              "fov_interpretable": interpretable,
              "scores": scores,
              "narrow_fields_meeting_transfer_gate_deg": qualifiers,
              "rows": rows}
    (OUT / "low-range-side-back-audit.json").write_text(
        json.dumps(output, indent=2) + "\n")
    (OUT / "audit_low_range_side_back_v64_source.py").write_bytes(
        SOURCE.read_bytes())
    print("valid starts", sum(case.get("valid_start", False) for case in cases),
          "moves", len(rows), "safe", output["evaluator_safe_moves"],
          "outside-forward unsafe", outside_unsafe)
    print(scores)
    print("narrow transfer qualifiers:", qualifiers)


if __name__ == "__main__":
    main()
