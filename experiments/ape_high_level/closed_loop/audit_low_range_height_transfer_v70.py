"""V70: frozen 4+8+11 cm point rule on new rotated low-crate scenes."""

from __future__ import annotations

import json
import math
from pathlib import Path

import mujoco
import numpy as np

from audit_3d_clearance_layout_v53 import (DELTAS, FRACTIONS,
                                             FLAT_MARGIN_M, STARTS,
                                             scene_solid_gaps)
from audit_near_floor_range_v54 import xyz_points
from depth_voxel import DepthVoxelMap, VOXEL_M
from footprint_route import scan_points
from goal_map_route import GoalMapRoute, footprint_distance
from robot_geometry import RobotGeometry
from run import ROOT, Room, sensed_range, sha


SOURCE = Path(__file__)
OUT = ROOT / "artifacts/ape-closed-loop-cove50-v70"
SCENES = ((301, .38, -.04, .09, .075, .030),
          (302, .42, -.10, .08, .08, .040),
          (303, .46, -.14, .12, .05, .055),
          (304, .50, -.06, .06, .11, .065),
          (305, .34, +.04, .08, .12, .025),
          (306, .38, +.08, .11, .06, .050),
          (307, .42, +.04, .07, .09, .070),
          (308, .46, -.10, .09, .07, .035))
YAW_OFFSETS = ((90, math.pi / 2), (180, math.pi))
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
    cases = []
    rows = []
    room = Room()
    try:
        robot = RobotGeometry(room.model)
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
                    range_11 = sensed_range(room, seed, scan_step,
                                             height_m=.11)
                    range_4 = sensed_range(room, seed, 50000 + scan_step,
                                            height_m=.04)
                    range_8 = sensed_range(room, seed, 80000 + scan_step,
                                            height_m=.08)
                    rgb, depth = room.capture("lower_rgb", depth=True)
                    camera_pose = room.camera_pose_estimate("lower_rgb")
                    actor_qpos = room.pose_data.qpos.copy()
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
                    points = {
                        "range_11": xyz_points(range_11, pose, .11, current),
                        "range_4": xyz_points(range_4, pose, .04, current),
                        "range_8": xyz_points(range_8, pose, .08, current),
                        "head_upper": head_upper}
                    case.update({"range_11_points": len(points["range_11"]),
                                 "range_4_points": len(points["range_4"]),
                                 "range_8_points": len(points["range_8"]),
                                 "head_upper_points": len(head_upper),
                                 "head_depth_rays_used": info["depth_rays_used"]})
                    if pose_index == 0 and yaw_deg == 90:
                        rgb.save(OUT / f"seed{seed}-head-rgb.png")
                    np.savez_compressed(
                        OUT / f"{case_id}-actor-input.npz",
                        range_11_72_beams_m=np.asarray([
                            np.nan if v is None else v for v in range_11]),
                        range_4_72_beams_m=np.asarray([
                            np.nan if v is None else v for v in range_4]),
                        range_8_72_beams_m=np.asarray([
                            np.nan if v is None else v for v in range_8]),
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
                        point_gaps = {name: math.inf for name in points}
                        truth = math.inf
                        closest = ("", "")
                        for fraction in FRACTIONS:
                            future = pose.copy()
                            future[:2] += fraction * delta
                            robot.set_state(robot.pose_qpos(actor_qpos, future))
                            solids = robot.primitives()
                            for name, cloud in points.items():
                                point_gaps[name] = min(point_gaps[name],
                                                       gap(cloud, solids))
                            room.actual[:] = future
                            room._forward()
                            observed = room._current_obstacle_clearance()
                            if observed[0] < truth:
                                truth = observed[0]
                                closest = (observed[1], observed[2])
                        rows.append({
                            "case": case_id, "seed": seed,
                            "crate_height_m": 2 * half_height,
                            "yaw_offset_deg": yaw_deg,
                            "direction_cell": [dx, dy],
                            "body_relative_move_rad": round(body_relative, 4),
                            "move_outside_forward_90": abs(body_relative) > math.pi / 4,
                            "candidate_xy_m": candidate.tolist(),
                            "flat_11_pass": min(flat_low, flat_upper) >= FLAT_MARGIN_M,
                            "point_4_11_pass": min(point_gaps["range_11"],
                                                   point_gaps["range_4"],
                                                   point_gaps["head_upper"]) >= POINT_MARGIN_M,
                            "point_4_8_11_pass": min(point_gaps.values()) >= POINT_MARGIN_M,
                            "point_gaps_m": {name: round(value, 4)
                                             for name, value in point_gaps.items()},
                            "evaluator_model_clearance_m": round(truth, 4),
                            "evaluator_closest_geom_pair": closest,
                            "evaluator_safe_5cm": truth >= .05})
                    cases.append(case)
    finally:
        room.close()

    scores = {name: score(rows, f"{name}_pass") for name in
              ("flat_11", "point_4_11", "point_4_8_11")}
    outside_unsafe = sum(not row["evaluator_safe_5cm"] and
                         row["move_outside_forward_90"] for row in rows)
    low_crate_unsafe = sum(not row["evaluator_safe_5cm"] and
                           row["crate_height_m"] < .08 and
                           "loop_crate" in row["evaluator_closest_geom_pair"][1]
                           for row in rows)
    valid_scenes = {case["seed"] for case in cases if case.get("valid_scene")}
    interpretable = (len(valid_scenes) == len(SCENES) and
                     outside_unsafe >= 20 and low_crate_unsafe >= 5)
    gate = (interpretable and scores["point_4_8_11"]["false_safe"] == 0 and
            scores["point_4_8_11"]["false_block"] <=
            scores["flat_11"]["false_block"])
    output = {"scope": "Fresh noisy 4/8/11cm virtual range/head depth on eight preregistered new crate layouts and side/rear headings; frozen .065m selected-body point rule, .055m flat proxy and independent offline >=.05m body clearance. No motion authority.",
              "source_sha256": sha(SOURCE),
              "selected_design_sha256": sha(ROOT / "robot/ape-current-design.json"),
              "scene_specs": SCENES, "start_specs": STARTS,
              "yaw_offsets_deg": [item[0] for item in YAW_OFFSETS],
              "point_margin_m": POINT_MARGIN_M,
              "cases": cases, "valid_moves": len(rows),
              "evaluator_safe_moves": sum(row["evaluator_safe_5cm"] for row in rows),
              "unsafe_moves_outside_forward_90": outside_unsafe,
              "unsafe_sub8cm_crate_moves": low_crate_unsafe,
              "transfer_interpretable": interpretable,
              "scores": scores, "three_plane_transfer_gate_pass": gate,
              "rows": rows}
    (OUT / "low-range-height-transfer.json").write_text(
        json.dumps(output, indent=2) + "\n")
    (OUT / "audit_low_range_height_transfer_v70_source.py").write_bytes(
        SOURCE.read_bytes())
    print("valid scenes", len(valid_scenes), "starts",
          sum(case.get("valid_start", False) for case in cases),
          "moves", len(rows), "safe", output["evaluator_safe_moves"])
    print("outside-forward unsafe", outside_unsafe,
          "sub8 crate unsafe", low_crate_unsafe)
    print(scores)
    print("three-plane transfer gate:", gate)


if __name__ == "__main__":
    main()
