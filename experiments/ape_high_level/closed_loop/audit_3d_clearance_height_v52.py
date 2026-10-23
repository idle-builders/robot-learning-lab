"""Compare flat and selected-body 3-D point clearances across table heights."""

from __future__ import annotations

from collections import defaultdict
import json
import math
from pathlib import Path

import numpy as np

from depth_voxel import DepthVoxelMap, VOXEL_M
from footprint_route import STATIC_MARGIN_M, scan_points
from goal_map_route import GoalMapRoute, footprint_distance
from robot_geometry import RobotGeometry
from run import ROOT, Room, sensed_range, sha


SOURCE = Path(__file__)
INPUT = ROOT / "artifacts/ape-closed-loop-cove50-v51"
OUT = ROOT / "artifacts/ape-closed-loop-cove50-v52"
HEIGHTS = (.25, .29, .33, .37)
DELTAS = ((1, 0), (1, 1), (0, 1), (-1, 1), (-1, 0),
          (-1, -1), (0, -1), (1, -1))
FRACTIONS = (.25, .5, .75, 1.)


def main() -> None:
    rows = []
    scene_cases = []
    inputs = {}
    room = Room()
    try:
        for seed in (85, 88):
            saved_path = INPUT / (
                f"furnished_table_corridor-seed{seed}-0-"
                "chest_rgbd_range_guard_goal_map-none/") / "first-stop-actor-state.npz"
            saved = np.load(saved_path)
            inputs[str(seed)] = sha(saved_path)
            pose = saved["estimated_pose_xyyaw"].copy()
            actual = saved["actual_pose_xyyaw_evaluator_only"].copy()
            actor_qpos = saved["actor_qpos_estimated"].copy()
            for height in HEIGHTS:
                room.reset()
                room.layout(seed, absent=False, furnished=True)
                side = 1 if seed % 2 else -1
                room.model.body_pos[room.model.body("loop_table").id] = (
                    .40, side * .46, 0.)
                room.set_table_height(height)
                room.actual[:] = actual
                room.estimated[:] = pose
                room.head = float(actor_qpos[room.head_addr])
                room._forward()
                start_clear = room._current_obstacle_clearance()[0]
                case = {"seed": seed, "table_height_m": height,
                        "start_model_clearance_m": round(start_clear, 4),
                        "valid_start": start_clear >= .05}
                if start_clear < .05:
                    scene_cases.append(case)
                    continue
                ranges = sensed_range(room, seed, -1)
                head_rgb, head_depth = room.capture("lower_rgb", depth=True)
                head_pose = room.camera_pose_estimate("lower_rgb")
                robot = RobotGeometry(room.model)
                robot.set_state(room.pose_data.qpos.copy())
                current_solids = robot.primitives()
                voxel = DepthVoxelMap()
                observation = voxel.observe(
                    head_depth, head_pose, 2.,
                    fovy_deg=float(room.model.cam_fovy[
                        room.model.camera("lower_rgb").id]),
                    self_hit=lambda point: RobotGeometry.solid_distance(
                        point, current_solids) < 0.)
                low_xy = scan_points(ranges, pose)
                upper_xy = voxel.upper_occupied_xy(2.)
                low_xyz = np.column_stack((low_xy, np.full(len(low_xy), .11)))
                low_xyz = np.asarray([
                    point for point in low_xyz
                    if RobotGeometry.solid_distance(point, current_solids) >= -.005
                ], dtype=float).reshape(-1, 3)
                upper_xyz = np.asarray([
                    [(x + .5) * VOXEL_M, (y + .5) * VOXEL_M,
                     (z + .5) * VOXEL_M]
                    for (x, y, z), hit_s in voxel.occupied.items()
                    if hit_s >= voxel.free.get((x, y, z), -math.inf) and
                    .18 <= (z + .5) * VOXEL_M <= .43
                ], dtype=float).reshape(-1, 3)
                case.update({"low_hits": len(low_xyz),
                             "head_depth_rays_used": observation["depth_rays_used"],
                             "upper_occupied_xyz": len(upper_xyz),
                             "upper_projected_xy": len(upper_xy)})
                case_name = f"seed{seed}-table{round(height * 100):02d}cm"
                OUT.mkdir(parents=True, exist_ok=True)
                np.savez_compressed(
                    OUT / f"{case_name}-actor-input.npz",
                    range_72_beams_m=np.asarray([
                        np.nan if reading is None else reading for reading in ranges]),
                    depth_m=head_depth, camera_origin_m=head_pose[0],
                    camera_rotation=head_pose[1],
                    estimated_pose_xyyaw=pose,
                    actor_qpos_estimated=actor_qpos)
                head_rgb.save(OUT / f"{case_name}-head-rgb.png")
                origin_cell = GoalMapRoute.cell(pose[:2])
                for dx, dy in DELTAS:
                    neighbor_cell = origin_cell[0] + dx, origin_cell[1] + dy
                    candidate = GoalMapRoute.xy(neighbor_cell)
                    delta = candidate - pose[:2]
                    low_2d = min(footprint_distance(
                        low_xy, pose[:2] + fraction * delta, pose[2])
                        for fraction in FRACTIONS)
                    upper_2d = min(footprint_distance(
                        upper_xy, pose[:2] + fraction * delta, pose[2])
                        for fraction in FRACTIONS)
                    low_3d = math.inf
                    upper_3d = math.inf
                    true_clear = math.inf
                    for fraction in FRACTIONS:
                        future_pose = pose.copy()
                        future_pose[:2] += fraction * delta
                        robot.set_state(robot.pose_qpos(actor_qpos, future_pose))
                        future_solids = robot.primitives()
                        if len(low_xyz):
                            low_3d = min(low_3d, *(
                                RobotGeometry.solid_distance(point, future_solids)
                                for point in low_xyz))
                        if len(upper_xyz):
                            upper_3d = min(upper_3d, *(
                                RobotGeometry.solid_distance(point, future_solids)
                                for point in upper_xyz))
                        room.actual[:] = (
                            actual[0] + fraction * delta[0],
                            actual[1] + fraction * delta[1], actual[2])
                        room._forward()
                        true_clear = min(true_clear,
                                         room._current_obstacle_clearance()[0])
                    voxel_coverage = voxel.buffered_upper_coverage(
                        np.array([candidate[0], candidate[1], pose[2]]), 2.)
                    rows.append({"seed": seed, "table_height_m": height,
                                 "direction_cell": [dx, dy],
                                 "candidate_xy_m": candidate.round(4).tolist(),
                                 "low_2d_gap_m": round(low_2d, 4),
                                 "upper_2d_gap_m": round(upper_2d, 4),
                                 "low_3d_sdf_gap_m": round(low_3d, 4),
                                 "upper_3d_sdf_gap_m": round(upper_3d, 4),
                                 "flat_2d_pass": min(low_2d, upper_2d) >= STATIC_MARGIN_M,
                                 "selected_3d_point_pass": min(low_3d, upper_3d) >= STATIC_MARGIN_M,
                                 "evaluator_model_clearance_m": round(true_clear, 4),
                                 "evaluator_safe_5cm": true_clear >= .05,
                                 "buffered_upper_coverage": voxel_coverage})
                scene_cases.append(case)
    finally:
        room.close()
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "audit_3d_clearance_height_v52_source.py").write_bytes(
        SOURCE.read_bytes())
    grouped = defaultdict(list)
    for row in rows:
        grouped[str(round(row["table_height_m"] * 100))].append(row)
    counts = {}
    for height, subset in grouped.items():
        counts[height] = {"moves": len(subset),
                          "evaluator_safe": sum(r["evaluator_safe_5cm"] for r in subset)}
        for mode, key in (("flat_2d", "flat_2d_pass"),
                          ("selected_3d_point", "selected_3d_point_pass")):
            counts[height][mode] = {
                "false_safe": sum(r[key] and not r["evaluator_safe_5cm"]
                                  for r in subset),
                "false_block": sum(not r[key] and r["evaluator_safe_5cm"]
                                   for r in subset),
                "passes": sum(r[key] for r in subset)}
    output = {"scope": "Two V51 stop poses, four table heights, fresh noisy 11cm range plus one joint-aware lower-head depth frame; eight quarter-sampled neighbor moves each. Point-SDF pass is diagnostic only; unknown remains unknown. Selected-model clearance evaluator-only.",
              "source_sha256": sha(SOURCE),
              "selected_design_sha256": sha(ROOT / "robot/ape-current-design.json"),
              "input_stop_state_sha256": inputs,
              "cases": scene_cases, "counts_by_height_cm": counts,
              "rows": rows}
    (OUT / "height-clearance-audit.json").write_text(
        json.dumps(output, indent=2) + "\n")
    for height, summary in counts.items():
        print(height, summary)


if __name__ == "__main__":
    main()
