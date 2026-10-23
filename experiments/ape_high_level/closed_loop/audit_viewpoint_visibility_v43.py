"""Replay first actor map and score bounded target-view routes against RGB truth.

The planner receives only its archived pose, range, target estimate and depth
frame. MuJoCo object segmentation is evaluator-only and labels whether the
named target would be visible from hypothetical reachable viewpoints.
"""

from __future__ import annotations

import json
import math
from pathlib import Path

import numpy as np

from depth_voxel import DepthVoxelMap
from footprint_route import scan_points
from goal_map_route import GoalMapRoute
from robot_geometry import RobotGeometry
from run import ROOT, Room, sensed_range, sha


SOURCE = Path(__file__)
INPUT = ROOT / "artifacts/ape-closed-loop-cove50-v42"
OUT = ROOT / "artifacts/ape-closed-loop-cove50-v43"
CAMERA = "chest_low50_x45"


def path_to(cell: tuple[int, int], origin: tuple[int, int],
            parent: dict) -> list[tuple[int, int]] | None:
    path = [cell]
    while path[-1] != origin:
        if path[-1] not in parent:
            return None
        path.append(parent[path[-1]])
    return list(reversed(path))


def main() -> None:
    rows = []
    input_hashes = {}
    replay_counts = {}
    room = Room()
    try:
        for seed in (85, 88):
            room.reset()
            room.layout(seed, absent=False, furnished=True)
            side = 1 if seed % 2 else -1
            room.model.body_pos[room.model.body("loop_table").id] = (
                .40, side * .46, 0.)
            room.set_table_height(.37)
            room._forward()
            episode = INPUT / (f"furnished_table_corridor-seed{seed}-0-"
                               "chest_rgbd_range_guard_goal_map-none")
            input_hashes[str(seed)] = {
                "trace_sha256": sha(episode / "trace.json"),
                "head_depth_frame_sha256": sha(
                    episode / "head-depth-frame-1.npz")}
            trace = json.loads((episode / "trace.json").read_text())
            first = next(event for event in trace["events"]
                         if event["action"] == "initial_goal_map_route_planned")
            archived = np.load(episode / "head-depth-frame-1.npz")
            pose = np.asarray(first["pose_estimate_xyyaw"], dtype=float)
            target = np.asarray(first["target_estimate_xy_m"], dtype=float)
            scan = first["range_72_beams_m"]
            stamp = float(first["t_s"])
            # The scripted owner can finish a body turn before confirmation.
            # The virtual gyro estimate includes the run's +0.003 rad bias.
            room.actual[2] = pose[2] - .003
            room._forward()
            robot = RobotGeometry(room.model)
            robot.set_state(archived["actor_qpos"])
            solids = robot.primitives()
            voxel = DepthVoxelMap()
            voxel.observe(archived["depth_m"],
                          (archived["camera_origin_m"],
                           archived["camera_rotation"]), stamp,
                          fovy_deg=float(room.model.cam_fovy[
                              room.model.camera("lower_rgb").id]),
                          self_hit=lambda point: RobotGeometry.solid_distance(
                              point, solids) < 0.)
            upper = voxel.upper_occupied_xy(stamp)
            route = GoalMapRoute()
            # The live loop initialized its range map at owner confirmation
            # before the guarded head look, then refreshed it before plan().
            route.observe(sensed_range(room, seed, -1), pose, 1.6)
            route.observe(scan, pose, stamp)
            route.plan(scan, pose, target, None, stamp, searching=False,
                       extra_obstacles_xy=upper,
                       allow_confirmed_frontier=True, true_frontier=True)
            if abs(len(route.last_reachable_costs) -
                   first["goal_map_route"]["reachable_cells"]) > 1:
                raise AssertionError(
                    f"Replayed actor map differs from V42 first route: "
                    f"{len(route.last_reachable_costs)} vs "
                    f"{first['goal_map_route']['reachable_cells']}")
            replay_counts[str(seed)] = {
                "replayed_reachable": len(route.last_reachable_costs),
                "live_reachable": first["goal_map_route"]["reachable_cells"],
                "replayed_upper_marks": len(upper),
                "live_upper_marks": first["goal_map_route"]["fresh_upper_obstacles"]}
            origin = route.cell(pose[:2])
            obstacles = np.concatenate((route.occupied_points(stamp),
                                        scan_points(scan, pose), upper))
            for cell, cost in route.last_reachable_costs.items():
                if cell == origin or cost > 1.2:
                    continue
                path = path_to(cell, origin, route.last_parent)
                if path is None or len(path) - 1 > 10:
                    continue
                xy = route.xy(cell)
                target_gap = float(np.linalg.norm(target - xy))
                if not (.27 <= target_gap <= .70):
                    continue
                neighbors = [(cell[0] + dx, cell[1] + dy) for dx, dy in
                             ((1, 0), (-1, 0), (0, 1), (0, -1))]
                unknown = sum(neighbor not in route.cells
                              for neighbor in neighbors)
                if not unknown:
                    continue
                visibility = []
                for waypoint in path[1:]:
                    view_xy = route.xy(waypoint)
                    sightline_gap = route.sightline_gap(
                        obstacles, view_xy, target)
                    bearing = math.atan2(target[1] - view_xy[1],
                                         target[0] - view_xy[0])
                    relative = abs(math.atan2(math.sin(bearing - pose[2]),
                                              math.cos(bearing - pose[2])))
                    visibility.append((sightline_gap, relative))
                predicted_visible_path = all(gap >= .08 and bearing <= .72
                                             for gap, bearing in visibility)
                # Evaluator-only image visibility; never enters the route.
                room.actual[:] = (xy[0], xy[1], pose[2])
                room._forward()
                box = room.truth_box("TomatoSauce", CAMERA)
                pixel_area = (0 if box is None else
                              (box[2] - box[0]) * (box[3] - box[1]))
                rows.append({"seed": seed,
                             "xy_m": [round(float(x), 3) for x in xy],
                             "target_distance_est_m": round(target_gap, 4),
                             "route_cost": round(float(cost), 4),
                             "path_steps": len(path) - 1,
                             "unknown_cardinal_neighbors": unknown,
                             "minimum_actor_sightline_gap_m": round(
                                 min(gap for gap, _ in visibility), 4),
                             "maximum_actor_relative_bearing_rad": round(
                                 max(bearing for _, bearing in visibility), 4),
                             "predicted_visible_path": predicted_visible_path,
                             "evaluator_target_visible_at_goal": pixel_area >= 150,
                             "evaluator_target_pixels_at_goal": pixel_area,
                             "path_cells": [list(item) for item in path]})
    finally:
        room.close()
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "audit_viewpoint_visibility_v43_source.py").write_bytes(
        SOURCE.read_bytes())
    output = {"scope": "V42 first-frame actor map replay on mirrored high-table development rooms. Reachable true frontier candidates within ten 4cm path cells are scored by actor range/depth sightline and fixed chest bearing. Actual target pixel visibility at hypothetical endpoint is evaluator-only; no detector run, temporal movement or action authority.",
              "source_sha256": sha(SOURCE),
              "selected_design_sha256": sha(ROOT / "robot/ape-current-design.json"),
              "inputs_sha256": input_hashes,
              "replay_counts": replay_counts,
              "rows": rows}
    (OUT / "viewpoint-visibility-audit.json").write_text(
        json.dumps(output, indent=2) + "\n")
    for seed in (85, 88):
        subset = [row for row in rows if row["seed"] == seed]
        predicted = [row for row in subset if row["predicted_visible_path"]]
        print(seed, "reachable_boundaries", len(subset),
              "actor_path_visible", len(predicted),
              "evaluator_visible", sum(row["evaluator_target_visible_at_goal"]
                                       for row in subset),
              "true_visible_of_predicted", sum(
                  row["evaluator_target_visible_at_goal"] for row in predicted))
        print("best predicted", sorted(predicted,
              key=lambda row: (row["target_distance_est_m"] +
                               .01 * row["path_steps"]))[:5])


if __name__ == "__main__":
    main()
