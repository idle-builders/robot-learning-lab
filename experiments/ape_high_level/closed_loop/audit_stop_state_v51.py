"""Attribute V50-equivalent actor stop to map, footprint, view or progress."""

from __future__ import annotations

import heapq
import json
import math
from pathlib import Path

import numpy as np

from depth_voxel import DepthVoxelMap, VOXEL_M
from footprint_route import STATIC_MARGIN_M, scan_points
from goal_map_route import GoalMapRoute, footprint_distance
from robot_geometry import RobotGeometry
from run import H, W, ROOT, Room, sha
from target_view import view_ray_evidence


SOURCE = Path(__file__)
INPUT = ROOT / "artifacts/ape-closed-loop-cove50-v51"
OUT = INPUT / "stop-state-attribution.json"
DELTAS = ((1, 0), (1, 1), (0, 1), (-1, 1), (-1, 0),
          (-1, -1), (0, -1), (1, -1))


def true_route(room: Room, start: np.ndarray,
               target: np.ndarray) -> dict:
    """Evaluator-only fixed-yaw search; never passed to actor."""
    clearance_cache = {}

    def clearance(xy: np.ndarray) -> float:
        key = tuple(np.round(xy, 4))
        if key not in clearance_cache:
            room.actual[:2] = xy
            room._forward()
            clearance_cache[key] = room._current_obstacle_clearance()[0]
        return clearance_cache[key]

    def xy(node: tuple[int, int]) -> np.ndarray:
        return start[:2] + .04 * np.asarray(node)

    queue = [(0., 0, (0, 0))]
    cost = {(0, 0): 0.}
    parent = {}
    serial = 0
    goal = None
    while queue and len(cost) < 3500:
        _, _, node = heapq.heappop(queue)
        point = xy(node)
        if np.linalg.norm(point - target) <= .34:
            goal = node
            break
        for dx, dy in DELTAS:
            nxt = node[0] + dx, node[1] + dy
            future = xy(nxt)
            if not (-.32 <= future[0] <= 1.16 and
                    -.72 <= future[1] <= .72):
                continue
            step = future - point
            if min(clearance(point + fraction * step)
                   for fraction in (.25, .5, .75, 1.)) < .05:
                continue
            new_cost = cost[node] + float(np.linalg.norm(step))
            if new_cost >= cost.get(nxt, math.inf):
                continue
            cost[nxt] = new_cost
            parent[nxt] = node
            serial += 1
            heuristic = max(float(np.linalg.norm(future - target)) - .34, 0.)
            heapq.heappush(queue, (new_cost + heuristic, serial, nxt))
    path = []
    if goal is not None:
        cursor = goal
        while cursor != (0, 0):
            path.append(cursor)
            cursor = parent[cursor]
        path.append((0, 0))
        path.reverse()
    return {"found": goal is not None,
            "path_xy_m": [xy(node).round(4).tolist() for node in path],
            "path_steps": max(0, len(path) - 1),
            "expanded_cells": len(cost),
            "start_clearance_m": clearance(start[:2]),
            "path_min_clearance_m": (min(clearance(xy(node)) for node in path)
                                      if path else None)}


def main() -> None:
    rows = []
    inputs = {}
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
            state_path = episode / "first-stop-actor-state.npz"
            trace_path = episode / "trace.json"
            head_path = episode / "head-depth-frame-1.npz"
            inputs[str(seed)] = {"state": sha(state_path),
                                 "trace": sha(trace_path),
                                 "head": sha(head_path)}
            state = np.load(state_path)
            head = np.load(head_path)
            pose = state["estimated_pose_xyyaw"]
            actual = state["actual_pose_xyyaw_evaluator_only"]
            target = state["target_estimate_xy_m"]
            stamp = float(state["capture_time_s"])
            ranges = [None if np.isnan(value) else float(value)
                      for value in state["range_72_beams_m"]]
            route = GoalMapRoute()
            route.cells = {
                (int(x), int(y)): (float(free_s), float(hit_s))
                for x, y, free_s, hit_s in state["range_map_cells"]}
            view_map = DepthVoxelMap()
            view_map.free = {
                (int(x), int(y), int(z)): float(stamp_s)
                for x, y, z, stamp_s in state["view_free_voxels"]}
            view_map.occupied = {
                (int(x), int(y), int(z)): float(stamp_s)
                for x, y, z, stamp_s in state["view_occupied_voxels"]}
            view_map.last_time_s = max(
                [item for item in view_map.free.values()] +
                [item for item in view_map.occupied.values()], default=None)
            upper = state["fresh_upper_obstacles_xy_m"]
            obstacle_sources = {
                "mapped_low_range": route.occupied_points(stamp),
                "current_low_range": scan_points(ranges, pose),
                "fresh_upper_head_depth": upper}
            obstacles = np.concatenate(tuple(obstacle_sources.values()))
            room.actual[:] = actual
            room.estimated[:] = pose
            room.head = float(state["actor_qpos_estimated"][room.head_addr])
            room._forward()
            camera_origin, camera_rotation = room.camera_pose_estimate(
                "chest_low50_x45")
            fovy = float(room.model.cam_fovy[
                room.model.camera("chest_low50_x45").id])
            true_start = actual.copy()
            true_target = state["target_true_xy_m_evaluator_only"]
            evaluator = true_route(room, true_start, true_target)
            low_xy = np.concatenate((obstacle_sources["mapped_low_range"],
                                     obstacle_sources["current_low_range"]))
            low_xyz = np.column_stack((low_xy, np.full(len(low_xy), .11)))
            robot = RobotGeometry(room.model)
            actor_qpos = state["actor_qpos_estimated"]
            robot.set_state(head["actor_qpos"])
            head_solids = robot.primitives()
            head_map = DepthVoxelMap()
            head_map.observe(
                head["depth_m"],
                (head["camera_origin_m"], head["camera_rotation"]),
                float(head["capture_time_s"]),
                fovy_deg=float(room.model.cam_fovy[
                    room.model.camera("lower_rgb").id]),
                self_hit=lambda point: RobotGeometry.solid_distance(
                    point, head_solids) < 0.)
            head_upper_xy = head_map.upper_occupied_xy(stamp)
            if (len(head_upper_xy) != len(upper) or
                    not np.allclose(head_upper_xy, upper)):
                raise AssertionError("Head upper marks differ from saved actor stop")
            upper_xyz = np.asarray([
                [(x + .5) * VOXEL_M, (y + .5) * VOXEL_M,
                 (z + .5) * VOXEL_M]
                for (x, y, z), hit_s in head_map.occupied.items()
                if hit_s >= stamp - 1.6 and
                hit_s >= head_map.free.get((x, y, z), -math.inf) and
                .18 <= (z + .5) * VOXEL_M <= .43
            ], dtype=float).reshape(-1, 3)
            robot.set_state(actor_qpos)
            current_solids = robot.primitives()
            low_xyz = np.asarray([
                point for point in low_xyz
                if RobotGeometry.solid_distance(point, current_solids) >= -.005
            ], dtype=float).reshape(-1, 3)
            neighbors = []
            origin_cell = route.cell(pose[:2])
            initial_distance = float(np.linalg.norm(target - pose[:2]))
            for dx, dy in DELTAS:
                cell = origin_cell[0] + dx, origin_cell[1] + dy
                candidate = route.xy(cell)
                step = candidate - pose[:2]
                known_free = route.known_free(cell, stamp)
                actor_clear = min(footprint_distance(
                    obstacles, pose[:2] + fraction * step, pose[2])
                    for fraction in (.5, 1.))
                source_gaps = {
                    name: round(min(footprint_distance(
                        points, pose[:2] + fraction * step, pose[2])
                        for fraction in (.5, 1.)), 4)
                    for name, points in obstacle_sources.items()}
                selected_body_low_gap = math.inf
                selected_body_upper_gap = math.inf
                for fraction in (.5, 1.):
                    future_pose = pose.copy()
                    future_pose[:2] += fraction * step
                    robot.set_state(robot.pose_qpos(actor_qpos, future_pose))
                    future_solids = robot.primitives()
                    if len(low_xyz):
                        selected_body_low_gap = min(
                            selected_body_low_gap,
                            *(RobotGeometry.solid_distance(point, future_solids)
                              for point in low_xyz))
                    if len(upper_xyz):
                        selected_body_upper_gap = min(
                            selected_body_upper_gap,
                            *(RobotGeometry.solid_distance(point, future_solids)
                              for point in upper_xyz))
                projected_origin = camera_origin.copy()
                projected_origin[:2] += candidate - pose[:2]
                view = view_ray_evidence(
                    view_map, (projected_origin, camera_rotation),
                    state["target_surface_xyz_m"], fovy, (H, W), stamp)
                gain = initial_distance - float(np.linalg.norm(target - candidate))
                true_candidate = actual[:2] + candidate - pose[:2]
                room.actual[:] = (true_candidate[0], true_candidate[1], actual[2])
                room._forward()
                true_clear = room._current_obstacle_clearance()[0]
                neighbors.append({"direction_cell": [dx, dy],
                                  "actor_xy_m": candidate.round(4).tolist(),
                                  "known_free_low_range": known_free,
                                  "actor_footprint_gap_m": round(actor_clear, 4),
                                  "footprint_gap_by_source_m": source_gaps,
                                  "selected_body_low_hit_sdf_gap_m": round(
                                      selected_body_low_gap, 4),
                                  "selected_body_upper_head_sdf_gap_m": round(
                                      selected_body_upper_gap, 4),
                                  "actor_footprint_pass": actor_clear >= STATIC_MARGIN_M,
                                  "target_distance_gain_m": round(gain, 4),
                                  "progress_rule_pass": gain >= .01,
                                  "view": view,
                                  "evaluator_model_clearance_m": round(true_clear, 4)})
            evaluator_first_actor = None
            if evaluator["found"] and evaluator["path_steps"]:
                true_first = np.asarray(evaluator["path_xy_m"][1])
                first_delta = true_first - actual[:2]
                actor_first = pose[:2] + first_delta
                first_step = actor_first - pose[:2]
                first_source_gaps = {
                    name: round(min(footprint_distance(
                        points, pose[:2] + fraction * first_step, pose[2])
                        for fraction in (.25, .5, .75, 1.)), 4)
                    for name, points in obstacle_sources.items()}
                first_low_gap = math.inf
                first_upper_gap = math.inf
                for fraction in (.25, .5, .75, 1.):
                    future_pose = pose.copy()
                    future_pose[:2] += fraction * first_step
                    robot.set_state(robot.pose_qpos(actor_qpos, future_pose))
                    solids = robot.primitives()
                    if len(low_xyz):
                        first_low_gap = min(
                            first_low_gap,
                            *(RobotGeometry.solid_distance(point, solids)
                              for point in low_xyz))
                    if len(upper_xyz):
                        first_upper_gap = min(
                            first_upper_gap,
                            *(RobotGeometry.solid_distance(point, solids)
                              for point in upper_xyz))
                future_origin = camera_origin.copy()
                future_origin[:2] += first_delta
                first_view = view_ray_evidence(
                    view_map, (future_origin, camera_rotation),
                    state["target_surface_xyz_m"], fovy, (H, W), stamp)
                evaluator_first_actor = {
                    "actor_xy_m": actor_first.round(4).tolist(),
                    "rounded_actor_grid_cell": list(route.cell(actor_first)),
                    "rounded_cell_known_free": route.known_free(
                        route.cell(actor_first), stamp),
                    "footprint_gap_by_source_m": first_source_gaps,
                    "selected_body_low_hit_sdf_gap_m": round(first_low_gap, 4),
                    "selected_body_upper_head_sdf_gap_m": round(
                        first_upper_gap, 4),
                    "view": first_view,
                    "target_distance_gain_m": round(
                        initial_distance -
                        float(np.linalg.norm(target - actor_first)), 4)}
            rows.append({"seed": seed, "stop_time_s": stamp,
                         "planner_reason": str(state["planner_reason"]),
                         "estimated_pose_xyyaw": pose.tolist(),
                         "target_estimate_xy_m": target.tolist(),
                         "range_map_cells": len(route.cells),
                         "obstacle_points_by_source": {
                             name: len(points) for name, points in
                             obstacle_sources.items()},
                         "low_hits_after_actor_self_filter": len(low_xyz),
                         "replayed_head_upper_3d_hits": len(upper_xyz),
                         "fresh_upper_obstacles": len(upper),
                         "target_surface_points": len(state["target_surface_xyz_m"]),
                         "evaluator_route": evaluator,
                         "evaluator_first_step_actor_diagnostics":
                             evaluator_first_actor,
                         "neighbors": neighbors})
    finally:
        room.close()
    OUT.parent.mkdir(parents=True, exist_ok=True)
    (OUT.parent / "audit_stop_state_v51_source.py").write_bytes(
        SOURCE.read_bytes())
    output = {"scope": "First V51 actor stop map/3-D view state, eight-neighbor attribution and evaluator-only fixed-yaw 5cm-clear route. Truth never enters actor decisions.",
              "source_sha256": sha(SOURCE),
              "selected_design_sha256": sha(ROOT / "robot/ape-current-design.json"),
              "inputs_sha256": inputs, "rows": rows}
    OUT.write_text(json.dumps(output, indent=2) + "\n")
    for row in rows:
        print(row["seed"], row["planner_reason"],
              "model_route", row["evaluator_route"]["found"],
              "route_steps", row["evaluator_route"]["path_steps"])
        for neighbor in row["neighbors"]:
            print(neighbor["direction_cell"],
                  int(neighbor["known_free_low_range"]),
                  round(neighbor["actor_footprint_gap_m"] * 100, 1),
                  int(neighbor["view"]["ray_supported"]),
                  round(neighbor["target_distance_gain_m"] * 100, 1),
                  round(neighbor["evaluator_model_clearance_m"] * 100, 1))


if __name__ == "__main__":
    main()
