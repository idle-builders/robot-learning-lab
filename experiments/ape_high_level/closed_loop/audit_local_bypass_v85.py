"""V85: bounded actor-only route around a rejected next waypoint."""

from __future__ import annotations

import heapq
import json
import math
from pathlib import Path

import numpy as np

from depth_voxel import DepthVoxelMap
from footprint_route import observed_free
from robot_geometry import RobotGeometry
from run import ROOT, Room, sha
from two_height_clearance import TwoHeightClearance


SOURCE = Path(__file__)
V82 = ROOT / "artifacts/ape-closed-loop-cove50-v82"
V83 = ROOT / "artifacts/ape-closed-loop-cove50-v83"
OUT = ROOT / "artifacts/ape-closed-loop-cove50-v85"
GRID_M = .04
MARGIN_M = .065
RADIUS_M = .35
GOAL_PROGRESS_M = .12
MAX_TRAVEL_COST_M = .8
NEIGHBORS = ((1, 0), (-1, 0), (0, 1), (0, -1),
             (1, 1), (1, -1), (-1, 1), (-1, -1))


def as_ranges(values: np.ndarray) -> list[float | None]:
    return [None if np.isnan(value) else float(value) for value in values]


def replay(seed: int, room: Room, robot: RobotGeometry) -> dict:
    input_path = V83 / f"seed{seed}-actor-stop-input.npz"
    trace_path = V82 / f"seed{seed}-oracle/trace.json"
    trace = json.loads(trace_path.read_text())
    with np.load(input_path) as saved:
        voxel = DepthVoxelMap()
        voxel.free = {(int(x), int(y), int(z)): float(stamp)
                      for x, y, z, stamp in saved["voxel_free"]}
        voxel.occupied = {(int(x), int(y), int(z)): float(stamp)
                          for x, y, z, stamp in saved["voxel_occupied"]}
        scan11 = as_ranges(saved["range_11_m"])
        scan8 = as_ranges(saved["range_8_m"])
        scan4 = as_ranges(saved["range_4_m"])
        qpos = saved["actor_qpos"].copy()
        pose = saved["estimated_pose_xyyaw"].copy()
        stamp = float(saved["capture_time_s"])
        blocked_xy = saved["blocked_next_xy_m"].copy()
    blocked_cell = tuple(int(round(float(value) / GRID_M))
                         for value in blocked_xy)
    origin = tuple(int(round(float(value) / GRID_M))
                   for value in pose[:2])
    direction = blocked_xy - pose[:2]
    direction /= np.linalg.norm(direction)
    visited = {tuple(node) for node in
               trace["actor_graph_path_cells"][:trace["executed_base_steps"] + 1]}
    point = TwoHeightClearance(robot, qpos, pose, scan11, scan4,
                               voxel, stamp, range_8=scan8)

    queue = [(0., 0, origin)]
    costs = {origin: 0.}
    parent = {}
    edge_gaps = {}
    serial = 0
    goal = None
    while queue and len(costs) < 500:
        cost, _, node = heapq.heappop(queue)
        if cost > costs[node] + 1e-9 or cost > MAX_TRAVEL_COST_M:
            continue
        node_xy = GRID_M * np.asarray(node, dtype=float)
        if (node != origin and
                float((node_xy - pose[:2]) @ direction) >= GOAL_PROGRESS_M):
            goal = node
            break
        start_xy = pose[:2] if node == origin else node_xy
        for dx, dy in NEIGHBORS:
            nxt = node[0] + dx, node[1] + dy
            if nxt == blocked_cell:
                continue
            end_xy = GRID_M * np.asarray(nxt, dtype=float)
            if np.linalg.norm(end_xy - pose[:2]) > RADIUS_M:
                continue
            if not (observed_free(end_xy, pose, scan11) and
                    observed_free(end_xy, pose, scan4)):
                continue
            edge_gap = point.edge_gap(start_xy, end_xy, pose[2])
            if edge_gap < MARGIN_M:
                continue
            edge_cost = (float(np.linalg.norm(end_xy - start_xy)) +
                         .002 / max(edge_gap, .03) +
                         (.04 if nxt in visited else 0.))
            new_cost = cost + edge_cost
            if new_cost >= costs.get(nxt, math.inf):
                continue
            costs[nxt] = new_cost
            parent[nxt] = node
            edge_gaps[nxt] = edge_gap
            serial += 1
            heapq.heappush(queue, (new_cost, serial, nxt))
    path = []
    if goal is not None:
        path = [goal]
        while path[-1] != origin:
            path.append(parent[path[-1]])
        path.reverse()
    first_xy = (None if len(path) < 2 else
                GRID_M * np.asarray(path[1], dtype=float))

    room.reset()
    room.layout(seed, absent=False, furnished=True)
    room.model.body_pos[room.model.body("loop_table").id] = (
        .40, (1 if seed % 2 else -1) * .46, 0.)
    room.set_table_height(.37)
    room.data.qpos[:] = qpos
    room.head = float(qpos[room.head_addr])
    room.estimated[:] = pose
    actual = np.asarray(trace["final_actual_pose_xyyaw_evaluator_only"],
                        dtype=float)
    true_first_gap = None
    if first_xy is not None:
        command = first_xy - pose[:2]
        samples = []
        for step in range(21):
            room.actual[:] = actual
            room.actual[:2] += .88 * command * step / 20
            room._forward()
            samples.append(float(room._current_obstacle_clearance()[0]))
        true_first_gap = min(samples)
    return {"seed": seed,
            "v82_trace_sha256": sha(trace_path),
            "v83_actor_input_sha256": sha(input_path),
            "actor_estimated_pose_xyyaw": pose.tolist(),
            "blocked_cell": blocked_cell,
            "desired_progress_direction_xy": direction.tolist(),
            "expanded_cells": len(costs),
            "actor_goal_cell": goal,
            "actor_goal_progress_m": (None if goal is None else round(
                float((GRID_M * np.asarray(goal) - pose[:2]) @ direction), 4)),
            "actor_route_cost_m": (None if goal is None else round(
                costs[goal], 4)),
            "actor_path_cells": path,
            "actor_path_edge_gaps_m": [round(edge_gaps[node], 4)
                                       for node in path[1:]],
            "chosen_first_xy_m": (None if first_xy is None else
                                  first_xy.tolist()),
            "chosen_first_evaluator_20ms_min_clearance_m": (
                None if true_first_gap is None else round(true_first_gap, 4)),
            "local_bypass_hypothesis_pass": (
                first_xy is not None and true_first_gap >= .05)}


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    room = Room()
    try:
        robot = RobotGeometry(room.model)
        rows = [replay(seed, room, robot) for seed in (85, 88)]
    finally:
        room.close()
    result = {"scope": "Actor-only 4cm 8-neighbor bounded Dijkstra around excluded V82 blocked waypoint using saved 4/8/11cm rays, depth voxels, selected-body 6.5cm point edge rule and visited-cell penalty. Evaluator truth scores only first chosen edge after planning; no action executed.",
              "source_sha256": sha(SOURCE),
              "selected_design_sha256": sha(ROOT / "robot/ape-current-design.json"),
              "rows": rows,
              "two_scene_local_bypass_screen_pass": all(
                  row["local_bypass_hypothesis_pass"] for row in rows)}
    (OUT / "local-bypass-screen-audit.json").write_text(
        json.dumps(result, indent=2) + "\n")
    (OUT / "audit_local_bypass_v85_source.py").write_bytes(
        SOURCE.read_bytes())
    for row in rows:
        print(row["seed"], "goal", row["actor_goal_cell"],
              "path", row["actor_path_cells"],
              "true first", row[
                  "chosen_first_evaluator_20ms_min_clearance_m"], flush=True)
    print("local bypass screen:", result["two_scene_local_bypass_screen_pass"])


if __name__ == "__main__":
    main()
