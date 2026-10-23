"""V87: can the actor's taught-object estimate replace oracle bypass direction?

Read-only replay of the exact V83 safety-stop beliefs. The actor search sees
only its target estimate, scans, depth map, selected-body model, and past
visited cells. Simulator clearance labels the chosen first edge afterward.
"""

from __future__ import annotations

import heapq
import json
import math
from pathlib import Path

import numpy as np

from audit_local_bypass_v85 import (GOAL_PROGRESS_M, GRID_M,
                                    MARGIN_M, MAX_TRAVEL_COST_M, NEIGHBORS,
                                    RADIUS_M, as_ranges)
from depth_voxel import DepthVoxelMap
from footprint_route import observed_free
from robot_geometry import RobotGeometry
from run import ROOT, Room, sha
from two_height_clearance import TwoHeightClearance


SOURCE = Path(__file__)
V82 = ROOT / "artifacts/ape-closed-loop-cove50-v82"
V83 = ROOT / "artifacts/ape-closed-loop-cove50-v83"
OUT = ROOT / "artifacts/ape-closed-loop-cove50-v87"


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
    target = np.asarray(trace["target_estimate_xy_m"], dtype=float)
    direction = target - pose[:2]
    direction /= np.linalg.norm(direction)
    blocked = tuple(int(round(float(value) / GRID_M))
                    for value in blocked_xy)
    origin = tuple(int(round(float(value) / GRID_M)) for value in pose[:2])
    visited = {tuple(node) for node in trace["actor_graph_path_cells"]
               [:trace["executed_base_steps"] + 1]}
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
            if nxt == blocked:
                continue
            end_xy = GRID_M * np.asarray(nxt, dtype=float)
            if np.linalg.norm(end_xy - pose[:2]) > RADIUS_M:
                continue
            if not (observed_free(end_xy, pose, scan11) and
                    observed_free(end_xy, pose, scan4)):
                continue
            gap = point.edge_gap(start_xy, end_xy, pose[2])
            if gap < MARGIN_M:
                continue
            new_cost = (cost + float(np.linalg.norm(end_xy - start_xy)) +
                        .002 / max(gap, .03) +
                        (.04 if nxt in visited else 0.))
            if new_cost >= costs.get(nxt, math.inf):
                continue
            costs[nxt] = new_cost
            parent[nxt] = node
            edge_gaps[nxt] = gap
            serial += 1
            heapq.heappush(queue, (new_cost, serial, nxt))
    path = []
    if goal is not None:
        path = [goal]
        while path[-1] != origin:
            path.append(parent[path[-1]])
        path.reverse()

    first_xy = None if len(path) < 2 else GRID_M * np.asarray(path[1])
    true_first_gap = None
    if first_xy is not None:
        room.reset()
        room.layout(seed, absent=False, furnished=True)
        room.model.body_pos[room.model.body("loop_table").id] = (
            .40, (1 if seed % 2 else -1) * .46, 0.)
        room.set_table_height(.37)
        room.data.qpos[:] = qpos
        room.head = float(qpos[room.head_addr])
        actual = np.asarray(trace["final_actual_pose_xyyaw_evaluator_only"])
        command = first_xy - pose[:2]
        samples = []
        for step in range(21):
            room.actual[:] = actual
            room.actual[:2] += .88 * command * step / 20
            room._forward()
            samples.append(float(room._current_obstacle_clearance()[0]))
        true_first_gap = min(samples)
    return {"seed": seed, "v83_actor_input_sha256": sha(input_path),
            "v82_trace_sha256": sha(trace_path),
            "actor_target_estimate_xy_m": target.tolist(),
            "actor_stop_pose_xyyaw": pose.tolist(),
            "actor_target_direction_xy": direction.tolist(),
            "excluded_rejected_cell": blocked,
            "expanded_cells": len(costs), "actor_goal_cell": goal,
            "actor_path_cells": path,
            "actor_path_edge_gaps_m": [round(edge_gaps[node], 4)
                                       for node in path[1:]],
            "first_edge_evaluator_min_clearance_m": (
                None if true_first_gap is None else round(true_first_gap, 4))}


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    room = Room()
    try:
        robot = RobotGeometry(room.model)
        rows = [replay(seed, room, robot) for seed in (85, 88)]
    finally:
        room.close()
    gate = all(row["actor_goal_cell"] is not None and
               row["first_edge_evaluator_min_clearance_m"] >= .05
               for row in rows)
    result = {"scope": "Frozen V85 bounded actor-only local bypass, replacing evaluator supplied progress direction with the actor's last taught-object XY estimate. No motion executed; first edge model clearance is evaluated only after selection.",
              "source_sha256": sha(SOURCE),
              "selected_design_sha256": sha(ROOT / "robot/ape-current-design.json"),
              "rows": rows,
              "target_direction_local_bypass_gate_pass": gate}
    (OUT / "actor-target-bypass-audit.json").write_text(
        json.dumps(result, indent=2) + "\n")
    (OUT / "audit_actor_target_bypass_v87_source.py").write_bytes(
        SOURCE.read_bytes())
    for row in rows:
        print(row["seed"], row["actor_goal_cell"],
              row["actor_path_cells"],
              row["first_edge_evaluator_min_clearance_m"], flush=True)
    print("actor target direction gate:", gate, flush=True)


if __name__ == "__main__":
    main()
