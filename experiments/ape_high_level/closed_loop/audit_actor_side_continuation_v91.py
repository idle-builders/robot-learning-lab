"""V91: actor-only outward graph route from V90 live side-look endpoint."""

from __future__ import annotations

import heapq
import json
import math
from pathlib import Path

import numpy as np

from depth_voxel import DepthVoxelMap
from footprint_route import observed_free
from robot_geometry import RobotGeometry
from run import ROOT, Room, sensed_range, sha
from two_height_clearance import TwoHeightClearance
from self_depth_mask import SelfDepthMask


SOURCE = Path(__file__)
V90 = ROOT / "artifacts/ape-closed-loop-cove50-v90"
OUT = ROOT / "artifacts/ape-closed-loop-cove50-v91"
GRID_M = .04
MARGIN_M = .065
RADIUS_M = .35
MAX_TRAVEL_COST_M = .8
MIN_SIDE_PROGRESS_M = .18
NEIGHBORS = ((1, 0), (-1, 0), (0, 1), (0, -1),
             (1, 1), (1, -1), (-1, 1), (-1, -1))


def replay(seed: int, room: Room, robot: RobotGeometry,
           *, self_mask: SelfDepthMask | None = None,
           allow_near_masked_depth: bool = False) -> dict:
    if allow_near_masked_depth and self_mask is None:
        raise ValueError("Near depth needs a calibrated pixel self mask")
    branch = V90 / f"seed{seed}-actor"
    trace_path = branch / "trace.json"
    trace = json.loads(trace_path.read_text())
    if trace["status"] != "viewpoint_reached":
        return {"seed": seed, "status": "viewpoint_not_reached"}
    pose = np.asarray(trace["final_estimated_pose_xyyaw"], dtype=float)
    actual = np.asarray(trace["final_actual_pose_xyyaw_evaluator_only"],
                        dtype=float)
    target = np.asarray(trace["target_estimate_xy_m"], dtype=float)
    forward = target - pose[:2]
    forward /= np.linalg.norm(forward)
    left = np.array([-forward[1], forward[0]])
    side = left if trace["side"] == "left" else -left
    visited = {tuple(node) for node in trace["actor_graph_path_cells"]}
    voxel = DepthVoxelMap()
    fovy = float(room.model.cam_fovy[room.model.camera("lower_rgb").id])
    last_qpos = None
    last_time = None
    inputs = []
    mask_totals = {"predicted_self_pixels": 0,
                   "masked_self_pixels": 0,
                   "closer_surface_pixels_preserved": 0}
    for index in range(1, 14):
        path = branch / f"frame-{index}.npz"
        with np.load(path) as z:
            qpos = z["actor_qpos"].copy()
            depth = z["depth_m"].copy()
            camera = (z["camera_origin_m"].copy(),
                      z["camera_rotation"].copy())
            time_s = float(z["capture_time_s"])
        robot.set_state(qpos)
        solids = robot.primitives()
        if self_mask is not None:
            depth, mask_info = self_mask.apply(depth, qpos)
            for key in mask_totals:
                mask_totals[key] += mask_info[key]
        voxel.observe(depth, camera, time_s, fovy_deg=fovy,
                      self_hit=(None if self_mask is not None else
                                lambda hit: RobotGeometry.solid_distance(
                                    hit, solids) < 0.),
                      self_masked=allow_near_masked_depth)
        last_qpos, last_time = qpos, time_s
        inputs.append(sha(path))
    assert last_qpos is not None and last_time is not None
    room.reset()
    room.layout(seed, absent=False, furnished=True)
    room.model.body_pos[room.model.body("loop_table").id] = (
        .40, (1 if seed % 2 else -1) * .46, 0.)
    room.set_table_height(.37)
    room.data.qpos[:] = last_qpos
    room.actual[:] = actual
    room.estimated[:] = pose
    room.head = float(last_qpos[room.head_addr])
    room.time_s = last_time
    room._forward()
    scan11 = sensed_range(room, seed, 191100, height_m=.11)
    scan8 = sensed_range(room, seed, 190800, height_m=.08)
    scan4 = sensed_range(room, seed, 190400, height_m=.04)
    point = TwoHeightClearance(robot, last_qpos, pose, scan11, scan4,
                               voxel, last_time, range_8=scan8)
    origin = tuple(int(round(float(value) / GRID_M)) for value in pose[:2])
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
                float((node_xy - pose[:2]) @ side) >= MIN_SIDE_PROGRESS_M):
            goal = node
            break
        start_xy = pose[:2] if node == origin else node_xy
        for dx, dy in NEIGHBORS:
            nxt = node[0] + dx, node[1] + dy
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
    neighbor_diagnostics = []
    for dx, dy in NEIGHBORS:
        node = (origin[0] + dx, origin[1] + dy)
        end_xy = GRID_M * np.asarray(node, dtype=float)
        free11 = observed_free(end_xy, pose, scan11)
        free4 = observed_free(end_xy, pose, scan4)
        actor_gap = point.edge_gap(pose[:2], end_xy, pose[2])
        by_source = {}
        for name, points in point.sources.items():
            source_gap = math.inf
            for fraction in (.25, .5, .75, 1.):
                future = pose.copy()
                future[:2] = pose[:2] + fraction * (end_xy - pose[:2])
                robot.set_state(robot.pose_qpos(last_qpos, future))
                if len(points):
                    source_gap = min(source_gap, float(np.min(
                        RobotGeometry.solid_distances(
                            points, robot.primitives()))))
            by_source[name] = (None if not math.isfinite(source_gap)
                               else round(source_gap, 4))
        command = end_xy - pose[:2]
        samples = []
        for step in range(21):
            room.actual[:] = actual
            room.actual[:2] += .88 * command * step / 20
            room._forward()
            samples.append(float(room._current_obstacle_clearance()[0]))
        neighbor_diagnostics.append({
            "cell": node, "observed_free_11": free11,
            "observed_free_4": free4,
            "actor_point_gap_m": round(actor_gap, 4),
            "actor_source_gaps_m": by_source,
            "evaluator_min_20ms_model_clearance_m": round(min(samples), 4)})
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
    return {"seed": seed, "v90_trace_sha256": sha(trace_path),
            "v90_frame_sha256": inputs,
            "actor_side": trace["side"],
            "actor_start_pose_xyyaw": pose.tolist(),
            "actor_target_estimate_xy_m": target.tolist(),
            "actor_side_unit_xy": side.tolist(),
            "actor_voxel_free": len(voxel.free),
            "actor_voxel_occupied": len(voxel.occupied),
            "actor_self_depth_mask_totals": (
                mask_totals if self_mask is not None else None),
            "actor_range_source_counts": point.source_counts(),
            "expanded_cells": len(costs),
            "actor_goal_cell": goal,
            "actor_side_progress_m": (None if goal is None else round(
                float((GRID_M * np.asarray(goal) - pose[:2]) @ side), 4)),
            "actor_goal_target_gap_m": (None if goal is None else round(
                float(np.linalg.norm(target - GRID_M * np.asarray(goal))), 4)),
            "actor_path_cells": path,
            "actor_path_edge_gaps_m": [round(edge_gaps[node], 4)
                                       for node in path[1:]],
            "neighbor_diagnostics_evaluator_only": neighbor_diagnostics,
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
               row["actor_side_progress_m"] >= MIN_SIDE_PROGRESS_M and
               row["first_edge_evaluator_min_clearance_m"] >= .05
               for row in rows)
    result = {"scope": "Actor-only bounded outward chosen-side graph search at V90 live side-look endpoints, from saved 3D head-depth, fresh 4/8/11cm scans and actor target estimate; no oracle progress/blocked waypoint. Evaluator geometry labels only selected first edge after choice. No motion executed.",
              "source_sha256": sha(SOURCE),
              "selected_design_sha256": sha(ROOT / "robot/ape-current-design.json"),
              "rows": rows,
              "actor_side_continuation_screen_pass": gate}
    (OUT / "actor-side-continuation-audit.json").write_text(
        json.dumps(result, indent=2) + "\n")
    (OUT / "audit_actor_side_continuation_v91_source.py").write_bytes(
        SOURCE.read_bytes())
    for row in rows:
        print(row["seed"], "goal", row["actor_goal_cell"],
              "side progress", row["actor_side_progress_m"],
              "target gap", row["actor_goal_target_gap_m"],
              "first clearance", row[
                  "first_edge_evaluator_min_clearance_m"], flush=True)
    print("actor side continuation screen:", gate, flush=True)


if __name__ == "__main__":
    main()
