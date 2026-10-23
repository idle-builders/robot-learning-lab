"""V84: actor-only two-edge local repair screen at V82 point stops."""

from __future__ import annotations

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
OUT = ROOT / "artifacts/ape-closed-loop-cove50-v84"
GRID_M = .04
MARGIN_M = .065
NEIGHBORS = ((1, 0), (-1, 0), (0, 1), (0, -1),
             (1, 1), (1, -1), (-1, 1), (-1, -1))


def as_ranges(values: np.ndarray) -> list[float | None]:
    return [None if np.isnan(value) else float(value) for value in values]


def replay(seed: int, room: Room, robot: RobotGeometry) -> dict:
    input_path = V83 / f"seed{seed}-actor-stop-input.npz"
    trace_path = V82 / f"seed{seed}-oracle/trace.json"
    trace = json.loads(trace_path.read_text())
    with np.load(input_path) as state:
        voxel = DepthVoxelMap()
        voxel.free = {(int(x), int(y), int(z)): float(stamp)
                      for x, y, z, stamp in state["voxel_free"]}
        voxel.occupied = {(int(x), int(y), int(z)): float(stamp)
                          for x, y, z, stamp in state["voxel_occupied"]}
        scan11 = as_ranges(state["range_11_m"])
        scan8 = as_ranges(state["range_8_m"])
        scan4 = as_ranges(state["range_4_m"])
        qpos = state["actor_qpos"].copy()
        pose = state["estimated_pose_xyyaw"].copy()
        time_s = float(state["capture_time_s"])
        blocked = state["blocked_next_xy_m"].copy()
    point = TwoHeightClearance(robot, qpos, pose, scan11, scan4,
                               voxel, time_s, range_8=scan8)
    origin = (int(round(pose[0] / GRID_M)),
              int(round(pose[1] / GRID_M)))
    visited = {tuple(node) for node in
               trace["actor_graph_path_cells"][:trace["executed_base_steps"] + 1]}
    candidates = []
    for dx, dy in NEIGHBORS:
        node = origin[0] + dx, origin[1] + dy
        xy = GRID_M * np.asarray(node, dtype=float)
        distance = float(np.linalg.norm(xy - pose[:2]))
        ray11 = observed_free(xy, pose, scan11)
        ray4 = observed_free(xy, pose, scan4)
        first_gap = point.edge_gap(pose[:2], xy, pose[2])
        second_gap = point.edge_gap(xy, blocked, pose[2])
        admissible = (node not in visited and distance <= .081 and
                      ray11 and ray4 and first_gap >= MARGIN_M and
                      second_gap >= MARGIN_M)
        travel = distance + float(np.linalg.norm(blocked - xy))
        candidates.append({"cell": node, "xy_m": xy.tolist(),
                           "visited": node in visited,
                           "command_distance_m": round(distance, 4),
                           "observed_free_11cm": ray11,
                           "observed_free_4cm": ray4,
                           "first_actor_gap_m": round(first_gap, 4),
                           "second_actor_gap_m": round(second_gap, 4),
                           "two_edge_travel_m": round(travel, 4),
                           "actor_admissible": bool(admissible)})
    admissible = sorted(
        (row for row in candidates if row["actor_admissible"]),
        key=lambda row: (row["two_edge_travel_m"],
                         -min(row["first_actor_gap_m"],
                              row["second_actor_gap_m"])))
    chosen = admissible[0] if admissible else None

    room.reset()
    room.layout(seed, absent=False, furnished=True)
    room.model.body_pos[room.model.body("loop_table").id] = (
        .40, (1 if seed % 2 else -1) * .46, 0.)
    room.set_table_height(.37)
    actual = np.asarray(trace["final_actual_pose_xyyaw_evaluator_only"],
                        dtype=float)
    room.data.qpos[:] = qpos
    room.estimated[:] = pose
    room.head = float(qpos[room.head_addr])
    room.time_s = time_s
    true_gap = None
    if chosen is not None:
        command = np.asarray(chosen["xy_m"]) - pose[:2]
        samples = []
        for step in range(21):
            room.actual[:] = actual
            room.actual[:2] += .88 * command * step / 20
            room._forward()
            samples.append(float(room._current_obstacle_clearance()[0]))
        true_gap = min(samples)
    return {"seed": seed, "v82_trace_sha256": sha(trace_path),
            "v83_actor_input_sha256": sha(input_path),
            "actor_pose_xyyaw": pose.tolist(),
            "blocked_next_xy_m": blocked.tolist(),
            "visited_cells": sorted(visited),
            "candidates": candidates,
            "actor_admissible_count": len(admissible),
            "actor_chosen_repair": chosen,
            "chosen_first_edge_evaluator_min_clearance_m": (
                None if true_gap is None else round(true_gap, 4)),
            "local_two_edge_repair_hypothesis_pass": (
                chosen is not None and true_gap >= .05)}


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    room = Room()
    try:
        robot = RobotGeometry(room.model)
        rows = [replay(seed, room, robot) for seed in (85, 88)]
    finally:
        room.close()
    result = {"scope": "Eight adjacent 4cm actor-only ray-cleared, frozen 6.5cm two-edge point-repair proposals from exact V82 stops, excluding visited path cells. Evaluator truth scores only the first chosen command afterward. No live repair action or guard change.",
              "source_sha256": sha(SOURCE),
              "selected_design_sha256": sha(ROOT / "robot/ape-current-design.json"),
              "rows": rows,
              "two_scene_local_repair_screen_pass": all(
                  row["local_two_edge_repair_hypothesis_pass"] for row in rows)}
    (OUT / "local-repair-screen-audit.json").write_text(
        json.dumps(result, indent=2) + "\n")
    (OUT / "audit_local_repair_v84_source.py").write_bytes(
        SOURCE.read_bytes())
    for row in rows:
        print(row["seed"], "admissible", row["actor_admissible_count"],
              "chosen", row["actor_chosen_repair"],
              "true first gap", row[
                  "chosen_first_edge_evaluator_min_clearance_m"])
    print("local repair screen:", result["two_scene_local_repair_screen_pass"])


if __name__ == "__main__":
    main()
