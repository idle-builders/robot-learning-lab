"""V98: actor-only course-along-bypass search viewpoint after V97 failure."""

from __future__ import annotations

import heapq
import json
import math
from pathlib import Path

import numpy as np

from depth_voxel import DepthVoxelMap
from footprint_route import observed_free
from robot_geometry import RobotGeometry
from run import H, ROOT, W, Room, sensed_range, sha
from self_depth_mask import SelfDepthMask
from two_height_clearance import TwoHeightClearance


SOURCE = Path(__file__)
V90 = ROOT / "artifacts/ape-closed-loop-cove50-v90"
V96 = ROOT / "artifacts/ape-closed-loop-cove50-v96"
V97 = ROOT / "artifacts/ape-closed-loop-cove50-v97"
OUT = ROOT / "artifacts/ape-closed-loop-cove50-v98"
GRID_M = .04
MARGIN_M = .065
RADIUS_M = .35
MAX_COST_M = .8
MIN_COURSE_PROGRESS_M = .18
NEIGHBORS = ((1, 0), (-1, 0), (0, 1), (0, -1),
             (1, 1), (1, -1), (-1, 1), (-1, -1))


def replay(seed: int, room: Room, robot: RobotGeometry,
           mask: SelfDepthMask, *,
           search_direction: str = "course") -> dict:
    prior_path = V96 / f"seed{seed}-repair/trace.json"
    view_path = V97 / f"seed{seed}/trace.json"
    prior = json.loads(prior_path.read_text())
    view = json.loads(view_path.read_text())
    if view["correct_target_reacquired"]:
        return {"seed": seed, "status": "already_reacquired"}
    target = np.asarray(view["actor_target_memory_xy_m"], dtype=float)
    final_path = V97 / f"seed{seed}/view-2-actor-input.npz"
    with np.load(final_path) as z:
        pose = z["estimated_pose_xyyaw"].copy()
        qpos = z["actor_qpos"].copy()
        stamp = float(z["capture_time_s"])
    actual = np.asarray(prior["evaluator_final_actual_xyyaw"])
    forward = target - pose[:2]
    forward /= np.linalg.norm(forward)
    left = np.array([-forward[1], forward[0]])
    side_name = json.loads((V90 / f"seed{seed}-actor/trace.json").read_text())[
        "side"]
    side = left if side_name == "left" else -left
    course = forward + side
    course /= np.linalg.norm(course)
    if search_direction not in ("course", "side"):
        raise ValueError("Unknown actor search direction")
    direction = course if search_direction == "course" else side
    voxel = DepthVoxelMap()
    input_paths = []
    v96_frames = sum("actor_input_sha256" in event
                     for event in prior["events"])
    all_inputs = [V96 / f"seed{seed}-repair/frame-{v96_frames}.npz"] + [
        V97 / f"seed{seed}/view-{index}-actor-input.npz"
        for index in range(3)]
    fovy = float(room.model.cam_fovy[room.model.camera("lower_rgb").id])
    for path in all_inputs:
        with np.load(path) as z:
            frame_qpos = z["actor_qpos"].copy()
            depth = z["depth_m"].copy()
            camera = (z["camera_origin_m"].copy(),
                      z["camera_rotation"].copy())
            time_s = float(z["capture_time_s"])
        filtered, _ = mask.apply(depth, frame_qpos)
        voxel.observe(filtered, camera, time_s, fovy_deg=fovy,
                      self_masked=True)
        input_paths.append(sha(path))
    room.reset()
    room.layout(seed, absent=False, furnished=True)
    room.model.body_pos[room.model.body("loop_table").id] = (
        .40, (1 if seed % 2 else -1) * .46, 0.)
    room.set_table_height(.37)
    room.data.qpos[:] = qpos
    room.actual[:] = actual
    room.estimated[:] = pose
    room.head = float(qpos[room.head_addr])
    room.time_s = stamp
    room._forward()
    scan11 = sensed_range(room, seed, 198011, height_m=.11)
    scan8 = sensed_range(room, seed, 198008, height_m=.08)
    scan4 = sensed_range(room, seed, 198004, height_m=.04)
    point = TwoHeightClearance(robot, qpos, pose, scan11, scan4,
                               voxel, stamp, range_8=scan8)
    prior_side = json.loads((V90 / f"seed{seed}-actor/trace.json").read_text())
    visited = {tuple(node) for node in prior_side["actor_graph_path_cells"]}
    visited |= {tuple(event["detail"]["plan_cells"][1]) for event in
                prior["events"] if event["action"] == "replanned_base_step"}
    origin = tuple(int(round(float(value) / GRID_M)) for value in pose[:2])
    queue = [(0., 0, origin)]
    costs = {origin: 0.}
    parent = {}
    gaps = {}
    serial = 0
    goal = None
    while queue and len(costs) < 500:
        cost, _, node = heapq.heappop(queue)
        if cost > costs[node] + 1e-9 or cost > MAX_COST_M:
            continue
        node_xy = GRID_M * np.asarray(node, dtype=float)
        if (node != origin and
                float((node_xy - pose[:2]) @ direction) >=
                MIN_COURSE_PROGRESS_M):
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
            gaps[nxt] = gap
            serial += 1
            heapq.heappush(queue, (new_cost, serial, nxt))
    path = []
    if goal is not None:
        path = [goal]
        while path[-1] != origin:
            path.append(parent[path[-1]])
        path.reverse()
    first_xy = None if len(path) < 2 else GRID_M * np.asarray(path[1])
    true_first = None
    if first_xy is not None:
        command = first_xy - pose[:2]
        samples = []
        for step in range(21):
            room.actual[:] = actual
            room.actual[:2] += .88 * command * step / 20
            room._forward()
            samples.append(float(room._current_obstacle_clearance()[0]))
        true_first = min(samples)
    return {"seed": seed, "status": "screened",
            "v96_trace_sha256": sha(prior_path),
            "v97_trace_sha256": sha(view_path),
            "actor_input_sha256": input_paths,
            "actor_pose_xyyaw": pose.tolist(),
            "actor_target_memory_xy_m": target.tolist(),
            "actor_side": side_name,
            "actor_course_unit_xy": course.tolist(),
            "actor_side_unit_xy": side.tolist(),
            "actor_search_direction": search_direction,
            "actor_voxel_occupied": len(voxel.occupied),
            "actor_point_source_counts": point.source_counts(),
            "expanded_cells": len(costs),
            "max_reachable_course_progress_m": round(max(
                float((GRID_M * np.asarray(node) - pose[:2]) @ course)
                for node in costs), 4),
            "max_reachable_side_progress_m": round(max(
                float((GRID_M * np.asarray(node) - pose[:2]) @ side)
                for node in costs), 4),
            "actor_goal_cell": goal,
            "actor_course_progress_m": (None if goal is None else round(
                float((GRID_M * np.asarray(goal) - pose[:2]) @ direction), 4)),
            "actor_goal_target_memory_gap_m": (None if goal is None else round(
                float(np.linalg.norm(target - GRID_M * np.asarray(goal))), 4)),
            "actor_path_cells": path,
            "actor_path_edge_gaps_m": [round(gaps[node], 4)
                                       for node in path[1:]],
            "first_edge_evaluator_min_clearance_m": (
                None if true_first is None else round(true_first, 4))}


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    room = Room()
    mask = SelfDepthMask(room.model, camera="lower_rgb",
                         width=W, height=H)
    try:
        robot = RobotGeometry(room.model)
        row = replay(85, room, robot, mask)
    finally:
        mask.close()
        room.close()
    gate = (row["status"] == "screened" and
            row["actor_goal_cell"] is not None and
            row["actor_course_progress_m"] >= MIN_COURSE_PROGRESS_M and
            row["first_edge_evaluator_min_clearance_m"] >= .05)
    result = {"scope": "Read-only actor-side search viewpoint for V97 seed85 target-not-reacquired branch. Course combines target memory bearing and committed side, using fresh4/8/11cm scans, self-masked near head depth,6.5cm selected-body point edge search. Evaluator only scores chosen first edge after selection. No object acceptance or motion.",
              "source_sha256": sha(SOURCE),
              "selected_design_sha256": sha(ROOT / "robot/ape-current-design.json"),
              "row": row,
              "post_detour_viewpoint_screen_pass": gate}
    (OUT / "post-detour-viewpoint-audit.json").write_text(
        json.dumps(result, indent=2) + "\n")
    (OUT / "audit_post_detour_viewpoint_v98_source.py").write_bytes(
        SOURCE.read_bytes())
    print("goal", row["actor_goal_cell"], "path", row["actor_path_cells"],
          "course progress", row["actor_course_progress_m"],
          "target memory gap", row["actor_goal_target_memory_gap_m"],
          "first true clearance", row[
              "first_edge_evaluator_min_clearance_m"], flush=True)
    print("post detour viewpoint screen:", gate, flush=True)


if __name__ == "__main__":
    main()
