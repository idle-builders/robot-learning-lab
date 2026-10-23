"""V80: bounded, monitored base-side observation from exact corridor stops."""

from __future__ import annotations

import heapq
import json
import math
from pathlib import Path

import imageio.v2 as imageio
import numpy as np
from PIL import ImageDraw

from audit_active_route_coverage_v77 import route_coverage
from depth_voxel import DepthVoxelMap
from fast_range_guard import FastRangeGuard
from localization import ScanLocalizer
from robot_geometry import RobotGeometry
from run import (ROOT, Room, sensed_near_floor_range, sensed_range, sha)
from two_height_clearance import TwoHeightClearance
from upper_obstacle_side import choose_side


SOURCE = Path(__file__)
STOP_RUN = ROOT / "artifacts/ape-closed-loop-cove50-v72-stopstate"
ROUTES = ROOT / "artifacts/ape-closed-loop-cove50-v77/active-route-coverage-audit.json"
OUT = ROOT / "artifacts/ape-closed-loop-cove50-v80"
CONFIG = "chest_rgbd_range_guard_goal_map"
GRID_M = .04
GOAL_SIDE_M = .20
MAX_GRAPH_COST_M = .6
POINT_MARGIN_M = .065
HEAD_YAWS = (0., -.65, +.65)
NEIGHBORS = ((1, 0), (-1, 0), (0, 1), (0, -1),
             (1, 1), (1, -1), (-1, 1), (-1, -1))


def choose_goal(reachable: dict, pose: np.ndarray,
                target: np.ndarray, side: int) -> dict:
    forward = target - pose[:2]
    forward /= np.linalg.norm(forward)
    left = np.array((-forward[1], forward[0]))
    ideal = pose[:2] + side * GOAL_SIDE_M * left
    distance, cost, goal = min(
        ((float(np.linalg.norm(GRID_M * np.asarray(node) - ideal)),
          cost, node)
         for node, cost in reachable.items()
         if cost <= MAX_GRAPH_COST_M),
        key=lambda row: (row[0], row[1]))
    return {"side": "left" if side > 0 else "right",
            "ideal_xy_m": ideal.tolist(),
            "goal_cell": goal,
            "goal_xy_m": (GRID_M * np.asarray(goal)).tolist(),
            "snap_error_m": round(distance, 4),
            "known_graph_cost_m": round(cost, 4)}


def graph_path(reachable: dict, origin: tuple[int, int],
               goal: tuple[int, int]) -> list[tuple[int, int]]:
    queue = [(0., origin)]
    costs = {origin: 0.}
    parent = {}
    while queue:
        travel, node = heapq.heappop(queue)
        if travel > costs[node] + 1e-9:
            continue
        if node == goal:
            break
        for dx, dy in NEIGHBORS:
            nxt = node[0] + dx, node[1] + dy
            if nxt not in reachable:
                continue
            # The saved actor graph already contains viable centers; fresh
            # selected-body sensing must still clear every chosen edge.
            candidate = travel + GRID_M * math.hypot(dx, dy)
            if candidate < costs.get(nxt, math.inf):
                costs[nxt] = candidate
                parent[nxt] = node
                heapq.heappush(queue, (candidate, nxt))
    if goal not in costs:
        return []
    path = [goal]
    while path[-1] != origin:
        path.append(parent[path[-1]])
    return list(reversed(path))


def run_branch(seed: int, side: int | None, route_row: dict,
               room: Room, robot: RobotGeometry, *,
               oracle_path_cells: list[tuple[int, int]] | None = None,
               output_root: Path | None = None) -> dict:
    episode = STOP_RUN / (f"furnished_table_corridor-seed{seed}-0-"
                          f"{CONFIG}-none")
    state_path = episode / "first-stop-actor-state.npz"
    with np.load(state_path) as state:
        pose = state["estimated_pose_xyyaw"].copy()
        actual = state["actual_pose_xyyaw_evaluator_only"].copy()
        target = state["target_estimate_xy_m"].copy()
        true_target = state["target_true_xy_m_evaluator_only"].copy()
        qpos = state["actor_qpos_estimated"].copy()
        start_time = float(state["capture_time_s"])
        reachable = {(int(x), int(y)): float(cost)
                     for x, y, cost in state["reachable_costs_cells"]}
    origin = (int(round(pose[0] / GRID_M)),
              int(round(pose[1] / GRID_M)))
    if side is None and oracle_path_cells is not None:
        raise ValueError("Automatic side choice cannot use an oracle path")
    if side is None:
        choice = {"side": "actor_pending"}
        path = []
    elif oracle_path_cells is None:
        choice = choose_goal(reachable, pose, target, side)
        path = graph_path(reachable, origin, tuple(choice["goal_cell"]))
    else:
        path = list(oracle_path_cells)
        if not path or path[0] != origin:
            raise AssertionError("Oracle path must begin at actor origin")
        choice = {"side": "oracle", "goal_cell": path[-1],
                  "goal_xy_m": (GRID_M * np.asarray(path[-1])).tolist(),
                  "source": "evaluator_route_upper_bound"}
    if side is not None and not path:
        raise AssertionError("Saved actor graph cannot reach nominated goal")

    room.reset()
    room.layout(seed, absent=False, furnished=True)
    room.model.body_pos[room.model.body("loop_table").id] = (
        .40, (1 if seed % 2 else -1) * .46, 0.)
    room.set_table_height(.37)
    room.data.qpos[:] = qpos
    room.actual[:] = actual
    room.estimated[:] = pose
    room.head = float(qpos[room.head_addr])
    room.time_s = start_time
    room._forward()

    monitor11 = FastRangeGuard(band_m=.05)
    monitor4 = FastRangeGuard(band_m=.05)
    localizer = ScanLocalizer()
    voxel = DepthVoxelMap()
    fovy = float(room.model.cam_fovy[room.model.camera("lower_rgb").id])
    events = []
    scan_serial = 0
    input_serial = 0
    first_voxel = None
    first_capture_time = None
    branch_name = f"seed{seed}-" + (
        "actor" if side is None else choice["side"])
    branch_dir = (output_root or OUT) / branch_name
    branch_dir.mkdir(parents=True, exist_ok=True)
    video_path = branch_dir / "episode.mp4"

    def next_scan(height: float) -> list[float | None]:
        nonlocal scan_serial
        scan_serial += 1
        return sensed_range(room, seed, 100000 + scan_serial,
                            height_m=height)

    def guard_probe():
        first11 = next_scan(.11)
        first4 = next_scan(.04)
        monitor11.update(first11, room.estimated, room.time_s)
        monitor4.update(first4, room.estimated, room.time_s)

        def probe(substep: int, fraction: float) -> dict | None:
            now11 = next_scan(.11)
            now4 = next_scan(.04)
            torso = monitor11.update(now11, room.estimated, room.time_s)
            low = monitor4.update(now4, room.estimated, room.time_s)
            if (torso["status"] == "near_body_change" or
                    low["status"] == "near_body_change"):
                return {"source": ("low4" if low["status"] == "near_body_change"
                                   else "torso11"),
                        "torso": torso, "low": low,
                        "range_11_m": now11,
                        "range_4_m": now4,
                        "capture_time_s": room.time_s,
                        "action_fraction": fraction}
            return None
        return probe

    def add_frame(video, action: str, detail: dict) -> None:
        rgb, depth = room.capture("lower_rgb", depth=True)
        camera_pose = room.camera_pose_estimate("lower_rgb")
        actor_qpos = room.pose_data.qpos.copy()
        robot.set_state(actor_qpos)
        solids = robot.primitives()
        status = voxel.observe(
            depth, camera_pose, room.time_s, fovy_deg=fovy,
            self_hit=lambda hit: RobotGeometry.solid_distance(
                hit, solids) < 0.)
        nonlocal input_serial, first_voxel, first_capture_time
        input_serial += 1
        input_path = branch_dir / f"frame-{input_serial}.npz"
        np.savez_compressed(
            input_path, rgb=np.asarray(rgb), depth_m=depth,
            camera_origin_m=camera_pose[0],
            camera_rotation=camera_pose[1],
            estimated_pose_xyyaw=room.estimated.copy(),
            actor_qpos=actor_qpos,
            capture_time_s=room.time_s)
        if first_voxel is None:
            import copy
            first_voxel = copy.deepcopy(voxel)
            first_capture_time = room.time_s
        frame = np.concatenate((np.asarray(rgb),
                                np.asarray(room.capture_external())), axis=1)
        from PIL import Image
        annotated = Image.fromarray(frame)
        draw = ImageDraw.Draw(annotated)
        draw.rectangle((0, 0, annotated.width, 34), fill=(0, 0, 0))
        draw.text((8, 8), f"V80 seed{seed} {choice['side']} | {action} | "
                  f"{room.time_s:.2f}s", fill=(255, 255, 255))
        video.append_data(np.asarray(annotated))
        events.append({"action": action, "time_s": round(room.time_s, 3),
                       "estimated_pose_xyyaw": room.estimated.tolist(),
                       "actor_input_sha256": sha(input_path),
                       "depth_rays_used": status["depth_rays_used"],
                       "voxel_free": len(voxel.free),
                       "voxel_occupied": len(voxel.occupied),
                       "detail": detail,
                       "evaluator_actual_pose_xyyaw": room.actual.tolist(),
                       "evaluator_clearance_m": round(
                           room._current_obstacle_clearance()[0], 4)})

    status = "not_started"
    side_cue = None
    start_scan = next_scan(.11)
    localization = localizer.initialize(start_scan, room.estimated)
    with imageio.get_writer(video_path, fps=4, codec="libx264", quality=7) as video:
        room.look(0., "lower_rgb", safety_probe=guard_probe())
        add_frame(video, "initial_head_depth", {"localizer": localization})
        if room.safety_stop_observation is not None:
            status = "initial_look_range_alarm"
        else:
            status = "moving"
        if side is None and status == "moving":
            side_cue = choose_side(voxel, room.estimated, target,
                                   room.time_s)
            nominated = side_cue["side"]
            if nominated is None:
                status = "ambiguous_actor_side_stop"
            else:
                choice = choose_goal(reachable, room.estimated, target,
                                     1 if nominated == "left" else -1)
                path = graph_path(reachable, origin,
                                  tuple(choice["goal_cell"]))
                if not path:
                    status = "no_actor_side_route"
            events.append({"action": "actor_side_decision",
                           "time_s": round(room.time_s, 3),
                           "side_cue": side_cue,
                           "viewpoint_choice": choice,
                           "outcome": status})
        for step_index, node in enumerate(path[1:], start=1):
            if status != "moving":
                break
            desired_xy = GRID_M * np.asarray(node, dtype=float)
            delta = desired_xy - room.estimated[:2]
            if np.linalg.norm(delta) > .081:
                status = "waypoint_command_unbounded"
                break
            scan11 = next_scan(.11)
            scan8 = next_scan(.08)
            scan4 = next_scan(.04)
            valid_count = sum(value is not None for value in scan11)
            if valid_count < 10:
                status = "range_coverage_stop"
                break
            room.camera_pose_estimate("lower_rgb")
            clearance = TwoHeightClearance(
                robot, room.pose_data.qpos.copy(), room.estimated,
                scan11, scan4, voxel, room.time_s, range_8=scan8)
            predicted_gap = clearance.edge_gap(
                room.estimated[:2], desired_xy, room.estimated[2])
            if predicted_gap < POINT_MARGIN_M:
                status = "actor_point_clearance_stop"
                events.append({"action": status, "step_index": step_index,
                               "predicted_gap_m": round(predicted_gap, 4),
                               "desired_xy_m": desired_xy.tolist(),
                               "range_11_m": scan11,
                               "range_8_m": scan8,
                               "range_4_m": scan4})
                break
            room.move(float(delta[0]), float(delta[1]), slip=.88,
                      safety_probe=guard_probe())
            if room.safety_stop_observation is not None:
                status = "move_range_alarm"
            elif room.path_min_clearance[0] < 0:
                status = "evaluator_model_overlap"
            new_scan = next_scan(.11)
            if status == "moving":
                corrected, localization = localizer.correct(
                    new_scan, room.estimated)
                if localization["status"] != "registered":
                    status = "localization_uncertain_stop"
                else:
                    room.estimated[:] = corrected
            add_frame(video, "guarded_base_step", {
                "step_index": step_index,
                "desired_xy_m": desired_xy.tolist(),
                "command_delta_xy_m": delta.tolist(),
                "predicted_3d_point_gap_m": round(predicted_gap, 4),
                "range_11_m": scan11,
                "range_8_m": scan8,
                "range_4_m": scan4,
                "localizer": localization,
                "outcome": status,
                "range_alarm": room.safety_stop_observation})
            if status != "moving":
                break
            room.look(0., "lower_rgb", safety_probe=guard_probe())
            if room.safety_stop_observation is not None:
                status = "look_range_alarm"
            add_frame(video, "reobserve_after_step", {
                "step_index": step_index, "outcome": status,
                "range_alarm": room.safety_stop_observation})
        if status == "moving":
            status = "viewpoint_reached"
            for head_yaw in HEAD_YAWS[1:]:
                room.look(head_yaw, "lower_rgb", safety_probe=guard_probe())
                if room.safety_stop_observation is not None:
                    status = "endpoint_look_range_alarm"
                add_frame(video, "endpoint_head_sweep", {
                    "head_yaw_rad": head_yaw, "outcome": status,
                    "range_alarm": room.safety_stop_observation})
                if status != "viewpoint_reached":
                    break

    route = np.asarray(route_row["reference_route"]["path_xy_m"], dtype=float)
    route_actual = np.asarray(route_row["reference_start_actual_xyyaw"])
    route_pose = np.asarray(route_row["actor_estimated_xyyaw"])
    actor_path = np.column_stack((route_pose[:2] + route - route_actual[:2],
                                  np.full(len(route), route_pose[2])))
    nearest = int(np.argmin(np.linalg.norm(
        actor_path[:, :2] - room.estimated[:2], axis=1)))
    downstream = actor_path[nearest:]
    baseline = route_coverage(robot, qpos, first_voxel, downstream,
                              first_capture_time)["first_eight"]
    final = route_coverage(robot, qpos, voxel, downstream,
                           room.time_s)["first_eight"]
    min_clearance = min(room.path_min_clearance[0],
                        room._current_obstacle_clearance()[0])
    unknown_reduction = (baseline["unknown_fraction"] -
                         final["unknown_fraction"])
    result = {"seed": seed, "side": choice["side"],
              "actor_side_cue": side_cue,
              "stop_state_sha256": sha(state_path),
              "actor_viewpoint_choice": choice,
              "actor_graph_path_cells": path,
              "oracle_route_supplied": oracle_path_cells is not None,
              "status": status,
              "executed_base_steps": sum(
                  event["action"] == "guarded_base_step" for event in events),
              "evaluator_min_path_clearance_m": round(min_clearance, 4),
              "evaluator_model_overlap": min_clearance < 0.,
              "final_estimated_pose_xyyaw": room.estimated.tolist(),
              "target_estimate_xy_m": target.tolist(),
              "final_target_gap_m": round(float(np.linalg.norm(
                  room.estimated[:2] - target)), 4),
              "evaluator_true_target_gap_m": round(float(np.linalg.norm(
                  room.actual[:2] - true_target)), 4),
              "final_actual_pose_xyyaw_evaluator_only": room.actual.tolist(),
              "nearest_reference_route_step_evaluator_only": nearest,
              "initial_downstream_coverage_evaluator_only": baseline,
              "final_downstream_coverage_evaluator_only": final,
              "unknown_fraction_reduction_evaluator_only": round(
                  unknown_reduction, 4),
              "events": events, "video_sha256": sha(video_path)}
    (branch_dir / "trace.json").write_text(json.dumps(result, indent=2) + "\n")
    print(seed, choice["side"], status,
          "steps", result["executed_base_steps"],
          "min", result["evaluator_min_path_clearance_m"],
          "unknown", baseline["unknown_fraction"],
          "->", final["unknown_fraction"], flush=True)
    return result


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    route_audit = json.loads(ROUTES.read_text())
    route_by_seed = {int(row["seed"]): row for row in route_audit["rows"]}
    room = Room()
    try:
        robot = RobotGeometry(room.model)
        rows = [run_branch(seed, side, route_by_seed[seed], room, robot)
                for seed in (85, 88) for side in (-1, 1)]
    finally:
        room.close()
    safe = all(not row["evaluator_model_overlap"] for row in rows)
    useful = all(any(row["seed"] == seed and
                     row["status"] == "viewpoint_reached" and
                     row["evaluator_min_path_clearance_m"] >= .05 and
                     row["unknown_fraction_reduction_evaluator_only"] >= .10
                     for row in rows)
                 for seed in (85, 88))
    output = {"scope": "Four live kinematic base-side look branches from exact V72 stops. Actor chooses reachable side goals, uses fresh 4/8/11cm selected-body prechecks, 10Hz 4/11cm alarms and scan localization; V77 truth route only evaluates post-action depth coverage. Native gait/stop unqualified.",
              "source_sha256": sha(SOURCE),
              "v77_audit_sha256": sha(ROUTES),
              "selected_design_sha256": sha(ROOT / "robot/ape-current-design.json"),
              "rows": [{key: value for key, value in row.items()
                        if key != "events"} for row in rows],
              "bounded_side_look_gate_pass": safe and useful}
    (OUT / "guarded-side-look-audit.json").write_text(
        json.dumps(output, indent=2) + "\n")
    (OUT / "run_guarded_side_look_v80_source.py").write_bytes(
        SOURCE.read_bytes())
    print("bounded side look gate:", output["bounded_side_look_gate_pass"])


if __name__ == "__main__":
    main()
