"""V86: receding-horizon actor-only local bypass from V82 stops."""

from __future__ import annotations

import heapq
import json
import math
from pathlib import Path

import imageio.v2 as imageio
import numpy as np
from PIL import Image, ImageDraw

from depth_voxel import DepthVoxelMap
from fast_range_guard import FastRangeGuard
from footprint_route import observed_free
from localization import ScanLocalizer
from robot_geometry import RobotGeometry
from run import ROOT, Room, sensed_range, sha
from two_height_clearance import TwoHeightClearance
from self_depth_mask import SelfDepthMask


SOURCE = Path(__file__)
V82 = ROOT / "artifacts/ape-closed-loop-cove50-v82"
V83 = ROOT / "artifacts/ape-closed-loop-cove50-v83"
V85 = ROOT / "artifacts/ape-closed-loop-cove50-v85/local-bypass-screen-audit.json"
OUT = ROOT / "artifacts/ape-closed-loop-cove50-v86"
GRID_M = .04
MARGIN_M = .065
RADIUS_M = .35
MAX_TRAVEL_COST_M = .8
MAX_MOVES = 10
NEIGHBORS = ((1, 0), (-1, 0), (0, 1), (0, -1),
             (1, 1), (1, -1), (-1, 1), (-1, -1))


def plan(pose: np.ndarray, anchor_xy: np.ndarray,
         goal: tuple[int, int], blocked: tuple[int, int] | None,
         visited: set[tuple[int, int]],
         scan11: list[float | None], scan4: list[float | None],
         point: TwoHeightClearance, *,
         clearance_cost_weight: float = .002,
         physical_travel_cap: bool = False,
         use_low_range_for_planning: bool = True
         ) -> tuple[list[tuple[int, int]],
                                             list[float], int]:
    origin = (int(round(pose[0] / GRID_M)),
              int(round(pose[1] / GRID_M)))
    queue = [(0., 0, origin)]
    costs = {origin: 0.}
    travel = {origin: 0.}
    parent = {}
    edge_gaps = {}
    serial = 0
    found = False
    while queue and len(costs) < 500:
        cost, _, node = heapq.heappop(queue)
        if (cost > costs[node] + 1e-9 or
                (travel[node] if physical_travel_cap else cost) >
                MAX_TRAVEL_COST_M):
            continue
        if node == goal:
            found = True
            break
        start_xy = (pose[:2] if node == origin else
                    GRID_M * np.asarray(node, dtype=float))
        for dx, dy in NEIGHBORS:
            nxt = node[0] + dx, node[1] + dy
            if blocked is not None and nxt == blocked:
                continue
            end_xy = GRID_M * np.asarray(nxt, dtype=float)
            if np.linalg.norm(end_xy - anchor_xy) > RADIUS_M:
                continue
            if not (observed_free(end_xy, pose, scan11) and
                    (not use_low_range_for_planning or
                     observed_free(end_xy, pose, scan4))):
                continue
            gap = point.edge_gap(start_xy, end_xy, pose[2])
            if gap < MARGIN_M:
                continue
            step_length = float(np.linalg.norm(end_xy - start_xy))
            next_travel = travel[node] + step_length
            if physical_travel_cap and next_travel > MAX_TRAVEL_COST_M:
                continue
            new_cost = (cost + step_length +
                        clearance_cost_weight / max(gap, .03) +
                        (.04 if nxt in visited else 0.))
            if new_cost >= costs.get(nxt, math.inf):
                continue
            costs[nxt] = new_cost
            travel[nxt] = next_travel
            parent[nxt] = node
            edge_gaps[nxt] = gap
            serial += 1
            heapq.heappush(queue, (new_cost, serial, nxt))
    if not found:
        return [], [], len(costs)
    path = [goal]
    while path[-1] != origin:
        path.append(parent[path[-1]])
    path.reverse()
    return path, [edge_gaps[node] for node in path[1:]], len(costs)


def run(seed: int, goal_row: dict, room: Room,
        robot: RobotGeometry, *, output_root: Path = OUT,
        video_label: str = "V86", goal_audit_path: Path = V85,
        initial: dict | None = None,
        self_mask: SelfDepthMask | None = None,
        allow_near_masked_depth: bool = False,
        max_moves: int = MAX_MOVES,
        motion_consistency_stop: bool = False,
        clearance_cost_weight: float = .002,
        physical_travel_cap: bool = False,
        committed_path_cells: list[tuple[int, int]] | None = None,
        max_committed_fallback_moves: int = 0,
        head_low_planning: bool = False) -> dict:
    if allow_near_masked_depth and self_mask is None:
        raise ValueError("Near depth needs a calibrated pixel self mask")
    if initial is None:
        input_path = V83 / f"seed{seed}-actor-stop-input.npz"
        oracle_trace_path = V82 / f"seed{seed}-oracle/trace.json"
        oracle_trace = json.loads(oracle_trace_path.read_text())
        with np.load(input_path) as saved:
            voxel = DepthVoxelMap()
            voxel.free = {(int(x), int(y), int(z)): float(stamp)
                          for x, y, z, stamp in saved["voxel_free"]}
            voxel.occupied = {(int(x), int(y), int(z)): float(stamp)
                              for x, y, z, stamp in saved["voxel_occupied"]}
            qpos = saved["actor_qpos"].copy()
            pose = saved["estimated_pose_xyyaw"].copy()
            start_time = float(saved["capture_time_s"])
        actual = np.asarray(
            oracle_trace["final_actual_pose_xyyaw_evaluator_only"])
        visited = {tuple(node) for node in oracle_trace[
            "actor_graph_path_cells"][:oracle_trace["executed_base_steps"] + 1]}
        provenance = {"v82_trace_sha256": sha(oracle_trace_path),
                      "v83_actor_input_sha256": sha(input_path)}
    else:
        voxel = initial["voxel"]
        qpos = initial["actor_qpos"].copy()
        pose = initial["estimated_pose_xyyaw"].copy()
        actual = initial["actual_pose_xyyaw_evaluator_only"].copy()
        start_time = float(initial["capture_time_s"])
        visited = {tuple(node) for node in initial["visited_cells"]}
        provenance = dict(initial["provenance"])
    anchor_xy = pose[:2].copy()
    goal = tuple(goal_row["actor_goal_cell"])
    blocked = (None if goal_row["blocked_cell"] is None else
               tuple(goal_row["blocked_cell"]))
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
    fovy = float(room.model.cam_fovy[room.model.camera("lower_rgb").id])
    low_guard = FastRangeGuard(band_m=.05)
    torso_guard = FastRangeGuard(band_m=.05)
    localizer = ScanLocalizer()
    scan_serial = 0
    frame_serial = 0
    inconsistent_motion = 0
    committed_fallback_moves = 0
    committed = ([] if committed_path_cells is None else
                 [tuple(cell) for cell in committed_path_cells])
    events = []
    branch = output_root / f"seed{seed}-repair"
    branch.mkdir(parents=True, exist_ok=True)
    video_path = branch / "episode.mp4"

    def sense(height: float) -> list[float | None]:
        nonlocal scan_serial
        scan_serial += 1
        return sensed_range(room, seed, 160000 + scan_serial,
                            height_m=height)

    def guard_probe():
        torso_guard.update(sense(.11), room.estimated, room.time_s)
        low_guard.update(sense(.04), room.estimated, room.time_s)

        def probe(substep: int, fraction: float) -> dict | None:
            scan11 = sense(.11)
            scan4 = sense(.04)
            upper = torso_guard.update(scan11, room.estimated, room.time_s)
            lower = low_guard.update(scan4, room.estimated, room.time_s)
            if (upper["status"] == "near_body_change" or
                    lower["status"] == "near_body_change"):
                return {"source": ("low4" if lower["status"] ==
                                   "near_body_change" else "torso11"),
                        "range_11_m": scan11, "range_4_m": scan4,
                        "torso": upper, "low": lower,
                        "capture_time_s": room.time_s,
                        "action_fraction": fraction}
            return None
        return probe

    def frame(video, action: str, detail: dict) -> None:
        nonlocal frame_serial
        rgb, depth = room.capture("lower_rgb", depth=True)
        camera_pose = room.camera_pose_estimate("lower_rgb")
        actor_qpos = room.pose_data.qpos.copy()
        robot.set_state(actor_qpos)
        solids = robot.primitives()
        filtered_depth = depth
        mask_info = None
        if self_mask is not None:
            filtered_depth, mask_info = self_mask.apply(depth, actor_qpos)
        observed = voxel.observe(
            filtered_depth, camera_pose, room.time_s, fovy_deg=fovy,
            self_hit=(None if self_mask is not None else
                      lambda hit: RobotGeometry.solid_distance(
                          hit, solids) < 0.),
            self_masked=allow_near_masked_depth)
        frame_serial += 1
        source = branch / f"frame-{frame_serial}.npz"
        np.savez_compressed(
            source, rgb=np.asarray(rgb), depth_m=depth,
            camera_origin_m=camera_pose[0],
            camera_rotation=camera_pose[1],
            actor_qpos=actor_qpos,
            estimated_pose_xyyaw=room.estimated.copy(),
            capture_time_s=room.time_s)
        combined = Image.fromarray(np.concatenate((np.asarray(rgb),
            np.asarray(room.capture_external())), axis=1))
        draw = ImageDraw.Draw(combined)
        draw.rectangle((0, 0, combined.width, 34), fill=(0, 0, 0))
        draw.text((8, 8), f"{video_label} seed{seed} local repair | {action} | "
                  f"{room.time_s:.2f}s", fill=(255, 255, 255))
        video.append_data(np.asarray(combined))
        events.append({"action": action,
                       "time_s": round(room.time_s, 3),
                       "actor_input_sha256": sha(source),
                       "estimated_pose_xyyaw": room.estimated.tolist(),
                       "evaluator_actual_pose_xyyaw": room.actual.tolist(),
                       "evaluator_clearance_m": round(
                           room._current_obstacle_clearance()[0], 4),
                       "depth_rays_used": observed["depth_rays_used"],
                       "self_depth_mask": mask_info,
                       "detail": detail})

    initial_scan = sense(.11)
    localization = localizer.initialize(initial_scan, room.estimated)
    status = "moving"
    with imageio.get_writer(video_path, fps=4, codec="libx264", quality=7) as video:
        frame(video, "resume_observation", {"localizer": localization})
        for move_index in range(1, max_moves + 1):
            current_cell = (int(round(room.estimated[0] / GRID_M)),
                            int(round(room.estimated[1] / GRID_M)))
            if current_cell == goal or np.linalg.norm(
                    GRID_M * np.asarray(goal) - room.estimated[:2]) < .025:
                status = "local_goal_reached"
                break
            scan11 = sense(.11)
            scan8 = sense(.08)
            scan4 = sense(.04)
            if sum(value is not None for value in scan11) < 10:
                status = "range_coverage_stop"
                break
            room.camera_pose_estimate("lower_rgb")
            point = TwoHeightClearance(
                robot, room.pose_data.qpos.copy(), room.estimated,
                scan11, ([None] * len(scan4) if head_low_planning else scan4),
                voxel, room.time_s, range_8=scan8,
                head_low=head_low_planning)
            route, gaps, expanded = plan(
                room.estimated, anchor_xy, goal, blocked, visited,
                scan11, scan4, point,
                clearance_cost_weight=clearance_cost_weight,
                physical_travel_cap=physical_travel_cap,
                use_low_range_for_planning=not head_low_planning)
            committed_fallback = False
            if (len(route) < 2 and committed and
                    committed_fallback_moves < max_committed_fallback_moves):
                current = min(range(len(committed)), key=lambda index:
                    np.linalg.norm(GRID_M * np.asarray(committed[index]) -
                                   room.estimated[:2]))
                if (np.linalg.norm(GRID_M * np.asarray(committed[current]) -
                                   room.estimated[:2]) <= .06 and
                        current + 1 < len(committed)):
                    next_xy = GRID_M * np.asarray(committed[current + 1],
                                                  dtype=float)
                    fallback_gap = point.edge_gap(room.estimated[:2],
                                                  next_xy,
                                                  room.estimated[2])
                    planning_scans = ((scan11, scan8) if head_low_planning else
                                      (scan11, scan8, scan4))
                    if (all(observed_free(next_xy, room.estimated, scan)
                            for scan in planning_scans) and
                            fallback_gap >= MARGIN_M):
                        route = [current_cell, committed[current + 1]]
                        gaps = [fallback_gap]
                        committed_fallback = True
            if len(route) < 2:
                status = "no_actor_bypass_route"
                events.append({"action": status, "time_s": room.time_s,
                               "move_index": move_index,
                               "expanded_cells": expanded,
                               "range_11_m": scan11,
                               "range_8_m": scan8,
                               "range_4_m": scan4})
                break
            next_cell = route[1]
            next_xy = GRID_M * np.asarray(next_cell, dtype=float)
            delta = next_xy - room.estimated[:2]
            if np.linalg.norm(delta) > .081:
                status = "unbounded_command_stop"
                break
            room.move(float(delta[0]), float(delta[1]), slip=.88,
                      safety_probe=guard_probe())
            if committed_fallback:
                committed_fallback_moves += 1
            if room.safety_stop_observation is not None:
                status = "move_range_alarm"
            elif room.path_min_clearance[0] < 0:
                status = "evaluator_model_overlap"
            post_scan = sense(.11)
            if status == "moving":
                predicted = room.estimated.copy()
                corrected, localization = localizer.correct(
                    post_scan, predicted)
                if localization["status"] != "registered":
                    status = "localization_uncertain_stop"
                else:
                    correction = float(np.linalg.norm(
                        corrected[:2] - predicted[:2]))
                    suspicious = correction > max(.025, .6 * np.linalg.norm(delta))
                    inconsistent_motion = (inconsistent_motion + 1
                                           if suspicious else 0)
                    if motion_consistency_stop and inconsistent_motion >= 2:
                        status = "localization_motion_disagreement_stop"
                    else:
                        room.estimated[:] = corrected
            visited.add(next_cell)
            frame(video, "replanned_base_step", {
                "move_index": move_index,
                "plan_cells": route,
                "plan_edge_gaps_m": [round(gap, 4) for gap in gaps],
                "committed_fallback": committed_fallback,
                "expanded_cells": expanded,
                "command_delta_xy_m": delta.tolist(),
                "range_11_m": scan11,
                "range_8_m": scan8,
                "range_4_m": scan4,
                "localizer": localization,
                "consecutive_motion_disagreements": inconsistent_motion,
                "range_alarm": room.safety_stop_observation,
                "outcome": status})
            if status != "moving":
                break
            room.look(0., "lower_rgb", safety_probe=guard_probe())
            if room.safety_stop_observation is not None:
                status = "look_range_alarm"
            frame(video, "reobserve_after_replan", {
                "move_index": move_index,
                "range_alarm": room.safety_stop_observation,
                "outcome": status})
            if status != "moving":
                break
        if status == "moving":
            status = "local_goal_reached" if np.linalg.norm(
                GRID_M * np.asarray(goal) - room.estimated[:2]) < .045 else (
                    "move_budget_stop")

    min_clearance = min(room.path_min_clearance[0],
                        room._current_obstacle_clearance()[0])
    result = {"seed": seed,
              **provenance,
              "goal_audit_sha256": sha(goal_audit_path),
              "actor_local_goal_cell": goal,
              "actor_blocked_cell": blocked,
              "clearance_cost_weight": clearance_cost_weight,
              "physical_travel_cap": physical_travel_cap,
              "head_low_planning": head_low_planning,
              "committed_fallback_moves": committed_fallback_moves,
              "status": status,
              "executed_moves": sum(event.get("action") ==
                                    "replanned_base_step" for event in events),
              "final_estimated_pose_xyyaw": room.estimated.tolist(),
              "evaluator_final_actual_xyyaw": room.actual.tolist(),
              "evaluator_min_path_clearance_m": round(min_clearance, 4),
              "evaluator_model_overlap": min_clearance < 0.,
              "blocked_cell_traversed": (blocked is not None and any(
                  tuple(event["detail"]["plan_cells"][1]) == blocked
                  for event in events if event.get("action") ==
                  "replanned_base_step")),
              "events": events,
              "video_sha256": sha(video_path)}
    (branch / "trace.json").write_text(json.dumps(result, indent=2) + "\n")
    print(seed, status, "moves", result["executed_moves"],
          "min clearance", result["evaluator_min_path_clearance_m"],
          flush=True)
    return result


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    audit = json.loads(V85.read_text())
    goals = {int(row["seed"]): row for row in audit["rows"]}
    room = Room()
    try:
        robot = RobotGeometry(room.model)
        rows = [run(seed, goals[seed], room, robot) for seed in (85, 88)]
    finally:
        room.close()
    gate = all(row["status"] == "local_goal_reached" and
               row["evaluator_min_path_clearance_m"] >= .05 and
               not row["evaluator_model_overlap"] and
               not row["blocked_cell_traversed"] for row in rows)
    result = {"scope": "Live receding-horizon 4cm actor local bypass to V85 actor goals from exact V82 stops; fresh virtual 4/8/11cm prechecks and independent 10Hz 4/11cm guard, scan localization and calibrated head-depth at every step. V77 oracle direction remains a nondeployable upper-bound context; native gait/stop unqualified.",
              "source_sha256": sha(SOURCE),
              "v85_audit_sha256": sha(V85),
              "selected_design_sha256": sha(ROOT / "robot/ape-current-design.json"),
              "rows": [{key: value for key, value in row.items()
                        if key != "events"} for row in rows],
              "live_local_bypass_gate_pass": gate}
    (OUT / "live-local-bypass-audit.json").write_text(
        json.dumps(result, indent=2) + "\n")
    (OUT / "run_live_local_bypass_v86_source.py").write_bytes(
        SOURCE.read_bytes())
    print("live local bypass gate:", gate)


if __name__ == "__main__":
    main()
