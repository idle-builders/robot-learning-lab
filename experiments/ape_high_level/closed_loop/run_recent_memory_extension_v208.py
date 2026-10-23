"""V208: V111 controller with one bounded recent-memory information extension."""

from __future__ import annotations

import json
import math
from pathlib import Path

import imageio.v2 as imageio
import numpy as np
from PIL import Image, ImageDraw

from audit_fresh_recovery_approach_v110 import accepted_source
from audit_moved_goal_reselection_v135 import course_goal
from audit_side_look_target_v81 import infer
from depth_voxel import DepthVoxelMap
from fast_range_guard import FastRangeGuard
from goal_map_route import GoalMapRoute
from localization import ScanLocalizer
from robot_geometry import RobotGeometry
from run import H, ROOT, W, Room, Vision, iou, sensed_range, sha
from self_depth_mask import SelfDepthMask
from two_height_clearance import TwoHeightClearance
from upper_obstacle_side import choose_side


SOURCE = Path(__file__)
V81 = ROOT / "artifacts/ape-closed-loop-cove50-v81"
V110 = ROOT / "artifacts/ape-closed-loop-cove50-v110/fresh-recovery-approach-audit.json"
OUT = ROOT / "artifacts/ape-closed-loop-cove50-v111"
MAX_MOVES = 24
MIN_TARGET_GAP_M = .27
MAX_TARGET_GAP_M = .40


def run(seed: int, room: Room, robot: RobotGeometry,
        mask: SelfDepthMask, vision: Vision,
        exemplar: np.ndarray, *, point_all_graph_edges: bool = False,
        motion_supported_scan: bool = False,
        base_point_margin_m: float = .065,
        initial_information_path_cells: list[list[int]] | None = None,
        promote_information_on_annulus: bool = False,
        visual_anchor_scan_recovery: bool = False,
        lost_target_side_recovery: bool = False,
        information_memory_age_limit_s: float = 4.,
        skip_reacquire_during_info_recovery: bool = False,
        view_body_turn: bool = False,
        body_turn_bearing_rad: float = 1.6,
        require_turn_point_clearance: bool = False,
        prefer_wider_information_route: bool = False,
        extend_novel_information_view: bool = False,
        allow_recent_information_extension: bool = False,
        navigation_turn_near_information_goal: bool = False,
        body_aware_range_guard: bool = False,
        allow_information_visual_reanchor: bool = False,
        target_annulus_outer_m: float = .40,
        light_multiplier: float = 1.) -> dict:
    target, actual, paths, accepted, audit_path = accepted_source(seed)
    voxel = DepthVoxelMap()
    fovy = float(room.model.cam_fovy[room.model.camera("lower_rgb").id])
    qpos = None
    pose = None
    stamp = None
    input_hashes = []
    for path in paths:
        with np.load(path) as z:
            qpos = z["actor_qpos"].copy()
            pose = z["estimated_pose_xyyaw"].copy()
            depth = z["depth_m"].copy()
            camera = (z["camera_origin_m"].copy(),
                      z["camera_rotation"].copy())
            stamp = float(z["capture_time_s"])
        filtered, _ = mask.apply(depth, qpos)
        voxel.observe(filtered, camera, stamp, fovy_deg=fovy,
                      self_masked=True)
        input_hashes.append(sha(path))
    assert qpos is not None and pose is not None and stamp is not None
    last_confirmed_time_s = float(accepted.get("last_seen_time_s",stamp))
    room.reset()
    room.layout(seed, absent=False, furnished=True)
    room.model.body_pos[room.model.body("loop_table").id] = (
        .40, (1 if seed % 2 else -1) * .46, 0.)
    room.set_table_height(.37)
    room.model.light_diffuse[0] *= light_multiplier
    room.data.qpos[:] = qpos
    room.actual[:] = actual
    room.estimated[:] = pose
    room.head = float(qpos[room.head_addr])
    room.time_s = stamp
    room._forward()
    planner = GoalMapRoute()
    localizer = ScanLocalizer()
    monitor11 = FastRangeGuard(
        band_m=.075 if body_aware_range_guard else .05,
        near_cluster_min_points=2 if body_aware_range_guard else None,
        single_point_critical_band_m=.07 if body_aware_range_guard else None)
    monitor4 = FastRangeGuard(
        band_m=.075 if body_aware_range_guard else .05,
        near_cluster_min_points=2 if body_aware_range_guard else None,
        single_point_critical_band_m=.07 if body_aware_range_guard else None)
    scan_serial = 0
    frame_serial = 0
    inconsistent = 0
    scan_rejections = 0
    consecutive_scan_rejections = 0
    events = []
    pose_errors = []
    committed_route: list[tuple[int, int]] = (
        [tuple(cell) for cell in initial_information_path_cells[1:]]
        if initial_information_path_cells is not None else [])
    route_kind = "information" if initial_information_path_cells is not None else "annulus"
    route_index = 0
    route_target = target.copy()
    taught_target_anchor = target.copy()
    route_replacements = 0
    visited_information_cells = set(tuple(cell) for cell in
                                    (initial_information_path_cells or []))
    information_extensions = 0
    information_extension_active = False
    information_preference_latched = False
    visual_scan_reanchors = 0
    lost_target_side_recoveries = 0
    view_body_turns = 0
    navigation_body_turns = 0
    branch = OUT / f"seed{seed}"
    branch.mkdir(parents=True, exist_ok=True)
    video_path = branch / "episode.mp4"

    def sense(height: float) -> list[float | None]:
        nonlocal scan_serial
        scan_serial += 1
        return sensed_range(room, seed, 1110000 + scan_serial,
                            height_m=height)

    def target_head_yaw() -> float:
        bearing = math.atan2(target[1] - room.estimated[1],
                             target[0] - room.estimated[0]) - room.estimated[2]
        return float(np.clip(math.atan2(math.sin(bearing),
                                        math.cos(bearing)), -1.2, 1.2))

    def guard_probe():
        def point_gap(height: float):
            if not body_aware_range_guard:
                return None
            def gap(points_xy: np.ndarray) -> np.ndarray:
                robot.set_state(robot.pose_qpos(room.data.qpos, room.estimated))
                return robot.solid_distances(
                    np.c_[points_xy, np.full(len(points_xy), height)],
                    robot.primitives())
            return gap

        monitor11.update(sense(.11), room.estimated, room.time_s,
                         point_gap=point_gap(.11))
        monitor4.update(sense(.04), room.estimated, room.time_s,
                        point_gap=point_gap(.04))

        def probe(substep: int, fraction: float) -> dict | None:
            scan11 = sense(.11)
            scan4 = sense(.04)
            upper = monitor11.update(scan11, room.estimated, room.time_s,
                                     point_gap=point_gap(.11))
            lower = monitor4.update(scan4, room.estimated, room.time_s,
                                    point_gap=point_gap(.04))
            hazard = ("near_body_change", "near_body_cluster", "critical_body_point")
            if upper["status"] in hazard or lower["status"] in hazard:
                return {"source": ("low4" if lower["status"] in hazard
                                   else "torso11"),
                        "upper": upper, "lower": lower,
                        "capture_time_s": room.time_s,
                        "action_fraction": fraction}
            return None
        return probe

    def frame(video, action: str, detail: dict) -> dict:
        nonlocal frame_serial, target, last_confirmed_time_s
        rgb, depth = room.capture("lower_rgb", depth=True)
        camera = room.camera_pose_estimate("lower_rgb")
        actor_qpos = room.pose_data.qpos.copy()
        filtered, mask_info = mask.apply(depth, actor_qpos)
        observed = voxel.observe(filtered, camera, room.time_s,
                                 fovy_deg=fovy, self_masked=True)
        frame_serial += 1
        source = branch / f"frame-{frame_serial}.npz"
        np.savez_compressed(source, rgb=np.asarray(rgb), depth_m=depth,
                            camera_origin_m=camera[0],
                            camera_rotation=camera[1],
                            actor_qpos=actor_qpos,
                            estimated_pose_xyyaw=room.estimated.copy(),
                            capture_time_s=room.time_s)
        detection = infer(vision, exemplar, rgb, depth, camera, target,
                          None, branch / f"frame-{frame_serial}-annotated.png")
        selected = next((row for row in detection["proposals"]
                         if row["box"] == detection["accepted_box"]), None)
        truth = room.truth_box("TomatoSauce", "lower_rgb")
        correct = (selected is not None and truth is not None and
                   iou(selected["box"], truth) >= .25)
        if selected is not None and selected["depth_world_xyz_m"] is not None:
            target = np.asarray(selected["depth_world_xyz_m"][:2], dtype=float)
            last_confirmed_time_s = room.time_s
        combined = Image.fromarray(np.concatenate((np.asarray(rgb),
            np.asarray(room.capture_external())), axis=1))
        draw = ImageDraw.Draw(combined)
        draw.rectangle((0, 0, combined.width, 34), fill=(0, 0, 0))
        draw.text((8, 8), f"V111 seed{seed} | {action} | "
                  f"accepted {selected is not None} | "
                  f"{room.time_s:.2f}s", fill=(255, 255, 255))
        video.append_data(np.asarray(combined))
        pose_error = float(np.linalg.norm(room.estimated[:2] - room.actual[:2]))
        pose_errors.append(pose_error)
        event = {"action": action,
                 "time_s": round(room.time_s, 3),
                 "actor_input_sha256": sha(source),
                 "estimated_pose_xyyaw": room.estimated.tolist(),
                 "evaluator_actual_pose_xyyaw": room.actual.tolist(),
                 "evaluator_pose_error_m": round(pose_error, 4),
                 "evaluator_model_clearance_m": round(
                     room._current_obstacle_clearance()[0], 4),
                 "actor_target_xy_m": target.tolist(),
                 "actor_target_gap_m": round(float(np.linalg.norm(
                     room.estimated[:2] - target)), 4),
                 "accepted_box": detection["accepted_box"],
                 "accepted_similarity": detection["accepted_similarity"],
                 "accepted_world_xyz_m": (None if selected is None else
                                           selected["depth_world_xyz_m"]),
                 "accepted_correct_evaluator_only": correct,
                 "evaluator_truth_box": truth,
                 "proposals": detection["proposals"],
                 "self_mask": mask_info,
                 "depth_rays_used": observed["depth_rays_used"],
                 "detail": detail}
        events.append(event)
        return event

    def current_planner_path() -> list[tuple[int, int]]:
        origin = planner.cell(room.estimated[:2])
        node = planner.goal
        reverse_path = []
        seen = set()
        while node is not None and node != origin and node not in seen:
            seen.add(node)
            reverse_path.append(node)
            node = planner.last_parent.get(node)
        return list(reversed(reverse_path)) if node == origin else []

    def path_min_point_gap(point: TwoHeightClearance,
                           cells: list[tuple[int, int]]) -> float:
        start = room.estimated[:2].copy()
        gaps = []
        for cell in cells:
            end = planner.xy(cell)
            if np.linalg.norm(end - start) < .025:
                continue
            gaps.append(point.edge_gap(start, end, room.estimated[2]))
            start = end
        return min(gaps) if gaps else math.inf

    start_scan = sense(.11)
    localization = localizer.initialize(start_scan, room.estimated)
    status = "moving"
    moves = 0
    with imageio.get_writer(video_path, fps=4, codec="libx264", quality=7) as video:
        if abs(target_head_yaw() - room.head) > .01:
            room.look(target_head_yaw(), "lower_rgb",
                      safety_probe=guard_probe())
        if room.safety_stop_observation is not None:
            status = "initial_look_range_alarm"
        last = frame(video, "initial_target_view",
                     {"localizer": localization,
                      "range_alarm": room.safety_stop_observation})
        if (status == "moving" and last["accepted_box"] is None and
                (route_kind != "information" or
                 room.time_s-last_confirmed_time_s>
                 information_memory_age_limit_s)):
            status = "initial_target_not_reconfirmed"
        for move_index in range(1, MAX_MOVES + 1):
            if status != "moving":
                break
            bearing = math.atan2(target[1] - room.estimated[1],
                                 target[0] - room.estimated[0]) - room.estimated[2]
            bearing = math.atan2(math.sin(bearing), math.cos(bearing))
            navigation_turn = (navigation_turn_near_information_goal and
                               information_extension_active and
                               committed_route and
                               view_body_turns + navigation_body_turns == 0 and
                               last["accepted_box"] is not None and
                               abs(bearing) > .8 and
                               np.linalg.norm(
                                   planner.xy(committed_route[-1]) -
                                   room.estimated[:2]) <= .085)
            view_turn = (view_body_turn and route_kind == "annulus" and
                         view_body_turns + navigation_body_turns == 0 and
                         last["accepted_box"] is not None and
                         abs(bearing) > body_turn_bearing_rad)
            if navigation_turn or view_turn:
                turn_delta = math.copysign(
                    .6 if navigation_turn else .3, bearing)
                old_pose = room.estimated.copy()
                desired_yaw = float(old_pose[2] + turn_delta)
                actor_turn_gap = None
                if require_turn_point_clearance:
                    turn_scans = {height: sense(height)
                                  for height in (.04, .08, .11)}
                    turn_point = TwoHeightClearance(
                        robot, room.pose_data.qpos.copy(), old_pose,
                        turn_scans[.11], turn_scans[.04], voxel,
                        room.time_s, range_8=turn_scans[.08])
                    actor_turn_gap = min(turn_point.edge_gap(
                        old_pose[:2], old_pose[:2], yaw)
                        for yaw in np.linspace(old_pose[2], desired_yaw, 13))
                    if actor_turn_gap < base_point_margin_m:
                        status = "body_turn_3d_blocked_stop"
                        events.append({"action": status,
                                       "time_s": room.time_s,
                                       "move_index": move_index,
                                       "actor_swept_point_gap_m":
                                           round(actor_turn_gap, 4),
                                       "required_point_margin_m":
                                           base_point_margin_m})
                        break
                monitor11.update(sense(.11), room.estimated, room.time_s)
                monitor4.update(sense(.04), room.estimated, room.time_s)

                def turn_probe(substep: int, fraction: float) -> dict | None:
                    predicted_pose = old_pose.copy()
                    predicted_pose[2] += fraction * turn_delta
                    upper = monitor11.update(sense(.11), predicted_pose,
                                             room.time_s)
                    lower = monitor4.update(sense(.04), predicted_pose,
                                            room.time_s)
                    if (upper["status"] == "near_body_change" or
                            lower["status"] == "near_body_change"):
                        return {"source": ("low4" if lower["status"] ==
                                           "near_body_change" else "torso11"),
                                "upper": upper, "lower": lower,
                                "capture_time_s": room.time_s,
                                "action_fraction": fraction}
                    return None

                room.look(desired_yaw, "upper_rgb", safety_probe=turn_probe)
                turn_alarm = room.safety_stop_observation
                if navigation_turn:
                    navigation_body_turns += 1
                else:
                    view_body_turns += 1
                if turn_alarm is not None:
                    status = "body_turn_range_alarm"
                elif view_turn:
                    committed_route = []
                    route_index = 0
                last = frame(video, ("body_turn_for_route" if navigation_turn
                                     else "body_turn_for_view"), {
                    "move_index": move_index,
                    "requested_body_yaw_rad": desired_yaw,
                    "target_bearing_before_rad": round(bearing, 4),
                    "turn_purpose": ("information_route_orientation" if
                                     navigation_turn else "target_view"),
                    "actor_swept_point_gap_m": (None if actor_turn_gap is None
                                                else round(actor_turn_gap, 4)),
                    "range_alarm": turn_alarm,
                    "kinematic_turn_surrogate": True})
                if status != "moving":
                    break
                if last["accepted_box"] is None:
                    status = "body_turn_target_not_reconfirmed"
                    break
                continue
            gap = float(np.linalg.norm(room.estimated[:2] - target))
            if (MIN_TARGET_GAP_M <= gap <= MAX_TARGET_GAP_M and
                    last["accepted_box"] is not None):
                status = "target_area_reached"
                break
            scan11 = sense(.11)
            scan8 = sense(.08)
            scan4 = sense(.04)
            if sum(value is not None for value in scan11) < 10:
                status = "range_coverage_stop"
                break
            room.camera_pose_estimate("lower_rgb")
            point = TwoHeightClearance(robot, room.pose_data.qpos.copy(),
                                       room.estimated, scan11, scan4,
                                       voxel, room.time_s, range_8=scan8)
            point_margin = base_point_margin_m + (
                .005 if motion_supported_scan and consecutive_scan_rejections else 0.)
            if (route_kind != "information" and
                    float(np.linalg.norm(target - route_target)) > .10):
                committed_route = []
                route_index = 0
                route_target = target.copy()
            command = planner.plan(scan11, room.estimated, target, None,
                                   room.time_s, searching=False,
                                   edge_clearance=point.edge_gap,
                                   edge_clearance_margin_m=point_margin,
                                   local_all_point_edges=True,
                                   point_all_graph_edges=point_all_graph_edges,
                                   annulus_outer_m=target_annulus_outer_m)
            if (route_kind == "information" and
                    promote_information_on_annulus and
                    not information_extension_active and
                    not information_preference_latched and
                    last["accepted_box"] is not None and command is not None):
                fresh_path = current_planner_path()
                if fresh_path:
                    target_route_gap = path_min_point_gap(point, fresh_path)
                    info_route_gap = path_min_point_gap(
                        point, committed_route[route_index:])
                    defer = (prefer_wider_information_route and
                             info_route_gap >= target_route_gap + .01)
                    events.append({"action": ("information_route_preferred" if
                                              defer else
                                              "information_to_annulus_transition"),
                                   "time_s": room.time_s,
                                   "move_index": move_index,
                                   "actor_input_sha256": last["actor_input_sha256"],
                                   "fresh_path_cells": fresh_path,
                                   "fresh_target_route_point_gap_m":
                                       round(target_route_gap, 4),
                                   "remaining_information_point_gap_m":
                                       round(info_route_gap, 4),
                                   "planner_debug": planner.last_debug})
                    if not defer:
                        committed_route = fresh_path
                        route_kind = "annulus"
                        route_index = 0
                        route_target = target.copy()
                    else:
                        information_preference_latched = True
            if not committed_route and command is not None and route_kind != "information":
                committed_route = current_planner_path()
                route_index = 0
                route_target = target.copy()
            for repair_attempt in range(2):
                while (route_index < len(committed_route) and
                       np.linalg.norm(planner.xy(committed_route[route_index]) -
                                      room.estimated[:2]) < .025):
                    route_index += 1
                if route_index >= len(committed_route):
                    if (route_kind == "annulus" and
                            target_annulus_outer_m <= MAX_TARGET_GAP_M and
                            last["accepted_box"] is not None and
                            np.linalg.norm(room.estimated[:2] - target) >
                            MAX_TARGET_GAP_M):
                        stop_path = current_planner_path()
                        if (stop_path and stop_path != committed_route and
                                np.linalg.norm(planner.xy(stop_path[-1]) - target)
                                <= MAX_TARGET_GAP_M):
                            committed_route = stop_path
                            route_index = 0
                            events.append({
                                "action": "annulus_stop_band_replan",
                                "time_s": room.time_s,
                                "move_index": move_index,
                                "actor_input_sha256": last["actor_input_sha256"],
                                "fresh_path_cells": stop_path,
                                "target_gap_m": round(float(np.linalg.norm(
                                    room.estimated[:2] - target)), 4)})
                            continue
                    if (route_kind == "information" and
                            extend_novel_information_view and
                            information_extensions == 0 and
                            (last["accepted_box"] is not None or
                             (allow_recent_information_extension and
                              room.time_s - last_confirmed_time_s <=
                              information_memory_age_limit_s))):
                        cue = choose_side(voxel, room.estimated, target,
                                          room.time_s)
                        sides = ([cue["side"]] if cue["side"] is not None
                                 else ["left", "right"])
                        proposals = []
                        for side in sides:
                            route = course_goal(
                                room.estimated, target, side,
                                {.04: scan4, .08: scan8, .11: scan11},
                                point, visited_information_cells,
                                margin_m=base_point_margin_m + .005)
                            if route["goal_cell"] is None:
                                continue
                            endpoint = planner.xy(tuple(route["goal_cell"]))
                            novelty = min(float(np.linalg.norm(
                                endpoint - planner.xy(cell)))
                                for cell in visited_information_cells)
                            if (novelty >= .12 and
                                    1 < len(route["path_cells"]) <= 8):
                                proposals.append((novelty, side, route))
                        if proposals:
                            novelty, side, route = max(proposals,
                                                       key=lambda row: row[0])
                            committed_route = [tuple(cell) for cell in
                                               route["path_cells"][1:]]
                            route_index = 0
                            visited_information_cells.update(committed_route)
                            information_extensions += 1
                            information_extension_active = True
                            events.append({
                                "action": "novel_information_view_extension",
                                "time_s": room.time_s,
                                "move_index": move_index,
                                "actor_input_sha256": last["actor_input_sha256"],
                                "side_evidence": cue,
                                "chosen_side": side,
                                "recent_memory_extension":
                                    last["accepted_box"] is None,
                                "target_memory_age_s": round(
                                    room.time_s - last_confirmed_time_s, 3),
                                "endpoint_novelty_m": round(novelty, 4),
                                "actor_view_route": route})
                            continue
                    status = ("information_goal_reached" if route_kind=="information" else
                              "route_exhausted_without_target" if committed_route else
                              "no_actor_annulus_route")
                    events.append({"action": status, "time_s": room.time_s,
                                   "planner_debug": planner.last_debug,
                                   "move_index": move_index})
                    break
                next_cell = committed_route[route_index]
                waypoint = planner.xy(next_cell)
                displacement = waypoint - room.estimated[:2]
                distance = float(np.linalg.norm(displacement))
                delta = min(.08, distance) * displacement / distance
                first_gap = point.edge_gap(room.estimated[:2],
                                           room.estimated[:2] + delta,
                                           room.estimated[2])
                choice = {"goal_xy_m": planner.xy(committed_route[-1]).tolist(),
                          "goal_kind": ("committed_information" if
                                        route_kind=="information" else "committed_annulus"),
                          "next_waypoint_xy_m": waypoint.tolist(),
                          "route_index": route_index,
                          "route_length": len(committed_route),
                          "committed_path_cells": committed_route,
                          "route_replacements": route_replacements,
                          "step_distance_m": round(float(np.linalg.norm(delta)),4),
                          "known_free_next": planner.known_free(next_cell,
                                                                room.time_s),
                          "fresh_route_available": command is not None,
                          "planner_debug": planner.last_debug}
                if choice["known_free_next"] and first_gap >= point_margin:
                    break
                fresh_path = (current_planner_path() if command is not None and
                              route_kind!="information" else [])
                if (repair_attempt == 0 and fresh_path and
                        fresh_path != committed_route[route_index:]):
                    committed_route = fresh_path
                    route_index = 0
                    route_target = target.copy()
                    route_replacements += 1
                    continue
                status = "local_3d_route_blocked_stop"
                events.append({"action": status, "time_s": room.time_s,
                               "move_index": move_index,
                               "route_choice": choice,
                               "actor_point_gap_m": round(first_gap,4)})
                break
            if status != "moving":
                break
            if np.linalg.norm(delta) > .081 or first_gap < point_margin:
                status = "invalid_first_step_stop"
                break
            room.move(float(delta[0]), float(delta[1]), slip=.88,
                      safety_probe=guard_probe())
            moves += 1
            move_alarm = room.safety_stop_observation
            if move_alarm is not None:
                status = "move_range_alarm"
            elif room.path_min_clearance[0] < 0:
                status = "evaluator_model_overlap"
            post_scan = sense(.11)
            pending_visual_reanchor = False
            if status == "moving":
                predicted = room.estimated.copy()
                corrected, localization = localizer.correct(
                    post_scan, predicted)
                if localization["status"] != "registered":
                    status = "localization_uncertain_stop"
                else:
                    correction = float(np.linalg.norm(
                        corrected[:2] - predicted[:2]))
                    direction = delta / np.linalg.norm(delta)
                    offset = corrected[:2] - predicted[:2]
                    along = float(offset @ direction)
                    lateral = float(direction[0] * offset[1] -
                                    direction[1] * offset[0])
                    unsupported = (motion_supported_scan and
                                   (along < -.42 * np.linalg.norm(delta) - .005 or
                                    along > .005 or abs(lateral) > .01))
                    if unsupported:
                        scan_rejections += 1
                        consecutive_scan_rejections += 1
                        localization = {**localization,
                                        "status": "registered_motion_rejected",
                                        "correction_along_m": round(along, 4),
                                        "correction_lateral_m": round(lateral, 4),
                                        "rejection_count": consecutive_scan_rejections}
                        room.estimated[:] = predicted
                        if consecutive_scan_rejections >= 2:
                            if visual_anchor_scan_recovery and (
                                    route_kind == "annulus" or
                                    allow_information_visual_reanchor):
                                pending_visual_reanchor = True
                            else:
                                status = "localization_ambiguous_stop"
                    else:
                        consecutive_scan_rejections = 0
                        suspicious = correction > max(.025,
                                                      .6 * np.linalg.norm(delta))
                        inconsistent = inconsistent + 1 if suspicious else 0
                        if inconsistent >= 2:
                            status = "localization_motion_disagreement_stop"
                        else:
                            room.estimated[:] = corrected
            room.look(target_head_yaw(), "lower_rgb",
                      safety_probe=guard_probe())
            look_alarm = room.safety_stop_observation
            if look_alarm is not None and status == "moving":
                status = "look_range_alarm"
            last = frame(video, "guarded_approach_step", {
                "move_index": move_index,
                "command_delta_xy_m": delta.tolist(),
                "planner_goal": choice,
                "first_edge_point_gap_m": round(first_gap, 4),
                "range_11_m": scan11,
                "range_8_m": scan8,
                "range_4_m": scan4,
                "localizer": localization,
                "consecutive_motion_disagreements": inconsistent,
                "consecutive_scan_rejections": consecutive_scan_rejections,
                "actor_point_margin_m": point_margin,
                "move_range_alarm": move_alarm,
                "look_range_alarm": look_alarm,
                "range_alarm": move_alarm if move_alarm is not None else look_alarm,
                "outcome": status})
            if pending_visual_reanchor and status == "moving":
                selected = next((proposal for proposal in last["proposals"]
                                 if proposal["box"] == last["accepted_box"]), None)
                anchor_drift = float(np.linalg.norm(target - taught_target_anchor))
                consistent = (selected is not None and
                              selected["distance_to_memory_m"] is not None and
                              selected["distance_to_memory_m"] <= .02 and
                              anchor_drift <= .04)
                if consistent:
                    old_localization = localization
                    localization = localizer.initialize(post_scan, room.estimated)
                    consecutive_scan_rejections = 0
                    visual_scan_reanchors += 1
                    events.append({"action": "visual_anchor_scan_reanchor",
                                   "time_s": room.time_s,
                                   "move_index": move_index,
                                   "actor_input_sha256": last["actor_input_sha256"],
                                   "target_innovation_m": selected["distance_to_memory_m"],
                                   "target_anchor_drift_m": round(anchor_drift, 4),
                                   "rejected_scan": old_localization,
                                   "new_scan_map": localization})
                else:
                    status = "localization_ambiguous_stop"
                    events.append({"action": status,
                                   "time_s": room.time_s,
                                   "move_index": move_index,
                                   "actor_input_sha256": last["actor_input_sha256"],
                                   "target_anchor_drift_m": round(anchor_drift, 4),
                                   "visual_anchor_consistent": False})
            if status != "moving":
                break
            if (information_extension_active and
                    committed_route and
                    np.linalg.norm(planner.xy(committed_route[-1]) -
                                   room.estimated[:2]) < .04):
                information_extension_active = False
                route_kind = "annulus"
                committed_route = []
                route_index = 0
                events.append({"action": "information_extension_complete",
                               "time_s": room.time_s,
                               "move_index": move_index,
                               "actor_input_sha256": last["actor_input_sha256"]})
            if last["accepted_box"] is None:
                skip_sweep = (skip_reacquire_during_info_recovery and
                              route_kind == "information" and
                              lost_target_side_recoveries > 0)
                if not skip_sweep:
                    bearing = target_head_yaw()
                    for offset in (-.25, +.25, -.50, +.50):
                        yaw = float(np.clip(bearing + offset, -1.2, 1.2))
                        room.look(yaw, "lower_rgb", safety_probe=guard_probe())
                        if room.safety_stop_observation is not None:
                            status = "reacquire_look_range_alarm"
                        last = frame(video, "reacquire_head_view", {
                            "move_index": move_index,
                            "head_yaw_rad": yaw,
                            "range_alarm": room.safety_stop_observation})
                        if status != "moving" or last["accepted_box"] is not None:
                            break
                if status == "moving" and last["accepted_box"] is None:
                    if route_kind=="information":
                        age=room.time_s-last_confirmed_time_s
                        if age>information_memory_age_limit_s:
                            status="target_memory_expired_stop"
                        else:
                            events.append({"action":"target_memory_information_continue",
                                           "time_s":room.time_s,
                                           "move_index":move_index,
                                           "head_sweep_skipped":skip_sweep,
                                           "target_memory_age_s":round(age,3),
                                           "remembered_target_xy_m":target.tolist()})
                    else:
                        age = room.time_s-last_confirmed_time_s
                        cue = choose_side(voxel, room.estimated, target,
                                          room.time_s)
                        if (lost_target_side_recovery and
                                lost_target_side_recoveries == 0 and
                                age <= information_memory_age_limit_s and
                                cue["side"] is not None):
                            view_scans = {height: sense(height)
                                          for height in (.04, .08, .11)}
                            view_clearance = TwoHeightClearance(
                                robot, room.pose_data.qpos.copy(),
                                room.estimated, view_scans[.11],
                                view_scans[.04], voxel, room.time_s,
                                range_8=view_scans[.08])
                            view_route = course_goal(
                                room.estimated, target, cue["side"],
                                view_scans, view_clearance, set(),
                                margin_m=base_point_margin_m + .005)
                            if (view_route["goal_cell"] is not None and
                                    1 < len(view_route["path_cells"]) <= 8):
                                route_kind = "information"
                                committed_route = [tuple(cell) for cell in
                                                   view_route["path_cells"][1:]]
                                route_index = 0
                                lost_target_side_recoveries += 1
                                events.append({
                                    "action": "lost_target_side_view_route",
                                    "time_s": room.time_s,
                                    "move_index": move_index,
                                    "actor_input_sha256": last["actor_input_sha256"],
                                    "target_memory_age_s": round(age, 3),
                                    "side_evidence": cue,
                                    "actor_view_route": view_route})
                                continue
                        status = "target_lost_search_needed"
                        events.append({"action": status,
                                       "time_s": room.time_s,
                                       "move_index": move_index,
                                       "side_evidence": cue,
                                       "target_memory_age_s": round(age, 3)})
                        break
        if status == "moving":
            gap = float(np.linalg.norm(room.estimated[:2] - target))
            status = ("target_area_reached" if
                      MIN_TARGET_GAP_M <= gap <= MAX_TARGET_GAP_M and
                      last["accepted_box"] is not None else
                      "move_budget_stop")

    min_clearance = min(room.path_min_clearance[0],
                        room._current_obstacle_clearance()[0])
    final_gap = float(np.linalg.norm(room.estimated[:2] - target))
    result = {"seed": seed,
              "accepted_source_audit_sha256": sha(audit_path),
              "initial_head_actor_input_sha256": input_hashes,
              "status": status,
              "executed_moves": moves,
              "route_replacements": route_replacements,
              "scan_rejections": scan_rejections,
              "base_actor_point_margin_m": base_point_margin_m,
              "initial_information_path_cells":initial_information_path_cells,
              "promote_information_on_annulus":promote_information_on_annulus,
              "visual_anchor_scan_recovery":visual_anchor_scan_recovery,
              "visual_scan_reanchors": visual_scan_reanchors,
              "lost_target_side_recovery": lost_target_side_recovery,
              "lost_target_side_recoveries": lost_target_side_recoveries,
              "information_memory_age_limit_s":
                  information_memory_age_limit_s,
              "skip_reacquire_during_info_recovery":
                  skip_reacquire_during_info_recovery,
              "view_body_turn": view_body_turn,
              "body_turn_bearing_rad": body_turn_bearing_rad,
              "require_turn_point_clearance": require_turn_point_clearance,
              "prefer_wider_information_route":
                  prefer_wider_information_route,
              "extend_novel_information_view":
                  extend_novel_information_view,
              "allow_recent_information_extension":
                  allow_recent_information_extension,
              "information_extensions": information_extensions,
              "information_preference_latched":
                  information_preference_latched,
              "view_body_turns": view_body_turns,
              "navigation_turn_near_information_goal":
                  navigation_turn_near_information_goal,
              "body_aware_range_guard": body_aware_range_guard,
              "allow_information_visual_reanchor":
                  allow_information_visual_reanchor,
              "target_annulus_outer_m": target_annulus_outer_m,
              "navigation_body_turns": navigation_body_turns,
              "final_actor_target_xy_m": target.tolist(),
              "final_actor_target_gap_m": round(final_gap, 4),
              "final_estimated_pose_xyyaw": room.estimated.tolist(),
              "evaluator_final_actual_xyyaw": room.actual.tolist(),
              "evaluator_min_model_clearance_m": round(min_clearance, 4),
              "evaluator_model_overlap": min_clearance < 0.,
              "evaluator_max_pose_error_m": round(max(pose_errors), 4),
              "wrong_object_acceptances": sum(
                  event.get("accepted_box") is not None and
                  not event.get("accepted_correct_evaluator_only", False)
                  for event in events),
              "final_fresh_target_accepted": bool(
                  last.get("accepted_correct_evaluator_only", False)),
              "events": events,
              "video_sha256": sha(video_path)}
    (branch / "trace.json").write_text(json.dumps(result, indent=2) + "\n")
    print(seed, status, "moves", moves, "gap", result[
        "final_actor_target_gap_m"], "min clearance", result[
        "evaluator_min_model_clearance_m"], "wrong",
        result["wrong_object_acceptances"], flush=True)
    return result


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    prerequisite = json.loads(V110.read_text())
    if not prerequisite["fresh_recovery_approach_screen_pass"]:
        raise AssertionError("V110 first-edge screen failed")
    exemplar_path = V81 / "owner-taught-exemplar.npy"
    vision = Vision("generic")
    exemplar = np.load(exemplar_path)
    room = Room()
    mask = SelfDepthMask(room.model, camera="lower_rgb",
                         width=W, height=H)
    try:
        robot = RobotGeometry(room.model)
        rows = [run(seed, room, robot, mask, vision, exemplar)
                for seed in (85, 88)]
    finally:
        mask.close()
        room.close()
    gate = all(row["status"] == "target_area_reached" and
               row["evaluator_min_model_clearance_m"] >= .05 and
               not row["evaluator_model_overlap"] and
               row["wrong_object_acceptances"] == 0 and
               row["final_fresh_target_accepted"] and
               MIN_TARGET_GAP_M <= row["final_actor_target_gap_m"] <=
               MAX_TARGET_GAP_M for row in rows)
    result = {"scope": "V111 live target-area approach from V109/V105 freshly recognized taught-can images, nearby graph all-edge 3D selected-body6.5cm checks, current known-free route commitment and guarded replacement, 10Hz4/11cm monitor, scan localization/motion consistency, memory-bearing head gaze and frozen taught-object image gate after every move. Kinematic present-only development.",
              "source_sha256": sha(SOURCE),
              "v110_audit_sha256": sha(V110),
              "v81_exemplar_sha256": sha(exemplar_path),
              "selected_design_sha256": sha(ROOT / "robot/ape-current-design.json"),
              "rows": [{key: value for key, value in row.items()
                        if key != "events"} for row in rows],
              "two_scene_renewed_live_approach_gate_pass": gate}
    (OUT / "renewed-live-approach-audit.json").write_text(
        json.dumps(result, indent=2) + "\n")
    (OUT / "run_renewed_live_approach_v111_source.py").write_bytes(
        SOURCE.read_bytes())
    print("two-scene renewed live approach gate:", gate, flush=True)


if __name__ == "__main__":
    main()
