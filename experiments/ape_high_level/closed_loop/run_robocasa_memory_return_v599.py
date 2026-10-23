"""Revisit an owner-taught item in a furnished kitchen after walking away.

The full V497 route and owner turn execute first. This continuation keeps the
same scene alive, moves to a previously observed free viewpoint, returns to the
remembered show viewpoint, and requires a new image before any item claim.
Command, range-registration and RGB-D pose sources use the same range obstacle
stream and hidden alternating slip. No actor decision uses simulator truth.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
import time
from concurrent.futures import ThreadPoolExecutor

import cv2
import mujoco
import numpy as np
from PIL import Image

from audit_robocasa_layout2_routes_v409 import clearance
import actor_range_refuge_v744
import dynamic_trail_map
import head_depth_upper_route
import held_pose_screen
import nominal_emergency_rank
import predeparture_track_hold
from rgbd_odometry import RGBDOdometry
import robocasa_person_interaction as interaction
import run_robocasa_joined_owner_route_v410 as owner
import run_robocasa_layout2_route_v407 as route
import run_robocasa_person_joined_v497 as joined
import run_robocasa_moving_prop_v423 as moving


ROOT = Path(__file__).resolve().parents[3]
SOURCE = Path(__file__)
OUT = ROOT / "artifacts/ape-robocasa-memory-return-v599"
SOURCES = ("command", "range", "rgbd")
ITEM_STATES = ("retained", "removed_while_away")
MAX_STEPS_PER_LEG = 18
STEP_TIME_S = moving.ACTION_S
SUBSTEP_S = moving.SUBSTEP_S
POSITION_TOLERANCE_M = .12
MEMORY_ASSOCIATION_M = .25


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def head_depth_compact_surface(depth: np.ndarray, estimated_xy: np.ndarray,
                               camera_position_z: float, camera_rotation: np.ndarray,
                               fovy_deg: float, radius_m: float) -> dict | None:
    """Project a nearby lower-image depth blob with calibrated camera geometry."""
    height, width = depth.shape
    rows, columns = np.where(np.isfinite(depth) & (depth > .15) &
                             (depth < .75) &
                             (np.arange(height)[:, None] > height * .375))
    if len(rows) < 20000:
        return None
    values = depth[rows, columns]
    focal = height / (2. * np.tan(np.deg2rad(fovy_deg) / 2.))
    local = np.column_stack(((columns + .5 - width / 2.) * values / focal,
                             (height / 2. - rows - .5) * values / focal,
                             -values))
    known_camera_position = np.r_[estimated_xy, camera_position_z]
    world = local @ camera_rotation.T + known_camera_position
    surface = np.median(world[:, :2], axis=0)
    outward = surface - estimated_xy
    if np.linalg.norm(outward) < .05:
        return None
    center = surface + radius_m * outward / np.linalg.norm(outward)
    return {"blob_pixels": len(rows),
            "centroid_px": [round(float(np.mean(columns)), 3),
                            round(float(np.mean(rows)), 3)],
            "median_depth_m": round(float(np.median(values)), 5),
            "surface_xy_m": surface.round(5).tolist(),
            "center_xy_m": center.round(5).tolist()}


def callback(*, source: str, item_state: str, case: dict,
             model, data, renderer, actual, estimate, time_s,
             grid, localizer, route_trace, owner_trace, vision, exemplar,
             script, rng, fault_profile: str = "none",
             fault_after_move: int = 3, run_label: str | None = None,
             post_stop_until_s: float | None = None,
             predeparture_watch_s: float = 0.,
             predeparture_escape: bool = False,
             predeparture_future_hold_probe: bool = False,
             predeparture_future_hold_select: bool = False,
             predeparture_future_hold_revalidate: bool = False,
             predeparture_future_hold_replan_probe: bool = False,
             predeparture_confirmed_track_hold_select: bool = False,
             predeparture_track_continuity_gate: bool = False,
             escape_distances_m: tuple[float, ...] = (.35, .45, .25),
             escape_tolerance_m: float = POSITION_TOLERANCE_M,
             future_probe: bool = False,
             hold_monitor: bool = False,
             hold_scan_interval_s: float = .4,
             held_reactive_escape: bool = False,
             hold_scan_dropout_window_s: tuple[float, float] | None = None,
             head_depth_fallback: bool = False,
             paired_scan_seed: int | None = None,
             depth_candidate_probe: bool = False,
             head_depth_processing_latency_s: float = 0.,
             head_upper_processing_latency_s: float = 0.,
             future_hold_screen: bool = False,
             best_effort_if_unsupported: bool = False,
             map_snapshot_at_hazard: bool = False,
             dynamic_trail_reopen: bool = False,
             dynamic_trail_reopen_during_reactive: bool = False,
             held_warning_confirmations: int = 1,
             emergency_candidate_strategy: str = "perpendicular_first",
             evaluator_abort_on_margin_breach: bool = True,
             held_view_probe_at_s: float | None = None,
             held_view_min_raw_range_m: float = .55,
             held_view_recheck: bool = False,
             held_view_response_probe: bool = False,
             held_view_emergency: bool = False,
             held_view_directional_emergency: bool = False,
             held_view_worker_range_dropout_s: tuple[float, float] | None =
                 None) -> dict:
    if held_warning_confirmations < 1:
        raise ValueError("held_warning_confirmations must be positive")
    if held_view_recheck and held_view_probe_at_s is None:
        raise ValueError("held-view recheck needs a scheduled view")
    if held_view_directional_emergency and not held_view_emergency:
        raise ValueError("directional filter needs the emergency response")
    if (held_view_worker_range_dropout_s is not None and
            not (0. <= held_view_worker_range_dropout_s[0] <
                 held_view_worker_range_dropout_s[1])):
        raise ValueError("worker range dropout must be an elapsed-time interval")
    if predeparture_future_hold_probe and post_stop_until_s is None:
        raise ValueError("predeparture future hold probe needs a hold horizon")
    if predeparture_future_hold_select and not (predeparture_future_hold_probe and
                                                future_probe):
        raise ValueError("predeparture hold selection needs the actor probe")
    if predeparture_future_hold_revalidate and not predeparture_future_hold_select:
        raise ValueError("early hold revalidation needs early hold selection")
    if predeparture_future_hold_replan_probe and not predeparture_future_hold_revalidate:
        raise ValueError("early hold replan probe needs revalidation")
    if predeparture_confirmed_track_hold_select and not predeparture_future_hold_select:
        raise ValueError("confirmed-track early hold selection needs early selection")
    if predeparture_track_continuity_gate and predeparture_watch_s <= 0.:
        raise ValueError("track-continuity gate needs a predeparture watch")
    if (head_upper_processing_latency_s < 0. or
            abs(head_upper_processing_latency_s / SUBSTEP_S -
                round(head_upper_processing_latency_s / SUBSTEP_S)) > 1.e-8):
        raise ValueError("head upper processing latency must be a nonnegative substep multiple")
    if emergency_candidate_strategy not in ("perpendicular_first",
                                            "nominal_ranked",
                                            "nominal_depth_veto"):
        raise ValueError("unsupported emergency candidate strategy")
    if paired_scan_seed is not None:
        rng = np.random.default_rng(paired_scan_seed)
    response = f"memory_{source}_{item_state}"
    if fault_profile != "none":
        response += f"_{fault_profile}"
    if run_label is not None:
        response += f"_{run_label}"
    branch = OUT / case["id"] / f"{response}-rgbd"
    branch.mkdir(parents=True, exist_ok=True)
    remembered = next((row for row in reversed(owner_trace)
                       if row.get("stage") == "fresh_owner_view" and
                       row.get("depth_point_xyz_m") is not None), None)
    if remembered is None:
        raise ValueError("Owner turn produced no actor-visible metric item memory")
    item_memory = np.asarray(remembered["depth_point_xyz_m"], dtype=float)
    memory_time_s = float(remembered["time_s"])
    return_view = estimate[:2].copy()
    away_view = np.asarray(route_trace[3]["estimated_xy_m"], dtype=float)
    trace, frames = [], []
    ape_body = model.body("audit_ape").id
    ape_geom = model.geom("audit_ape_geom").id
    prop_body = model.body("route_obstacle").id
    camera_id = model.camera(interaction.CAMERA).id
    head_id = model.camera("head").id
    head_base_quat = model.cam_quat[head_id].copy()
    head_aim_start_s = None
    head_aim_delta_rad = 0.
    head_aim_track = None
    previous_head_depth = None
    head_depth_capture_count = 0
    dynamic_trail_reclaimed_cells = 0
    vision_odom = RGBDOdometry() if source == "rgbd" else None
    motion_count = 0
    min_proxy_gap = .1
    route_state = "running"
    last_pose_health = "command_only" if source == "command" else "registered"
    range_recovery = False
    fault_applied = False
    # The owner-show stop ends the original route. A new task request may
    # navigate to already observed viewpoints while retaining the grid's
    # moving-track monitor and known-free requirement.
    grid.mode = "navigate_after_fresh_remap"
    if predeparture_track_continuity_gate:
        # A shape last seen many seconds ago cannot lend its ID and velocity
        # to a newly visible low object without fresh temporal association.
        grid.multiple.max_association_gap_s = 1.0

    def place_cameras() -> None:
        model.cam_pos[camera_id, :2] = actual
        model.cam_pos[head_id, :2] = actual
        if head_aim_start_s is not None:
            progress = np.clip((time_s - head_aim_start_s) / .4, 0., 1.)
            angle = progress * head_aim_delta_rad
            yaw_quat = np.array([np.cos(angle / 2.), 0., 0., np.sin(angle / 2.)])
            aimed = np.empty(4)
            mujoco.mju_mulQuat(aimed, yaw_quat, head_base_quat)
            model.cam_quat[head_id] = aimed
        model.body_pos[prop_body, :2] = script(time_s)
        mujoco.mj_forward(model, data)

    place_cameras()
    if vision_odom is not None:
        rgb, depth, _ = owner.capture(renderer, data, interaction.CAMERA, True)
        if depth is None:
            raise ValueError("Initial RGB-D localization frame missing")
        rotation = data.cam_xmat[camera_id].reshape(3, 3).copy()
        vision_odom.initialize(Image.fromarray(rgb), depth, estimate.copy(), rotation)

    watch_tracks = []
    watch_trigger = None
    previous_watch_scan = None
    beam_changes = []
    escape_target = None
    early_hold_track_ids = []
    if predeparture_watch_s > 0.:
        watch_steps = int(round(predeparture_watch_s / SUBSTEP_S))
        for watch_step in range(1, watch_steps + 1):
            time_s = round(time_s + SUBSTEP_S, 5)
            place_cameras()
            guard_range = moving.raw_nearest_range(model, data, actual)
            gap, nearest = clearance(model, data, ape_body, ape_geom, actual)
            min_proxy_gap = min(min_proxy_gap, gap)
            trace.append({"stage": "predeparture_guard", "time_s": round(time_s, 4),
                          "actor_guard_range_m": guard_range,
                          "evaluator_proxy_gap_m": round(gap, 5),
                          "evaluator_nearest_geom": nearest,
                          "evaluator_actual_xy_m": actual.round(5).tolist()})
            if guard_range < moving.GUARD_CENTER_RANGE_M:
                route_state = "predeparture_near_body_alarm_ask_owner"
                watch_trigger = "near_body_alarm"
                trace.append({"stage": "predeparture_owner_help",
                              "time_s": round(time_s, 4),
                              "speech_intent": "Could you clear that moving thing before I go?",
                              "actor_reason": watch_trigger})
                break
            if watch_step % int(round(STEP_TIME_S / SUBSTEP_S)):
                continue
            scan = joined.SCAN(model, data, actual, 5., rng)
            grid.next_capture_time_s = time_s
            grid.update(scan, estimate[:2], 5.)
            track = grid.track_history[-1]
            center = track.get("center_xy_m")
            center = None if center is None else np.asarray(center, dtype=float)
            watch_row = {"stage": "predeparture_range_scan",
                         "time_s": round(time_s, 4),
                         "range_72_beams_m": scan,
                         "actor_estimated_xy_m": estimate[:2].round(5).tolist(),
                         "track_status": track.get("status"),
                         "track_center_xy_m": (None if center is None else
                                               center.round(5).tolist()),
                         "track_radius_m": track.get("radius_m")}
            if future_probe:
                watch_row["actor_multi_tracks"] = grid.multi_history[-1]["tracks"]
            if predeparture_escape and previous_watch_scan is not None:
                old = np.array([np.nan if value is None else value
                                for value in previous_watch_scan], dtype=float)
                new = np.array([np.nan if value is None else value
                                for value in scan], dtype=float)
                shortening = old - new
                changed = np.flatnonzero(np.isfinite(shortening) &
                                         (shortening >= .07) & (new < 1.0))
                if len(changed) >= 3 and changed[-1] - changed[0] <= 8:
                    center_beam = float(np.average(
                        changed, weights=shortening[changed]))
                    event = {"time_s": round(time_s, 4),
                             "beam_indices": changed.tolist(),
                             "weighted_center_beam": round(center_beam, 4),
                             "largest_shortening_m": round(
                                 float(np.max(shortening[changed])), 4)}
                    beam_changes.append(event)
                    watch_row["actor_beam_change"] = event
                    if (len(beam_changes) >= 2 and
                            beam_changes[-2]["time_s"] >= time_s - .4001):
                        change = center_beam - beam_changes[-2]["weighted_center_beam"]
                        if abs(change) >= .5:
                            bearing = estimate[2] - np.pi + 2 * np.pi * center_beam / 72
                            tangent = (np.array([-np.sin(bearing), np.cos(bearing)])
                                       if change < 0 else
                                       np.array([np.sin(bearing), -np.cos(bearing)]))
                            candidates = []
                            for distance in escape_distances_m:
                                candidate = estimate[:2] + distance * tangent
                                requested, status, path = grid.plan(
                                    estimate[:2], candidate)
                                candidates.append({"goal_xy_m": candidate.round(5).tolist(),
                                                   "plan_status": status,
                                                   "has_request": requested is not None,
                                                   "path_cells": len(path)})
                                if status == "known_goal" and requested is not None:
                                    escape_target = candidate
                                    break
                            watch_row["actor_escape_candidates"] = candidates
                            watch_row["actor_bearing_change_beams"] = round(change, 4)
                            if future_probe:
                                alternatives = []
                                early_tracks = ([
                                    track for track in grid.multi_history[-1]["tracks"]
                                    if track["status"] == "confirmed_now" and
                                    track["hits"] >= 3 and
                                    track["radius_m"] >= .18 and
                                    track["velocity_xy_m_s"] is not None and
                                    np.linalg.norm(track["velocity_xy_m_s"]) >= .1]
                                    if predeparture_future_hold_probe else [])
                                for radius in (.30, .45, .60):
                                    for direction in range(16):
                                        angle = 2 * np.pi * direction / 16
                                        option = estimate[:2] + radius * np.array(
                                            [np.cos(angle), np.sin(angle)])
                                        request, state, option_path = grid.plan(
                                            estimate[:2], option)
                                        if state == "known_goal" and request is not None:
                                            candidate = {
                                                "goal_xy_m": option.round(5).tolist(),
                                                "distance_m": radius,
                                                "direction_index": direction,
                                                "path_cells": len(option_path)}
                                            if predeparture_future_hold_probe:
                                                path_xy = [estimate[:2].tolist()] + [
                                                    grid.xy(cell).tolist()
                                                    for cell in option_path[1:]]
                                                candidate["early_future_hold_by_track"] = [
                                                    {"track_id": track["id"],
                                                     "screen": held_pose_screen.screen(
                                                         path_xy_m=path_xy,
                                                         track_center_xy_m=track[
                                                             "center_xy_m"],
                                                         track_velocity_xy_m_s=track[
                                                             "velocity_xy_m_s"],
                                                         track_radius_m=track["radius_m"],
                                                         track_capture_time_s=track[
                                                             "last_seen_s"],
                                                         decision_time_s=time_s,
                                                         hold_until_s=post_stop_until_s,
                                                         pose_error_m=(.05 if source ==
                                                                       "range" else .12))}
                                                    for track in early_tracks]
                                                candidate["early_supported_under_models"] = (
                                                    bool(early_tracks) and all(
                                                        row["screen"][
                                                            "supported_under_models"]
                                                        for row in candidate[
                                                            "early_future_hold_by_track"]))
                                            alternatives.append(candidate)
                                watch_row["actor_known_free_alternatives"] = alternatives
                                if predeparture_future_hold_probe:
                                    watch_row["actor_early_future_hold_probe"] = {
                                        "actor_track_ids": [track["id"]
                                                            for track in early_tracks],
                                        "known_free_candidate_count": len(alternatives),
                                        "supported_candidate_count": sum(
                                            candidate["early_supported_under_models"]
                                            for candidate in alternatives),
                                        "probe_only":
                                            not predeparture_future_hold_select}
                                if predeparture_future_hold_select:
                                    supported = [candidate for candidate in alternatives
                                                 if candidate[
                                                     "early_supported_under_models"]]
                                    if supported:
                                        selected = max(supported, key=lambda candidate: (
                                            min(row["screen"]["worst_modeled_gap_m"]
                                                for row in candidate[
                                                    "early_future_hold_by_track"]),
                                            -candidate[
                                                "early_future_hold_by_track"][0][
                                                    "screen"]["path_length_m"]))
                                        escape_target = np.asarray(
                                            selected["goal_xy_m"], dtype=float)
                                        early_hold_track_ids = [track["id"]
                                                                for track in early_tracks]
                                        watch_trigger = "modeled_supported_early_hold"
                                        watch_row["actor_early_selected_goal_xy_m"] = (
                                            selected["goal_xy_m"])
                                        watch_row["actor_early_selection_status"] = (
                                            "modeled_supported")
                                    else:
                                        escape_target = None
                                        route_state = "no_actor_supported_predeparture_wait"
                                        watch_trigger = "no_supported_early_hold"
                                        watch_row["actor_early_selected_goal_xy_m"] = None
                                        watch_row["actor_early_selection_status"] = (
                                            "no_supported_pose")
                            if not predeparture_future_hold_select:
                                watch_trigger = "beam_change_escape"
                                if escape_target is None:
                                    route_state = "no_actor_certified_escape"
                previous_watch_scan = scan
            elif predeparture_escape:
                previous_watch_scan = scan
            if center is not None:
                watch_tracks.append((time_s, center))
                if len(watch_tracks) >= 2:
                    previous_time, previous_center = watch_tracks[-2]
                    dt = time_s - previous_time
                    speed = float(np.linalg.norm(center - previous_center) / dt)
                    closing = float((np.linalg.norm(previous_center - estimate[:2]) -
                                     np.linalg.norm(center - estimate[:2])) / dt)
                    distance = float(np.linalg.norm(center - estimate[:2]))
                    watch_row.update({"actor_track_speed_mps": round(speed, 4),
                                      "actor_closing_speed_mps": round(closing, 4),
                                      "actor_track_distance_m": round(distance, 4)})
                    if (.15 <= speed <= 1.0 and closing >= .10 and
                            distance < 1.2):
                        route_state = "moving_hazard_hold_ask_owner"
                        watch_trigger = "range_track_closing"
            if (predeparture_confirmed_track_hold_select and
                    route_state == "running" and escape_target is None and
                    "actor_early_future_hold_probe" not in watch_row and
                    any(track["status"] == "confirmed_now" and
                        track["hits"] >= 3 and track["radius_m"] >= .18 and
                        track["velocity_xy_m_s"] is not None and
                        np.linalg.norm(track["velocity_xy_m_s"]) >= .1
                        for track in grid.multi_history[-1]["tracks"])):
                scored = predeparture_track_hold.screen(
                    grid, estimate[:2], time_s, source, post_stop_until_s)
                watch_row["actor_known_free_alternatives"] = scored[
                    "alternatives"]
                watch_row["actor_early_future_hold_probe"] = {
                    "actor_track_ids": scored["actor_track_ids"],
                    "known_free_candidate_count": scored[
                        "known_free_candidate_count"],
                    "supported_candidate_count": scored[
                        "supported_candidate_count"],
                    "probe_only": False,
                    "trigger": "confirmed_motion_track"}
                watch_row["actor_confirmed_track_trigger"] = {
                    "track_ids": scored["actor_track_ids"],
                    "capture_time_s": round(time_s, 4)}
                if scored["selected"] is not None:
                    selected = scored["selected"]
                    escape_target = np.asarray(selected["goal_xy_m"], dtype=float)
                    early_hold_track_ids = scored["actor_track_ids"]
                    watch_trigger = "modeled_supported_confirmed_track_hold"
                    watch_row["actor_early_selected_goal_xy_m"] = selected[
                        "goal_xy_m"]
                    watch_row["actor_early_selection_status"] = "modeled_supported"
                else:
                    route_state = "no_actor_supported_predeparture_wait"
                    watch_trigger = "no_supported_confirmed_track_hold"
                    watch_row["actor_early_selected_goal_xy_m"] = None
                    watch_row["actor_early_selection_status"] = "no_supported_pose"
            if (predeparture_track_continuity_gate and
                    watch_step == watch_steps and route_state == "running" and
                    escape_target is None):
                unresolved = [track for track in grid.multi_history[-1]["tracks"]
                              if track["status"] == "candidate_now" and
                              track["radius_m"] >= .18]
                if unresolved:
                    route_state = "unresolved_new_object_wait"
                    watch_trigger = "new_range_object_needs_temporal_confirmation"
                    watch_row["actor_unresolved_new_track_ids"] = [
                        track["id"] for track in unresolved]
            trace.append(watch_row)
            renderer.update_scene(data, camera="head")
            frames.append(joined.navigation_frame(
                renderer.render().copy(), grid, estimate[:2], [],
                f"watch before memory departure: {track.get('status')}",
                watch_step // 20))
            if route_state != "running":
                trace.append({"stage": "predeparture_owner_help",
                              "time_s": round(time_s, 4),
                              "speech_intent": "Could you clear that moving thing before I go?",
                              "actor_reason": watch_trigger})
                break
            if escape_target is not None:
                break

    phases = (() if route_state != "running" else
              (("evade", escape_target),) if escape_target is not None else
              (("away", away_view), ("return", return_view)))
    def execute_phase(phase, goal):
        nonlocal time_s, route_state, last_pose_health, range_recovery
        nonlocal fault_applied, motion_count, min_proxy_gap
        nonlocal dynamic_trail_reclaimed_cells
        reached = False
        for phase_step in range(MAX_STEPS_PER_LEG):
            place_cameras()
            tolerance = escape_tolerance_m if phase.startswith("evade") else POSITION_TOLERANCE_M
            if float(np.linalg.norm(estimate[:2] - goal)) <= tolerance:
                reached = True
                break
            scan = joined.SCAN(model, data, actual, 5., rng)
            scan_time = float(time_s)
            if source == "range" or range_recovery:
                estimate[:], localization = localizer.correct(scan, estimate)
                last_pose_health = localization["status"]
                if last_pose_health not in ("mapped", "registered"):
                    route_state = "pose_health_stop"
                    break
            else:
                localization = {"status": last_pose_health}
            grid.next_capture_time_s = scan_time
            grid.update(scan, estimate[:2], 5.)
            if dynamic_trail_reopen_during_reactive and phase == "evade_reactive":
                trail_update = dynamic_trail_map.reopen_observed_trail(
                    grid, scan, estimate[:2], scan_time,
                    grid.multi_history, 5.,
                    route.RADIUS_M + route.MARGIN_M)
                dynamic_trail_reclaimed_cells += (
                    trail_update["reclaimed_cell_count"])
                trace.append({"stage": "memory_dynamic_trail_reopen",
                              "phase": phase,
                              "actor_estimated_xy_m": estimate[:2].round(5).tolist(),
                              "range_72_beams_m": scan,
                              "actor_multi_tracks": grid.multi_history[-1]["tracks"],
                              **trail_update})
            requested, plan_status, path = grid.plan(estimate[:2], goal)
            row = {"stage": "memory_route_decision", "phase": phase,
                   "step": phase_step, "time_s": round(scan_time, 4),
                   "source": source, "actor_estimated_xy_m": estimate[:2].round(5).tolist(),
                   "pose_source_used": "range_recovery" if range_recovery else source,
                   "evaluator_actual_xy_m": actual.round(5).tolist(),
                   "range_72_beams_m": scan,
                   "range_capture_time_s": round(scan_time, 4),
                   "localization": localization,
                   "plan_status": plan_status,
                   "requested_xy_m": None if requested is None else requested.tolist(),
                   "memory_item_xyz_m": item_memory.round(5).tolist(),
                   "memory_capture_time_s": memory_time_s,
                   "motion_substeps": []}
            trace.append(row)
            renderer.update_scene(data, camera="head")
            rgb_navigation = renderer.render().copy()
            frames.append(joined.navigation_frame(
                rgb_navigation, grid, estimate[:2], path,
                f"memory {phase}: {row['pose_source_used']} / {plan_status}",
                motion_count))
            if requested is None or plan_status != "known_goal":
                route_state = "no_observed_free_return_route"
                break
            if (phase == "evade" and predeparture_future_hold_revalidate and
                    early_hold_track_ids):
                path_xy = [estimate[:2].tolist()] + [
                    grid.xy(cell).tolist() for cell in path[1:]]
                tracked = [track for track in grid.multi_history[-1]["tracks"]
                           if track["id"] in early_hold_track_ids and
                           track["status"] == "confirmed_now" and
                           track["velocity_xy_m_s"] is not None and
                           abs(track["last_seen_s"] - scan_time) < .001]
                screens = [{"track_id": track["id"],
                            "screen": held_pose_screen.screen(
                                path_xy_m=path_xy,
                                track_center_xy_m=track["center_xy_m"],
                                track_velocity_xy_m_s=track["velocity_xy_m_s"],
                                track_radius_m=track["radius_m"],
                                track_capture_time_s=track["last_seen_s"],
                                decision_time_s=scan_time,
                                hold_until_s=post_stop_until_s,
                                pose_error_m=(.05 if source == "range" else .12))}
                           for track in tracked]
                supported_now = (len(tracked) == len(early_hold_track_ids) and
                                 all(item["screen"]["supported_under_models"]
                                     for item in screens))
                trace.append({"stage": "memory_early_hold_revalidation",
                              "time_s": round(scan_time, 4),
                              "actor_estimated_xy_m": estimate[:2].round(5).tolist(),
                              "actor_goal_xy_m": goal.round(5).tolist(),
                              "actor_original_track_ids": early_hold_track_ids,
                              "actor_fresh_track_count": len(tracked),
                              "actor_screens": screens,
                              "actor_supported_now": supported_now})
                if not supported_now:
                    if predeparture_future_hold_replan_probe:
                        alternatives = []
                        for radius in (.30, .45, .60):
                            for direction in range(16):
                                angle = 2. * np.pi * direction / 16
                                option = estimate[:2] + radius * np.array(
                                    [np.cos(angle), np.sin(angle)])
                                option_request, option_status, option_path = grid.plan(
                                    estimate[:2], option)
                                if option_status != "known_goal" or option_request is None:
                                    continue
                                option_path_xy = [estimate[:2].tolist()] + [
                                    grid.xy(cell).tolist() for cell in option_path[1:]]
                                option_screens = [{"track_id": track["id"],
                                                   "screen": held_pose_screen.screen(
                                                       path_xy_m=option_path_xy,
                                                       track_center_xy_m=track[
                                                           "center_xy_m"],
                                                       track_velocity_xy_m_s=track[
                                                           "velocity_xy_m_s"],
                                                       track_radius_m=track["radius_m"],
                                                       track_capture_time_s=track[
                                                           "last_seen_s"],
                                                       decision_time_s=scan_time,
                                                       hold_until_s=post_stop_until_s,
                                                       pose_error_m=(.05 if source ==
                                                                     "range" else .12))}
                                                  for track in tracked]
                                alternatives.append({
                                    "goal_xy_m": option.round(5).tolist(),
                                    "distance_m": radius,
                                    "direction_index": direction,
                                    "path_cells": len(option_path),
                                    "screens": option_screens,
                                    "supported_under_models": bool(tracked) and
                                        len(tracked) == len(early_hold_track_ids) and
                                        all(item["screen"]["supported_under_models"]
                                            for item in option_screens)})
                        trace.append({
                            "stage": "memory_early_hold_replan_probe",
                            "time_s": round(scan_time, 4),
                            "actor_estimated_xy_m": estimate[:2].round(5).tolist(),
                            "actor_original_track_ids": early_hold_track_ids,
                            "actor_fresh_tracks": tracked,
                            "known_free_candidate_count": len(alternatives),
                            "supported_candidate_count": sum(
                                item["supported_under_models"] for item in alternatives),
                            "actor_known_free_alternatives": alternatives,
                            "probe_only": True})
                    route_state = "early_hold_model_invalidated"
                    break
            command = np.asarray(requested) - estimate[:2]
            command_length = float(np.linalg.norm(command))
            if command_length > .25:
                route_state = "pose_jump_route_veto"
                row["actor_route_command_length_m"] = round(command_length, 5)
                break
            if command_length > .10:
                command *= .10 / command_length
            row["actor_bounded_command_xy_m"] = command.round(5).tolist()
            slip = .58 if motion_count % 2 == 0 else .94
            true_delta = command * slip  # simulator actuation; actor gets only command
            start_actual = actual.copy()
            interrupted = False
            for substep in range(1, int(round(STEP_TIME_S / SUBSTEP_S)) + 1):
                next_time = time_s + SUBSTEP_S
                fraction = substep * SUBSTEP_S / STEP_TIME_S
                next_actual = start_actual + fraction * true_delta
                model.body_pos[prop_body, :2] = script(next_time)
                model.cam_pos[head_id, :2] = actual
                mujoco.mj_forward(model, data)
                guard_range = moving.raw_nearest_range(model, data, actual)
                if guard_range < moving.GUARD_CENTER_RANGE_M:
                    time_s = next_time
                    row["motion_substeps"].append({
                        "time_s": round(next_time, 4),
                        "actor_guard_range_m": guard_range,
                        "stopped_before_move": True})
                    route_state = "near_body_stop"
                    interrupted = True
                    break
                actual[:] = next_actual
                time_s = next_time
                model.cam_pos[head_id, :2] = actual
                model.cam_pos[camera_id, :2] = actual
                gap, nearest = clearance(model, data, ape_body, ape_geom, actual)
                min_proxy_gap = min(min_proxy_gap, gap)
                row["motion_substeps"].append({
                    "time_s": round(time_s, 4),
                    "actor_guard_range_m": guard_range,
                    "evaluator_proxy_gap_m": round(gap, 5),
                    "evaluator_nearest_geom": nearest,
                    "evaluator_actual_xy_m": actual.round(5).tolist(),
                    "stopped_before_move": False})
                if gap < .05:
                    row["evaluator_margin_breach_seen"] = True
                    if evaluator_abort_on_margin_breach:
                        route_state = "evaluator_margin_breach"
                        interrupted = True
                        break
            if interrupted:
                break
            estimate[:2] += command
            motion_count += 1
            place_cameras()
            if vision_odom is not None and not range_recovery:
                if (not fault_applied and motion_count == fault_after_move and
                        fault_profile in ("stop_on_rgbd_loss", "range_fallback")):
                    fault_applied = True
                    last_pose_health = "depth_packet_missing"
                    row["post_move_localization"] = {
                        "status": "depth_packet_missing",
                        "capture_time_s": round(time_s, 4),
                        "policy": fault_profile}
                    if fault_profile == "range_fallback":
                        range_recovery = True
                        last_pose_health = "awaiting_fresh_range_registration"
                        row["post_move_localization"]["next_source"] = "range_recovery"
                    else:
                        route_state = "pose_health_stop"
                    if route_state != "running":
                        break
                    continue
                rgb, depth, _ = owner.capture(renderer, data, interaction.CAMERA, True)
                if depth is None:
                    route_state = "pose_health_stop"
                    last_pose_health = "depth_missing"
                    break
                corrected, localization = vision_odom.correct(
                    Image.fromarray(rgb), depth, estimate.copy(),
                    data.cam_xmat[camera_id].reshape(3, 3).copy())
                last_pose_health = localization["status"]
                row["post_move_localization"] = localization
                if last_pose_health != "registered":
                    route_state = "pose_health_stop"
                    break
                estimate[:] = corrected
            row["evaluator_post_move_pose_error_m"] = round(
                float(np.linalg.norm(estimate[:2] - actual)), 5)
        if route_state != "running":
            return
        if not reached:
            route_state = f"{phase}_step_budget_exhausted"
            return
        trace.append({"stage": "memory_viewpoint_reached", "phase": phase,
                      "time_s": round(time_s, 4),
                      "actor_estimated_xy_m": estimate[:2].round(5).tolist(),
                      "evaluator_actual_xy_m": actual.round(5).tolist()})
        if phase == "away" and item_state == "removed_while_away":
            model.body_pos[model.body("shown_can").id] = (-10., -10., .43)
            mujoco.mj_forward(model, data)
            trace.append({"stage": "owner_removes_item_evaluator_script",
                          "time_s": round(time_s, 4)})

    for phase, goal in phases:
        execute_phase(phase, goal)
        if route_state != "running":
            break

    if escape_target is not None:
        if route_state == "running":
            route_state = "evaded_wait_for_owner_clearance"
        if not any(row["stage"] == "predeparture_owner_help" for row in trace):
            trace.append({"stage": "predeparture_owner_help",
                          "time_s": round(time_s, 4),
                          "speech_intent": "Could you clear that moving thing before I go?",
                          "actor_reason": (watch_trigger if route_state ==
                                           "evaded_wait_for_owner_clearance" else
                                           "escape_incomplete"),
                          "actor_route_status": route_state})

    if route_state != "running":
        place_cameras()
        renderer.update_scene(data, camera="head")
        frames.append(joined.navigation_frame(
            renderer.render().copy(), grid, estimate[:2], [],
            f"memory stop: {route_state} / {last_pose_health}", motion_count))

    post_stop_min_gap = .1
    post_stop_breach = False
    reactive_escape_count = 0
    reactive_escape_time_s = None
    reactive_escape_goal = None
    held_warning_streaks = {}
    hold_missing_scan_count = 0
    held_view_probe_attempted = False
    held_view_probe = None
    held_recheck_result = None
    held_recheck_invalidated = False
    if route_state != "running" and post_stop_until_s is not None:
        sample = 0
        while time_s + SUBSTEP_S <= post_stop_until_s + 1.e-6:
            time_s = round(time_s + SUBSTEP_S, 5)
            place_cameras()
            guard_range = moving.raw_nearest_range(model, data, actual)
            gap, nearest = clearance(model, data, ape_body, ape_geom, actual)
            min_proxy_gap = min(min_proxy_gap, gap)
            post_stop_min_gap = min(post_stop_min_gap, gap)
            post_stop_breach |= gap < .05
            if held_recheck_result is not None and (
                    guard_range < moving.GUARD_CENTER_RANGE_M):
                held_recheck_invalidated = True
            trace.append({"stage": "memory_post_stop_hold",
                          "time_s": round(time_s, 4),
                          "actor_guard_range_m": guard_range,
                          "evaluator_proxy_gap_m": round(gap, 5),
                          "evaluator_nearest_geom": nearest,
                          "evaluator_actual_xy_m": actual.round(5).tolist(),
                          "evaluator_prop_xy_m": np.asarray(script(time_s)).round(5).tolist()})
            sample += 1
            if hold_monitor and (sample % int(round(hold_scan_interval_s / SUBSTEP_S)) == 0 or
                                 time_s + SUBSTEP_S > post_stop_until_s):
                if (hold_scan_dropout_window_s is not None and
                        hold_scan_dropout_window_s[0] <= time_s <
                        hold_scan_dropout_window_s[1]):
                    hold_missing_scan_count += 1
                    if held_recheck_result is not None:
                        held_recheck_invalidated = True
                    trace.append({"stage": "memory_post_stop_range_missing",
                                  "time_s": round(time_s, 4),
                                  "source_health": "missing",
                                  "actor_estimated_xy_m": estimate[:2].round(5).tolist(),
                                  "actor_last_track_capture_time_s": (
                                      grid.multi_history[-1]["capture_time_s"] if
                                      grid.multi_history else None)})
                    if head_depth_fallback and reactive_escape_count == 0:
                        if head_aim_start_s is None:
                            candidates = [track for track in
                                          grid.multi_history[-1]["tracks"]
                                          if track["status"] == "confirmed_now" and
                                          track["velocity_xy_m_s"] is not None and
                                          track["radius_m"] >= .18 and
                                          track["center_xy_m"][1] < -3.]
                            if candidates:
                                head_aim_track = min(candidates, key=lambda track:
                                    np.linalg.norm(np.asarray(track["center_xy_m"]) -
                                                   estimate[:2]))
                                center = np.asarray(head_aim_track["center_xy_m"])
                                direction = center - estimate[:2]
                                head_aim_delta_rad = -float(np.arctan2(
                                    direction[0], direction[1]))
                                head_aim_start_s = time_s
                                trace.append({"stage": "memory_head_depth_aim",
                                              "time_s": round(time_s, 4),
                                              "actor_last_range_track": head_aim_track,
                                              "actor_commanded_yaw_rad": round(
                                                  head_aim_delta_rad, 5),
                                              "actor_ready_time_s": round(time_s + .4, 4)})
                        if (head_aim_start_s is not None and
                                time_s >= head_aim_start_s + .4 - 1.e-6):
                            rgb, depth, capture_wall = owner.capture(
                                renderer, data, "head", True)
                            rgb_path = branch / f"head_depth_{time_s:.2f}_rgb.png"
                            depth_path = branch / f"head_depth_{time_s:.2f}_m.npy"
                            Image.fromarray(rgb).save(rgb_path)
                            np.save(depth_path, depth)
                            observation = head_depth_compact_surface(
                                depth, estimate[:2], model.cam_pos[head_id, 2],
                                data.cam_xmat[head_id].reshape(3, 3),
                                model.cam_fovy[head_id],
                                head_aim_track["radius_m"])
                            head_depth_capture_count += 1
                            row = {"stage": "memory_head_depth_fallback",
                                   "time_s": round(time_s, 4),
                                   "range_last_seen_s": head_aim_track["last_seen_s"],
                                   "actor_estimated_xy_m": estimate[:2].round(5).tolist(),
                                   "actor_commanded_yaw_rad": round(
                                       head_aim_delta_rad, 5),
                                   "capture_wall_s": round(capture_wall, 5),
                                   "rgb_path": str(rgb_path),
                                   "rgb_sha256": digest(rgb_path),
                                   "depth_path": str(depth_path),
                                   "depth_sha256": digest(depth_path),
                                   "actor_depth_observation": observation}
                            trace.append(row)
                            if observation is not None and previous_head_depth is not None:
                                old_time, old_center = previous_head_depth
                                center = np.asarray(observation["center_xy_m"])
                                velocity = (center - old_center) / (time_s - old_time)
                                if velocity[0] < -.30:
                                    radius = head_aim_track["radius_m"]
                                    projected_gap = min(float(np.linalg.norm(
                                        estimate[:2] - (center + velocity * dt)) -
                                        radius - .17 - .03 - .02 * dt)
                                        for dt in np.arange(0., 1.201, .1))
                                    if projected_gap < .05:
                                        perpendicular = np.array([-velocity[1],
                                                                  velocity[0]])
                                        if np.dot(perpendicular, estimate[:2] - center) < 0:
                                            perpendicular *= -1
                                        perpendicular /= np.linalg.norm(perpendicular)
                                        alternatives = []
                                        for distance in (.35, .45, .25):
                                            option = estimate[:2] + distance * perpendicular
                                            request, status, path = grid.plan(
                                                estimate[:2], option)
                                            alternatives.append({
                                                "goal_xy_m": option.round(5).tolist(),
                                                "plan_status": status,
                                                "has_request": request is not None,
                                                "path_cells": len(path)})
                                            if status == "known_goal" and request is not None:
                                                reactive_escape_goal = option
                                                break
                                        if (reactive_escape_goal is None and
                                                depth_candidate_probe):
                                            sweep = []
                                            for distance in (.15, .20, .25, .30,
                                                             .35, .40, .45):
                                                for bearing in np.linspace(
                                                        0., 2. * np.pi, 16,
                                                        endpoint=False):
                                                    option = estimate[:2] + distance * np.array(
                                                        [np.cos(bearing),
                                                         np.sin(bearing)])
                                                    request, status, path = grid.plan(
                                                        estimate[:2], option)
                                                    speed = .16
                                                    arrival = distance / speed
                                                    horizon = min(2.4,
                                                        post_stop_until_s - time_s)
                                                    future_gap = min(float(np.linalg.norm(
                                                        estimate[:2] +
                                                        min(1., dt / arrival) *
                                                        (option - estimate[:2]) -
                                                        (center + velocity * dt)) -
                                                        radius - .17 - .03 - .02 * dt)
                                                        for dt in np.arange(
                                                            0., horizon + .001, .1))
                                                    sweep.append({
                                                        "goal_xy_m": option.round(5).tolist(),
                                                        "distance_m": distance,
                                                        "bearing_rad": round(float(bearing), 5),
                                                        "plan_status": status,
                                                        "has_request": request is not None,
                                                        "path_cells": len(path),
                                                        "straight_constant_velocity_gap_m":
                                                            round(future_gap, 5)})
                                            trace.append({
                                                "stage": "memory_depth_candidate_probe",
                                                "time_s": round(time_s, 4),
                                                "actor_candidates": sweep,
                                                "probe_only": True})
                                        reactive_escape_count = 1
                                        reactive_escape_time_s = time_s
                                        trace.append({
                                            "stage": "memory_held_hazard_decision",
                                            "time_s": round(time_s, 4),
                                            "actor_source": "head_depth_after_range_loss",
                                            "actor_track": {
                                                "center_xy_m": center.round(5).tolist(),
                                                "velocity_xy_m_s": velocity.round(5).tolist(),
                                                "radius_m": radius,
                                                "range_track_id": head_aim_track["id"]},
                                            "actor_projected_gap_m": round(
                                                projected_gap, 5),
                                            "actor_candidate_goals": alternatives,
                                            "actor_selected_goal_xy_m": (
                                                None if reactive_escape_goal is None else
                                                reactive_escape_goal.round(5).tolist())})
                                        if reactive_escape_goal is not None:
                                            route_state = "running"
                                            if head_depth_processing_latency_s > 0.:
                                                latency_start_s = time_s
                                                latency_steps = int(np.ceil(
                                                    head_depth_processing_latency_s /
                                                    SUBSTEP_S))
                                                for _ in range(latency_steps):
                                                    time_s = round(time_s + SUBSTEP_S, 5)
                                                    place_cameras()
                                                    guard_range = moving.raw_nearest_range(
                                                        model, data, actual)
                                                    gap, nearest = clearance(
                                                        model, data, ape_body,
                                                        ape_geom, actual)
                                                    min_proxy_gap = min(min_proxy_gap, gap)
                                                    post_stop_min_gap = min(
                                                        post_stop_min_gap, gap)
                                                    post_stop_breach |= gap < .05
                                                    trace.append({
                                                        "stage": "memory_post_stop_hold",
                                                        "time_s": round(time_s, 4),
                                                        "actor_guard_range_m": guard_range,
                                                        "evaluator_proxy_gap_m": round(gap, 5),
                                                        "evaluator_nearest_geom": nearest,
                                                        "evaluator_actual_xy_m":
                                                            actual.round(5).tolist(),
                                                        "evaluator_prop_xy_m": np.asarray(
                                                            script(time_s)).round(5).tolist(),
                                                        "actor_depth_processing_wait": True})
                                                    sample += 1
                                                trace.append({
                                                    "stage": "memory_head_depth_processing_wait",
                                                    "time_s": round(time_s, 4),
                                                    "actor_capture_decision_time_s": round(
                                                        latency_start_s, 4),
                                                    "requested_latency_s":
                                                        head_depth_processing_latency_s,
                                                    "elapsed_s": round(
                                                        time_s - latency_start_s, 4)})
                                                if guard_range < moving.GUARD_CENTER_RANGE_M:
                                                    route_state = "near_body_stop"
                                            reactive_escape_time_s = time_s
                                            if route_state == "running":
                                                execute_phase("evade_reactive",
                                                              reactive_escape_goal)
                                            if route_state == "running":
                                                route_state = "evaded_wait_for_owner_clearance"
                                            trace.append({
                                                "stage": "memory_held_reaction_result",
                                                "time_s": round(time_s, 4),
                                                "actor_route_status": route_state,
                                                "actor_estimated_xy_m":
                                                estimate[:2].round(5).tolist()})
                            if observation is not None:
                                previous_head_depth = (
                                    time_s,
                                    np.asarray(observation["center_xy_m"]))
                    if sample % 20 == 0 or time_s + SUBSTEP_S > post_stop_until_s:
                        renderer.update_scene(data, camera="head")
                        frames.append(joined.navigation_frame(
                            renderer.render().copy(), grid, estimate[:2], [],
                            f"held with range missing: {route_state}", motion_count))
                    continue
                scan = joined.SCAN(model, data, actual, 5., rng)
                grid.next_capture_time_s = time_s
                grid.update(scan, estimate[:2], 5.)
                if dynamic_trail_reopen:
                    trail_update = dynamic_trail_map.reopen_observed_trail(
                        grid, scan, estimate[:2], time_s,
                        grid.multi_history, 5.,
                        route.RADIUS_M + route.MARGIN_M)
                    dynamic_trail_reclaimed_cells += (
                        trail_update["reclaimed_cell_count"])
                    trace.append({"stage": "memory_dynamic_trail_reopen",
                                  **trail_update})
                trace.append({"stage": "memory_post_stop_range_scan",
                              "time_s": round(time_s, 4),
                              "range_72_beams_m": scan,
                              "actor_estimated_xy_m": estimate[:2].round(5).tolist(),
                              "actor_multi_tracks": grid.multi_history[-1]["tracks"]})
                if held_recheck_result is not None:
                    for track in grid.multi_history[-1]["tracks"]:
                        velocity = track["velocity_xy_m_s"]
                        if (track["status"] != "confirmed_now" or
                                track["radius_m"] < .18 or velocity is None or
                                np.linalg.norm(velocity) < .05):
                            continue
                        support = held_pose_screen.screen(
                            path_xy_m=[estimate[:2].tolist()] * 2,
                            track_center_xy_m=track["center_xy_m"],
                            track_velocity_xy_m_s=velocity,
                            track_radius_m=track["radius_m"],
                            track_capture_time_s=track["last_seen_s"],
                            decision_time_s=time_s,
                            hold_until_s=min(time_s + 2.,
                                             post_stop_until_s + 2.),
                            pose_error_m=.05)
                        if not support["supported_under_models"]:
                            held_recheck_invalidated = True
                            trace.append({
                                "stage": "memory_held_view_recheck_revoked",
                                "time_s": round(time_s, 4),
                                "actor_track_id": track["id"],
                                "actor_hold_screen": support})
                if held_reactive_escape and reactive_escape_count == 0:
                    for moving_track in grid.multi_history[-1]["tracks"]:
                        velocity = moving_track["velocity_xy_m_s"]
                        if (moving_track["status"] != "confirmed_now" or
                                velocity is None or moving_track["radius_m"] < .18):
                            continue
                        velocity = np.asarray(velocity, dtype=float)
                        center = np.asarray(moving_track["center_xy_m"], dtype=float)
                        toward_ape = estimate[:2] - center
                        closing_speed = float(np.dot(velocity, toward_ape) /
                                              max(np.linalg.norm(toward_ape), 1.e-6))
                        if closing_speed < .1:
                            continue
                        projected_gap = min(
                            float(np.linalg.norm(estimate[:2] -
                                                 (center + velocity * dt)) -
                                  moving_track["radius_m"] - .17 - .03 - .02 * dt)
                            for dt in np.arange(0., 1.201, .1))
                        if projected_gap >= .05:
                            continue
                        if held_warning_confirmations > 1:
                            previous = held_warning_streaks.get(moving_track["id"])
                            streak = (previous[1] + 1 if previous is not None and
                                      abs(time_s - previous[0] -
                                          hold_scan_interval_s) < .001 else 1)
                            held_warning_streaks[moving_track["id"]] = (
                                time_s, streak)
                            trace.append({
                                "stage": "memory_held_hazard_warning",
                                "time_s": round(time_s, 4),
                                "actor_track": moving_track,
                                "actor_projected_gap_m": round(projected_gap, 5),
                                "actor_warning_streak": streak,
                                "actor_confirmations_required":
                                    held_warning_confirmations})
                            if streak < held_warning_confirmations:
                                continue
                        perpendicular = np.array([-velocity[1], velocity[0]])
                        if np.dot(perpendicular, estimate[:2] - center) < 0:
                            perpendicular *= -1
                        perpendicular /= np.linalg.norm(perpendicular)
                        alternatives = []
                        selected_support_status = "none"
                        emergency_candidates = []
                        if future_hold_screen:
                            for distance in (.25, .35, .45):
                                for bearing_index in range(8):
                                    bearing = 2. * np.pi * bearing_index / 8
                                    option = estimate[:2] + distance * np.array(
                                        [np.cos(bearing), np.sin(bearing)])
                                    request, status, path = grid.plan(
                                        estimate[:2], option)
                                    candidate = {
                                        "goal_xy_m": option.round(5).tolist(),
                                        "plan_status": status,
                                        "has_request": request is not None,
                                        "path_cells": len(path)}
                                    if status == "known_goal" and request is not None:
                                        path_xy = [estimate[:2].tolist()] + [
                                            grid.xy(cell).tolist() for cell in path[1:]]
                                        candidate["future_hold_screen"] = (
                                            held_pose_screen.screen(
                                                path_xy_m=path_xy,
                                                track_center_xy_m=center.tolist(),
                                                track_velocity_xy_m_s=velocity.tolist(),
                                                track_radius_m=moving_track["radius_m"],
                                                track_capture_time_s=moving_track["last_seen_s"],
                                                decision_time_s=time_s,
                                                hold_until_s=post_stop_until_s,
                                                pose_error_m=(.05 if source == "range"
                                                              else .12)))
                                    alternatives.append(candidate)
                            viable = [candidate for candidate in alternatives
                                      if candidate.get("future_hold_screen", {}).get(
                                          "supported_under_models")]
                            if viable:
                                selected = max(viable, key=lambda candidate: (
                                    candidate["future_hold_screen"]["worst_modeled_gap_m"],
                                    -candidate["future_hold_screen"]["path_length_m"]))
                                reactive_escape_goal = np.asarray(
                                    selected["goal_xy_m"], dtype=float)
                                selected_support_status = "modeled_supported"
                            elif best_effort_if_unsupported:
                                if emergency_candidate_strategy in (
                                        "nominal_ranked", "nominal_depth_veto"):
                                    emergency_candidates = nominal_emergency_rank.rank(
                                        alternatives, moving_track, time_s,
                                        post_stop_until_s,
                                        .05 if source == "range" else .12)
                                    if emergency_candidate_strategy == "nominal_depth_veto":
                                        place_cameras()
                                        rgb_upper, depth_upper, capture_wall_s = (
                                            owner.capture(renderer, data, "head", True))
                                        upper_points = head_depth_upper_route.observed_upper_points(
                                            depth_upper, estimate[:2],
                                            float(model.cam_pos[head_id, 2]),
                                            data.cam_xmat[head_id].reshape(3, 3).copy(),
                                            float(model.cam_fovy[head_id]))
                                        upper_path = branch / (
                                            f"head-upper-{time_s:.2f}s.png")
                                        upper_depth_path = branch / (
                                            f"head-upper-{time_s:.2f}s-depth.npy")
                                        Image.fromarray(rgb_upper).save(upper_path)
                                        np.save(upper_depth_path, depth_upper)
                                        trace.append({
                                            "stage": "memory_head_upper_depth_capture",
                                            "time_s": round(time_s, 4),
                                            "source_health": "fresh" if len(upper_points)
                                                else "no_upper_points",
                                            "capture_wall_s": round(capture_wall_s, 5),
                                            "actor_estimated_xy_m":
                                                estimate[:2].round(5).tolist(),
                                            "head_camera_z_m":
                                                round(float(model.cam_pos[head_id, 2]), 5),
                                            "head_camera_rotation":
                                                data.cam_xmat[head_id].reshape(3, 3).round(7).tolist(),
                                            "head_fovy_deg":
                                                round(float(model.cam_fovy[head_id]), 5),
                                            "upper_point_count": len(upper_points),
                                            "rgb_path": str(upper_path),
                                            "rgb_sha256": digest(upper_path),
                                            "depth_path": str(upper_depth_path),
                                            "depth_sha256": digest(upper_depth_path)})
                                        for candidate in emergency_candidates:
                                            candidate["upper_depth_route"] = (
                                                head_depth_upper_route.route_upper_gap(
                                                    upper_points,
                                                    candidate["future_hold_screen"]["path_xy_m"]))
                                        available_emergency_candidates = [
                                            candidate for candidate in emergency_candidates
                                            if candidate["upper_depth_route"][
                                                "minimum_observed_gap_m"] is not None and
                                            candidate["upper_depth_route"][
                                                "minimum_observed_gap_m"] >= .05]
                                    else:
                                        available_emergency_candidates = emergency_candidates
                                    if available_emergency_candidates:
                                        reactive_escape_goal = np.asarray(
                                            available_emergency_candidates[0]["goal_xy_m"])
                                        selected_support_status = (
                                            "unqualified_depth_veto_known_free"
                                            if emergency_candidate_strategy ==
                                            "nominal_depth_veto" else
                                            "unqualified_nominal_ranked_known_free")
                                else:
                                    for distance in (.35, .45, .25):
                                        option = estimate[:2] + distance * perpendicular
                                        request, status, path = grid.plan(
                                            estimate[:2], option)
                                        emergency_candidates.append({
                                            "goal_xy_m": option.round(5).tolist(),
                                            "plan_status": status,
                                            "has_request": request is not None,
                                            "path_cells": len(path)})
                                        if status == "known_goal" and request is not None:
                                            reactive_escape_goal = option
                                            selected_support_status = (
                                                "unqualified_emergency_known_free")
                                            break
                        else:
                            for distance in (.35, .45, .25):
                                option = estimate[:2] + distance * perpendicular
                                request, status, path = grid.plan(estimate[:2], option)
                                alternatives.append({"goal_xy_m": option.round(5).tolist(),
                                                     "plan_status": status,
                                                     "has_request": request is not None,
                                                     "path_cells": len(path)})
                                if status == "known_goal" and request is not None:
                                    reactive_escape_goal = option
                                    selected_support_status = "current_grid_only"
                                    break
                        reactive_escape_count = 1
                        reactive_escape_time_s = time_s
                        map_snapshot = None
                        if map_snapshot_at_hazard:
                            map_snapshot = {
                                "xs_m": grid.xs.round(5).tolist(),
                                "ys_m": grid.ys.round(5).tolist(),
                                "free_indices_yx": np.argwhere(grid.free).tolist(),
                                "occupied_indices_yx": np.argwhere(
                                    grid.occupied).tolist()}
                        trace.append({"stage": "memory_held_hazard_decision",
                                      "time_s": round(time_s, 4),
                                      "actor_track": moving_track,
                                      "actor_projected_gap_m": round(projected_gap, 5),
                                      "actor_future_hold_screen_requested":
                                          future_hold_screen,
                                      "actor_candidate_goals": alternatives,
                                      "actor_emergency_candidates": emergency_candidates,
                                      "actor_selected_goal_support_status":
                                          selected_support_status,
                                      "actor_map_snapshot": map_snapshot,
                                      "actor_selected_goal_xy_m": (
                                          None if reactive_escape_goal is None else
                                          reactive_escape_goal.round(5).tolist())})
                        upper_wait_guard_stop = False
                        if (emergency_candidate_strategy == "nominal_depth_veto" and
                                head_upper_processing_latency_s > 0.):
                            capture_time_s = time_s
                            for _ in range(int(round(
                                    head_upper_processing_latency_s / SUBSTEP_S))):
                                time_s = round(time_s + SUBSTEP_S, 5)
                                place_cameras()
                                guard_range = moving.raw_nearest_range(
                                    model, data, actual)
                                gap, nearest = clearance(
                                    model, data, ape_body, ape_geom, actual)
                                min_proxy_gap = min(min_proxy_gap, gap)
                                post_stop_min_gap = min(post_stop_min_gap, gap)
                                post_stop_breach |= gap < .05
                                trace.append({
                                    "stage": "memory_post_stop_hold",
                                    "time_s": round(time_s, 4),
                                    "actor_guard_range_m": guard_range,
                                    "evaluator_proxy_gap_m": round(gap, 5),
                                    "evaluator_nearest_geom": nearest,
                                    "evaluator_actual_xy_m": actual.round(5).tolist(),
                                    "evaluator_prop_xy_m": np.asarray(
                                        script(time_s)).round(5).tolist(),
                                    "actor_head_upper_processing_wait": True})
                                sample += 1
                                if guard_range < moving.GUARD_CENTER_RANGE_M:
                                    route_state = "near_body_stop"
                                    upper_wait_guard_stop = True
                                    break
                            trace.append({
                                "stage": "memory_head_upper_processing_wait",
                                "actor_capture_decision_time_s": round(
                                    capture_time_s, 4),
                                "time_s": round(time_s, 4),
                                "requested_latency_s": head_upper_processing_latency_s,
                                "elapsed_s": round(time_s - capture_time_s, 4),
                                "actor_capture_age_s_at_dispatch": round(
                                    time_s - capture_time_s, 4),
                                "actor_guard_interrupted": upper_wait_guard_stop,
                                "actor_range_track_refreshed": False})
                            reactive_escape_time_s = time_s
                        if reactive_escape_goal is not None and not upper_wait_guard_stop:
                            route_state = "running"
                            execute_phase("evade_reactive", reactive_escape_goal)
                            if route_state == "running":
                                route_state = (
                                    "evaded_unqualified_wait_for_owner_clearance"
                                    if selected_support_status ==
                                    "unqualified_emergency_known_free" else
                                    "evaded_wait_for_owner_clearance")
                            trace.append({"stage": "memory_held_reaction_result",
                                          "time_s": round(time_s, 4),
                                          "actor_route_status": route_state,
                                          "actor_selected_goal_support_status":
                                              selected_support_status,
                                          "actor_estimated_xy_m": estimate[:2].round(5).tolist()})
                        elif upper_wait_guard_stop:
                            trace.append({
                                "stage": "memory_held_reaction_result",
                                "time_s": round(time_s, 4),
                                "actor_route_status": route_state,
                                "actor_selected_goal_support_status":
                                    selected_support_status,
                                "actor_estimated_xy_m":
                                    estimate[:2].round(5).tolist()})
                        elif future_hold_screen:
                            route_state = "no_actor_supported_held_pose"
                            trace.append({
                                "stage": "memory_held_reaction_result",
                                "time_s": round(time_s, 4),
                                "actor_route_status": route_state,
                                "actor_estimated_xy_m": estimate[:2].round(5).tolist(),
                                "speech_intent":
                                    "Please help; I cannot find a supported place to wait."})
                        break
            if sample % 20 == 0 or time_s + SUBSTEP_S > post_stop_until_s:
                renderer.update_scene(data, camera="head")
                frames.append(joined.navigation_frame(
                    renderer.render().copy(), grid, estimate[:2], [],
                    f"held after {route_state}: gap {gap:.2f} m", motion_count))
            if (held_view_probe_at_s is not None and
                    not held_view_probe_attempted and
                    time_s >= held_view_probe_at_s and
                    time_s + 2. <= post_stop_until_s):
                held_view_probe_attempted = True
                scan = joined.SCAN(model, data, actual, 5., rng)
                estimate[:], localization = localizer.correct(scan, estimate)
                last_pose_health = localization["status"]
                grid.next_capture_time_s = time_s
                grid.update(scan, estimate[:2], 5.)
                raw_range = moving.raw_nearest_range(model, data, actual)
                recent = grid.multi_history[-1]["tracks"]
                moving_tracks = [track for track in recent
                                 if track["status"] == "confirmed_now" and
                                 track["radius_m"] >= .18 and
                                 track["velocity_xy_m_s"] is not None and
                                 np.linalg.norm(track["velocity_xy_m_s"]) >= .05 and
                                 time_s - track["last_seen_s"] <= .11]
                screens = [{"track_id": track["id"],
                            "screen": held_pose_screen.screen(
                                path_xy_m=[estimate[:2].tolist()] * 2,
                                track_center_xy_m=track["center_xy_m"],
                                track_velocity_xy_m_s=track["velocity_xy_m_s"],
                                track_radius_m=track["radius_m"],
                                track_capture_time_s=track["last_seen_s"],
                                decision_time_s=time_s,
                                hold_until_s=min(time_s + 2., post_stop_until_s),
                                pose_error_m=.05)} for track in moving_tracks]
                gate = (source == "range" and
                        last_pose_health in ("mapped", "registered") and
                        np.linalg.norm(estimate[:2] - return_view) >= .20 and
                        raw_range >= held_view_min_raw_range_m and
                        hold_missing_scan_count == 0 and
                        all(row["screen"]["supported_under_models"]
                            for row in screens))
                row = {"stage": "memory_held_view_probe_gate",
                       "time_s": round(time_s, 4),
                       "actor_estimated_xy_m": estimate[:2].round(5).tolist(),
                       "memory_show_view_xy_m": return_view.round(5).tolist(),
                       "actor_departure_distance_m": round(float(
                           np.linalg.norm(estimate[:2] - return_view)), 5),
                       "range_72_beams_m": scan,
                       "source_health": last_pose_health,
                       "localization": localization,
                       "actor_guard_range_m": raw_range,
                       "actor_view_min_raw_range_m": held_view_min_raw_range_m,
                       "actor_moving_tracks": moving_tracks,
                       "actor_hold_screens": screens,
                       "actor_view_allowed": gate}
                trace.append(row)
                if gate:
                    interaction.aim_camera(model, interaction.CAMERA,
                                           estimate[:2], item_memory[:2], 20)
                    place_cameras()
                    rgb, depth, capture_wall = owner.capture(
                        renderer, data, interaction.CAMERA, True)
                    capture_time_s = time_s
                    image = Image.fromarray(rgb)
                    image_path = branch / "held-view-probe.png"
                    depth_path = branch / "held-view-probe-depth.npz"
                    image.save(image_path)
                    np.savez_compressed(depth_path, depth_m=depth)

                    def match_image():
                        proposals = vision.detect(image)
                        scores = (vision.feature([image.crop(item["box"])
                                                  for item in proposals]) @ exemplar
                                  if proposals else np.empty(0))
                        return proposals, scores

                    worker_alarm = False
                    worker_track_alarm = False
                    worker_source_missing = False
                    worker_response_probe = None
                    with ThreadPoolExecutor(max_workers=1) as pool:
                        future = pool.submit(match_image)
                        worker_samples = 0
                        while not future.done():
                            time.sleep(SUBSTEP_S)
                            time_s = round(time_s + SUBSTEP_S, 5)
                            worker_samples += 1
                            place_cameras()
                            near_range = moving.raw_nearest_range(
                                model, data, actual)
                            gap, nearest = clearance(
                                model, data, ape_body, ape_geom, actual)
                            min_proxy_gap = min(min_proxy_gap, gap)
                            post_stop_min_gap = min(post_stop_min_gap, gap)
                            post_stop_breach |= gap < .05
                            worker_alarm |= near_range < moving.GUARD_CENTER_RANGE_M
                            trace.append({"stage": "memory_held_view_worker_monitor",
                                          "time_s": round(time_s, 4),
                                          "actor_guard_range_m": near_range,
                                          "actor_alarm": worker_alarm,
                                          "evaluator_proxy_gap_m": round(gap, 5),
                                          "evaluator_nearest_geom": nearest,
                                          "evaluator_actual_xy_m":
                                              actual.round(5).tolist(),
                                          "evaluator_prop_xy_m": np.asarray(
                                              script(time_s)).round(5).tolist()})
                            if worker_samples % int(round(
                                    hold_scan_interval_s / SUBSTEP_S)) == 0:
                                if (held_view_worker_range_dropout_s is not None and
                                        held_view_worker_range_dropout_s[0] <=
                                        time_s - capture_time_s <
                                        held_view_worker_range_dropout_s[1]):
                                    worker_source_missing = True
                                    hold_missing_scan_count += 1
                                    trace.append({
                                        "stage": "memory_held_view_worker_range_missing",
                                        "time_s": round(time_s, 4),
                                        "source_health": "missing",
                                        "actor_estimated_xy_m":
                                            estimate[:2].round(5).tolist(),
                                        "actor_last_track_capture_time_s":
                                            grid.multi_history[-1][
                                                "capture_time_s"] if
                                                grid.multi_history else None})
                                    continue
                                worker_scan = joined.SCAN(
                                    model, data, actual, 5., rng)
                                grid.next_capture_time_s = time_s
                                grid.update(worker_scan, estimate[:2], 5.)
                                worker_tracks = grid.multi_history[-1]["tracks"]
                                worker_screens = []
                                for worker_track in worker_tracks:
                                    velocity = worker_track["velocity_xy_m_s"]
                                    if (worker_track["status"] != "confirmed_now" or
                                            worker_track["radius_m"] < .18 or
                                            velocity is None or
                                            np.linalg.norm(velocity) < .05):
                                        continue
                                    support = held_pose_screen.screen(
                                        path_xy_m=[estimate[:2].tolist()] * 2,
                                        track_center_xy_m=worker_track[
                                            "center_xy_m"],
                                        track_velocity_xy_m_s=velocity,
                                        track_radius_m=worker_track["radius_m"],
                                        track_capture_time_s=worker_track[
                                            "last_seen_s"],
                                        decision_time_s=time_s,
                                        hold_until_s=time_s + 2.,
                                        pose_error_m=.05)
                                    worker_screens.append({"track_id":
                                                           worker_track["id"],
                                                           "screen": support})
                                    worker_track_alarm |= not support[
                                        "supported_under_models"]
                                trace.append({
                                    "stage": "memory_held_view_worker_range_scan",
                                    "time_s": round(time_s, 4),
                                    "range_72_beams_m": worker_scan,
                                    "actor_estimated_xy_m":
                                        estimate[:2].round(5).tolist(),
                                    "actor_multi_tracks":
                                        worker_tracks,
                                    "actor_hold_screens": worker_screens,
                                    "actor_track_alarm": worker_track_alarm})
                                if ((held_view_response_probe or
                                     held_view_emergency) and
                                        worker_track_alarm and
                                        worker_response_probe is None):
                                    failing = next((item for item in
                                                    worker_screens if not item[
                                                        "screen"][
                                                            "supported_under_models"]),
                                                   None)
                                    threat = next((item for item in worker_tracks
                                                   if failing is not None and
                                                   item["id"] == failing[
                                                       "track_id"]), None)
                                    person_view = next((item for item in
                                                        reversed(owner_trace)
                                                        if item.get("actor", {}).get(
                                                            "point_xyz_est_m") is not None),
                                                       None)
                                    person_xy = (None if person_view is None else
                                                 person_view["actor"][
                                                     "point_xyz_est_m"][:2])
                                    goal, evidence = actor_range_refuge_v744.choose(
                                        grid, estimate[:2], threat,
                                        threat["velocity_xy_m_s"], person_xy,
                                        allow_best_effort=held_view_emergency,
                                        require_initial_radial_away=
                                            held_view_directional_emergency)
                                    worker_response_probe = {
                                        "stage": "memory_held_view_response_probe",
                                        "time_s": round(time_s, 4),
                                        "probe_only": not held_view_emergency,
                                        "actor_threat_track": threat,
                                        "actor_person_surface_xy_m": person_xy,
                                        "actor_estimated_xy_m":
                                            estimate[:2].round(5).tolist(),
                                        "actor_chosen_goal_xy_m": None if goal is
                                            None else goal.round(5).tolist(),
                                        "actor_selection": evidence,
                                        "planning_wall_s_not_sim_advanced":
                                            evidence.get("selection_wall_s")}
                                    trace.append(worker_response_probe)
                                    if held_view_emergency and goal is not None:
                                        route_state = "running"
                                        reactive_escape_goal = goal.copy()
                                        reactive_escape_time_s = time_s
                                        reactive_escape_count += 1
                                        execute_phase("evade_reactive", goal)
                                        if route_state == "running":
                                            route_state = (
                                                "evaded_unqualified_wait_for_owner_clearance"
                                                if evidence["status"] ==
                                                "unqualified_best_effort" else
                                                "evaded_wait_for_owner_clearance")
                                        worker_response_probe[
                                            "actor_executed_route_status"] = route_state
                                        worker_response_probe[
                                            "actor_end_estimated_xy_m"] = (
                                                estimate[:2].round(5).tolist())
                                        trace.append({
                                            "stage": "memory_held_view_response_result",
                                            "time_s": round(time_s, 4),
                                            "actor_route_status": route_state,
                                            "actor_selected_support_status":
                                                evidence["status"],
                                            "actor_end_estimated_xy_m":
                                                estimate[:2].round(5).tolist()})
                        proposals, scores = future.result()
                    ranked = sorted(zip(proposals, scores),
                                    key=lambda pair: float(pair[1]), reverse=True)
                    choice = (ranked[0] if ranked and
                              float(ranked[0][1]) >= owner.THRESHOLD else None)
                    box = None if choice is None else choice[0]["box"]
                    point = (None if box is None else
                             interaction.depth_point(box, depth, data, model,
                                                     interaction.CAMERA))
                    if point is not None:
                        point = np.asarray(point, dtype=float)
                        point[:2] += estimate[:2] - actual
                    gt_box, _ = interaction.evaluator.evaluator_box(
                        renderer, data, model, interaction.CAMERA,
                        "shown_can_geom")
                    held_view_probe = {
                        "stage": "memory_held_view_probe_result",
                        "capture_time_s": round(capture_time_s, 4),
                        "recognition_end_time_s": round(time_s, 4),
                        "worker_alarm": worker_alarm,
                        "worker_track_alarm": worker_track_alarm,
                        "worker_source_missing": worker_source_missing,
                        "worker_response_probe": worker_response_probe,
                        "candidate_count": len(proposals),
                        "top_similarity": None if not ranked else float(ranked[0][1]),
                        "selected_box": box,
                        "actor_depth_item_xyz_m": None if point is None else
                            point.round(5).tolist(),
                        "actor_memory_distance_m": None if point is None else
                            round(float(np.linalg.norm(point[:2] -
                                                       item_memory[:2])), 5),
                        "item_box_iou_evaluator_only": owner.iou(box, gt_box),
                        "image_path": str(image_path),
                        "image_sha256": digest(image_path),
                        "depth_path": str(depth_path),
                        "depth_sha256": digest(depth_path),
                        "probe_only": True}
                    trace.append(held_view_probe)
                    frames.append(owner.owner_frame(
                        rgb, "held station | fresh view probe", box, gt_box))
                    if held_view_recheck:
                        first_consistent = bool(
                            box is not None and point is not None and
                            not worker_alarm and not worker_track_alarm and
                            not worker_source_missing and
                            np.linalg.norm(point[:2] - item_memory[:2]) <=
                            MEMORY_ASSOCIATION_M)
                        recheck = {
                            "stage": "memory_held_view_recheck_result",
                            "time_s": round(time_s, 4),
                            "first_view_consistent": first_consistent,
                            "actor_claim_at_recheck": False,
                            "reason": "first_view_unconfirmed"}
                        if first_consistent:
                            tracker = owner.FastActor(
                                cv2.cvtColor(np.asarray(image.crop(box)),
                                             cv2.COLOR_RGB2BGR), use_depth=False)
                            recheck_alarm = False
                            for _ in range(int(round(.1 / SUBSTEP_S))):
                                time_s = round(time_s + SUBSTEP_S, 5)
                                place_cameras()
                                near_range = moving.raw_nearest_range(
                                    model, data, actual)
                                gap, nearest = clearance(
                                    model, data, ape_body, ape_geom, actual)
                                min_proxy_gap = min(min_proxy_gap, gap)
                                post_stop_min_gap = min(post_stop_min_gap, gap)
                                post_stop_breach |= gap < .05
                                recheck_alarm |= (
                                    near_range < moving.GUARD_CENTER_RANGE_M)
                                trace.append({
                                    "stage": "memory_held_view_recheck_monitor",
                                    "time_s": round(time_s, 4),
                                    "actor_guard_range_m": near_range,
                                    "evaluator_proxy_gap_m": round(gap, 5),
                                    "evaluator_nearest_geom": nearest,
                                    "evaluator_actual_xy_m":
                                        actual.round(5).tolist(),
                                    "evaluator_prop_xy_m": np.asarray(
                                        script(time_s)).round(5).tolist()})
                            fresh_scan = joined.SCAN(
                                model, data, actual, 5., rng)
                            estimate[:], fresh_localization = localizer.correct(
                                fresh_scan, estimate)
                            last_pose_health = fresh_localization["status"]
                            grid.next_capture_time_s = time_s
                            grid.update(fresh_scan, estimate[:2], 5.)
                            tracking_camera_x_est = float(estimate[0])
                            fresh_rgb, fresh_depth, fresh_capture_wall = (
                                owner.capture(renderer, data,
                                              interaction.CAMERA, True))
                            started = time.perf_counter()
                            tracked = tracker.observe(
                                Image.fromarray(fresh_rgb), None,
                                tracking_camera_x_est)
                            fresh_age_s = (fresh_capture_wall +
                                           time.perf_counter() - started)
                            fresh_status = tracked["status"]
                            fresh_box = (tracked["box"] if
                                         fresh_age_s <= owner.MAX_FRESH_AGE_S
                                         else None)
                            fresh_point = (None if fresh_box is None else
                                           interaction.depth_point(
                                               fresh_box, fresh_depth, data,
                                               model, interaction.CAMERA))
                            if fresh_point is not None:
                                fresh_point = np.asarray(
                                    fresh_point, dtype=float)
                                fresh_point[:2] += estimate[:2] - actual
                            fresh_gt_box, _ = interaction.evaluator.evaluator_box(
                                renderer, data, model, interaction.CAMERA,
                                "shown_can_geom")
                            fresh_image_path = branch / "held-view-recheck.png"
                            fresh_depth_path = branch / "held-view-recheck-depth.npz"
                            Image.fromarray(fresh_rgb).save(fresh_image_path)
                            np.savez_compressed(
                                fresh_depth_path, depth_m=fresh_depth)
                            recheck.update({
                                "time_s": round(time_s, 4),
                                "reason": "fresh_view_result",
                                "recheck_alarm": recheck_alarm,
                                "fresh_source_health": last_pose_health,
                                "fresh_localization": fresh_localization,
                                "fresh_range_72_beams_m": fresh_scan,
                                "fresh_track_status": fresh_status,
                                "fresh_frame_age_s": fresh_age_s,
                                "fresh_box": fresh_box,
                                "two_view_iou": owner.iou(box, fresh_box),
                                "fresh_depth_item_xyz_m": None if
                                    fresh_point is None else
                                    fresh_point.round(5).tolist(),
                                "actor_memory_distance_m": None if
                                    fresh_point is None else round(float(
                                        np.linalg.norm(fresh_point[:2] -
                                                       item_memory[:2])), 5),
                                "item_box_iou_evaluator_only": owner.iou(
                                    fresh_box, fresh_gt_box),
                                "fresh_item_error_m_evaluator_only": None if
                                    fresh_point is None else float(np.linalg.norm(
                                        fresh_point - model.body_pos[
                                            model.body("shown_can").id])),
                                "image_path": str(fresh_image_path),
                                "image_sha256": digest(fresh_image_path),
                                "depth_path": str(fresh_depth_path),
                                "depth_sha256": digest(fresh_depth_path)})
                            recheck["actor_claim_at_recheck"] = bool(
                                not recheck_alarm and
                                last_pose_health == "registered" and
                                fresh_status ==
                                    "visually_reacquired_unlocalized" and
                                fresh_box is not None and
                                owner.iou(box, fresh_box) >= .6 and
                                fresh_point is not None and
                                np.linalg.norm(fresh_point[:2] -
                                               item_memory[:2]) <=
                                    MEMORY_ASSOCIATION_M)
                            frames.append(owner.owner_frame(
                                fresh_rgb, "held station | fresh recheck",
                                fresh_box, fresh_gt_box))
                            recheck["fresh_views"] = [{
                                "index": 1,
                                "time_s": round(time_s, 4),
                                "track_status": fresh_status,
                                "image_path": str(fresh_image_path),
                                "image_sha256": digest(fresh_image_path),
                                "depth_path": str(fresh_depth_path),
                                "depth_sha256": digest(fresh_depth_path)}]
                            if (fresh_status == "candidate_unconfirmed" and
                                    not recheck_alarm):
                                for _ in range(int(round(.1 / SUBSTEP_S))):
                                    time_s = round(time_s + SUBSTEP_S, 5)
                                    place_cameras()
                                    near_range = moving.raw_nearest_range(
                                        model, data, actual)
                                    gap, nearest = clearance(
                                        model, data, ape_body, ape_geom, actual)
                                    min_proxy_gap = min(min_proxy_gap, gap)
                                    post_stop_min_gap = min(post_stop_min_gap, gap)
                                    post_stop_breach |= gap < .05
                                    recheck_alarm |= (
                                        near_range < moving.GUARD_CENTER_RANGE_M)
                                    trace.append({
                                        "stage": "memory_held_view_recheck_monitor",
                                        "time_s": round(time_s, 4),
                                        "actor_guard_range_m": near_range,
                                        "evaluator_proxy_gap_m": round(gap, 5),
                                        "evaluator_nearest_geom": nearest,
                                        "evaluator_actual_xy_m":
                                            actual.round(5).tolist(),
                                        "evaluator_prop_xy_m": np.asarray(
                                            script(time_s)).round(5).tolist()})
                                fresh_scan = joined.SCAN(
                                    model, data, actual, 5., rng)
                                estimate[:], fresh_localization = (
                                    localizer.correct(fresh_scan, estimate))
                                last_pose_health = fresh_localization["status"]
                                grid.next_capture_time_s = time_s
                                grid.update(fresh_scan, estimate[:2], 5.)
                                fresh_rgb, fresh_depth, fresh_capture_wall = (
                                    owner.capture(renderer, data,
                                                  interaction.CAMERA, True))
                                started = time.perf_counter()
                                tracked = tracker.observe(
                                    Image.fromarray(fresh_rgb), None,
                                    tracking_camera_x_est)
                                fresh_age_s = (fresh_capture_wall +
                                               time.perf_counter() - started)
                                fresh_status = tracked["status"]
                                fresh_box = (tracked["box"] if
                                             fresh_age_s <= owner.MAX_FRESH_AGE_S
                                             else None)
                                fresh_point = (None if fresh_box is None else
                                               interaction.depth_point(
                                                   fresh_box, fresh_depth, data,
                                                   model, interaction.CAMERA))
                                if fresh_point is not None:
                                    fresh_point = np.asarray(
                                        fresh_point, dtype=float)
                                    fresh_point[:2] += estimate[:2] - actual
                                fresh_gt_box, _ = (
                                    interaction.evaluator.evaluator_box(
                                        renderer, data, model,
                                        interaction.CAMERA, "shown_can_geom"))
                                fresh_image_path = (
                                    branch / "held-view-recheck-02.png")
                                fresh_depth_path = (
                                    branch / "held-view-recheck-02-depth.npz")
                                Image.fromarray(fresh_rgb).save(fresh_image_path)
                                np.savez_compressed(
                                    fresh_depth_path, depth_m=fresh_depth)
                                recheck.update({
                                    "time_s": round(time_s, 4),
                                    "recheck_alarm": recheck_alarm,
                                    "fresh_source_health": last_pose_health,
                                    "fresh_localization": fresh_localization,
                                    "fresh_range_72_beams_m": fresh_scan,
                                    "fresh_track_status": fresh_status,
                                    "fresh_frame_age_s": fresh_age_s,
                                    "fresh_box": fresh_box,
                                    "two_view_iou": owner.iou(box, fresh_box),
                                    "fresh_depth_item_xyz_m": None if
                                        fresh_point is None else
                                        fresh_point.round(5).tolist(),
                                    "actor_memory_distance_m": None if
                                        fresh_point is None else round(float(
                                            np.linalg.norm(fresh_point[:2] -
                                                           item_memory[:2])), 5),
                                    "item_box_iou_evaluator_only": owner.iou(
                                        fresh_box, fresh_gt_box),
                                    "fresh_item_error_m_evaluator_only": None if
                                        fresh_point is None else float(
                                            np.linalg.norm(fresh_point -
                                                model.body_pos[model.body(
                                                    "shown_can").id])),
                                    "image_path": str(fresh_image_path),
                                    "image_sha256": digest(fresh_image_path),
                                    "depth_path": str(fresh_depth_path),
                                    "depth_sha256": digest(fresh_depth_path)})
                                recheck["fresh_views"].append({
                                    "index": 2,
                                    "time_s": round(time_s, 4),
                                    "track_status": fresh_status,
                                    "image_path": str(fresh_image_path),
                                    "image_sha256": digest(fresh_image_path),
                                    "depth_path": str(fresh_depth_path),
                                    "depth_sha256": digest(fresh_depth_path)})
                                recheck["actor_claim_at_recheck"] = bool(
                                    not recheck_alarm and
                                    last_pose_health == "registered" and
                                    fresh_status ==
                                        "visually_reacquired_unlocalized" and
                                    fresh_box is not None and
                                    owner.iou(box, fresh_box) >= .6 and
                                    fresh_point is not None and
                                    np.linalg.norm(fresh_point[:2] -
                                                   item_memory[:2]) <=
                                        MEMORY_ASSOCIATION_M)
                                frames.append(owner.owner_frame(
                                    fresh_rgb, "held station | confirmed recheck",
                                    fresh_box, fresh_gt_box))
                        held_recheck_result = recheck
                        trace.append(recheck)

    item_status = "not_rechecked"
    chosen_box = None
    fresh_box = None
    fresh_status = "not_captured"
    fresh_age_s = None
    estimated_item = None
    evaluator_item_error = None
    if held_recheck_result is not None:
        item_status = ("fresh_metric_item_refound_from_held_station" if
                       held_recheck_result["actor_claim_at_recheck"] and
                       not held_recheck_invalidated else
                       "held_view_unconfirmed_ask_owner")
        chosen_box = (None if held_view_probe is None else
                      held_view_probe["selected_box"])
        fresh_box = held_recheck_result.get("fresh_box")
        fresh_status = held_recheck_result.get("fresh_track_status", "not_captured")
        fresh_age_s = held_recheck_result.get("fresh_frame_age_s")
        estimated_item = held_recheck_result.get("fresh_depth_item_xyz_m")
        evaluator_item_error = held_recheck_result.get(
            "fresh_item_error_m_evaluator_only")
    if route_state == "running":
        interaction.aim_camera(model, interaction.CAMERA, estimate[:2],
                               item_memory[:2], 20)
        place_cameras()
        captures = []
        rgb, depth, capture_wall = owner.capture(
            renderer, data, interaction.CAMERA, True)
        image = Image.fromarray(rgb)
        started = time.perf_counter()
        proposals = vision.detect(image)
        scores = (vision.feature([image.crop(item["box"]) for item in proposals])
                  @ exemplar if proposals else np.empty(0))
        ranked = sorted(zip(proposals, scores),
                        key=lambda pair: float(pair[1]), reverse=True)
        recognition_wall = time.perf_counter() - started
        time_s += max(.1, capture_wall + recognition_wall)
        choice = ranked[0] if ranked and float(ranked[0][1]) >= owner.THRESHOLD else None
        chosen_box = None if choice is None else choice[0]["box"]
        tracker = (None if chosen_box is None else owner.FastActor(
            cv2.cvtColor(np.asarray(image.crop(chosen_box)), cv2.COLOR_RGB2BGR),
            use_depth=False))
        for index in range(4):
            if index > 0:
                rgb, depth, capture_wall = owner.capture(
                    renderer, data, interaction.CAMERA, True)
                if tracker is None:
                    fresh_status = "no_initial_candidate"
                    fresh_age_s = capture_wall
                    fresh_box = None
                else:
                    started = time.perf_counter()
                    tracked = tracker.observe(
                        Image.fromarray(rgb), None, float(estimate[0]))
                    fresh_age_s = capture_wall + time.perf_counter() - started
                    fresh_status = tracked["status"]
                    if fresh_age_s > owner.MAX_FRESH_AGE_S:
                        fresh_status = "stale_packet"
                        tracker.reset_continuity()
                    fresh_box = tracked["box"] if fresh_status != "stale_packet" else None
                time_s += max(.1, fresh_age_s)
            image_path = branch / f"return-view-{index:02d}.png"
            depth_path = branch / f"return-depth-{index:02d}.npz"
            Image.fromarray(rgb).save(image_path)
            np.savez_compressed(depth_path, depth_m=depth)
            box = chosen_box if index == 0 else fresh_box
            if index > 0 and box is not None:
                point = interaction.depth_point(box, depth, data, model,
                                                interaction.CAMERA)
                if point is not None:
                    point = np.asarray(point, dtype=float)
                    point[:2] += estimate[:2] - actual
                    estimated_item = point.tolist()
                    evaluator_item_error = float(np.linalg.norm(
                        point - model.body_pos[model.body("shown_can").id]))
            gt_box, _ = interaction.evaluator.evaluator_box(
                renderer, data, model, interaction.CAMERA, "shown_can_geom")
            frames.append(owner.owner_frame(
                rgb, f"memory return | {source} | fresh {index}", box, gt_box))
            captures.append({"stage": "memory_fresh_view", "index": index,
                             "time_s": round(time_s, 4), "image_path": str(image_path),
                             "image_sha256": digest(image_path),
                             "depth_path": str(depth_path),
                             "depth_sha256": digest(depth_path),
                             "selected_box": box,
                             "top_similarity": (None if index or not ranked
                                                else float(ranked[0][1])),
                             "candidate_count": len(proposals) if index == 0 else None,
                             "recognition_wall_s": recognition_wall if index == 0 else None,
                             "fresh_track_status": fresh_status if index else None,
                             "fresh_frame_age_s": fresh_age_s if index else None,
                             "item_box_iou_evaluator_only": owner.iou(box, gt_box),
                             "actor_estimated_xy_m": estimate[:2].round(5).tolist(),
                             "evaluator_actual_xy_m": actual.round(5).tolist()})
            if index and (tracker is None or
                          fresh_status == "visually_reacquired_unlocalized"):
                break
        trace.extend(captures)
        same_view = (chosen_box is not None and fresh_box is not None and
                     owner.iou(chosen_box, fresh_box) >= .6 and
                     fresh_status == "visually_reacquired_unlocalized")
        metric_health = source != "command" and last_pose_health == "registered"
        item_consistent = (estimated_item is not None and
                           float(np.linalg.norm(np.asarray(estimated_item)[:2] -
                                                item_memory[:2])) <= MEMORY_ASSOCIATION_M)
        if same_view and item_consistent and metric_health:
            item_status = "fresh_metric_item_refound"
        elif same_view:
            item_status = "fresh_visual_candidate_pose_or_memory_unverified"
        else:
            item_status = "item_not_refound_ask_owner"
    result = {"source": source, "item_state": item_state,
              "run_label": run_label,
              "predeparture_watch_s_requested": predeparture_watch_s,
              "predeparture_watch_trigger": watch_trigger,
              "predeparture_escape_requested": predeparture_escape,
              "predeparture_future_hold_probe_requested":
                  predeparture_future_hold_probe,
              "predeparture_future_hold_select_requested":
                  predeparture_future_hold_select,
              "predeparture_future_hold_revalidate_requested":
                  predeparture_future_hold_revalidate,
              "predeparture_future_hold_replan_probe_requested":
                  predeparture_future_hold_replan_probe,
              "predeparture_confirmed_track_hold_select_requested":
                  predeparture_confirmed_track_hold_select,
              "predeparture_track_continuity_gate_requested":
                  predeparture_track_continuity_gate,
              "future_probe_requested": future_probe,
              "hold_monitor_requested": hold_monitor,
              "hold_scan_interval_s_requested": hold_scan_interval_s,
              "held_reactive_escape_requested": held_reactive_escape,
              "held_reactive_escape_count": reactive_escape_count,
              "held_reactive_escape_time_s": reactive_escape_time_s,
              "held_view_probe_at_s_requested": held_view_probe_at_s,
              "held_view_min_raw_range_m_requested": held_view_min_raw_range_m,
              "held_view_probe": held_view_probe,
              "held_view_recheck_requested": held_view_recheck,
              "held_view_response_probe_requested": held_view_response_probe,
              "held_view_emergency_requested": held_view_emergency,
              "held_view_directional_emergency_requested":
                  held_view_directional_emergency,
              "held_view_worker_range_dropout_s_requested":
                  held_view_worker_range_dropout_s,
              "held_view_recheck": held_recheck_result,
              "held_view_recheck_invalidated": held_recheck_invalidated,
              "hold_scan_dropout_window_s_requested": (
                  None if hold_scan_dropout_window_s is None else
                  list(hold_scan_dropout_window_s)),
              "hold_missing_scan_count": hold_missing_scan_count,
              "head_depth_fallback_requested": head_depth_fallback,
              "head_depth_capture_count": head_depth_capture_count,
              "paired_scan_seed_requested": paired_scan_seed,
              "depth_candidate_probe_requested": depth_candidate_probe,
              "head_depth_processing_latency_s_requested":
                  head_depth_processing_latency_s,
              "head_upper_processing_latency_s_requested":
                  head_upper_processing_latency_s,
              "future_hold_screen_requested": future_hold_screen,
              "best_effort_if_unsupported_requested":
                  best_effort_if_unsupported,
              "map_snapshot_at_hazard_requested": map_snapshot_at_hazard,
              "dynamic_trail_reopen_requested": dynamic_trail_reopen,
              "dynamic_trail_reopen_during_reactive_requested":
                  dynamic_trail_reopen_during_reactive,
              "held_warning_confirmations_requested":
                  held_warning_confirmations,
              "emergency_candidate_strategy_requested":
                  emergency_candidate_strategy,
              "evaluator_abort_on_margin_breach_requested":
                  evaluator_abort_on_margin_breach,
              "dynamic_trail_reclaimed_cells":
                  dynamic_trail_reclaimed_cells,
              "held_reactive_escape_goal_xy_m": (None if reactive_escape_goal is None
                                                  else reactive_escape_goal.round(5).tolist()),
              "escape_distances_m_requested": list(escape_distances_m),
              "escape_tolerance_m_requested": escape_tolerance_m,
              "predeparture_escape_goal_xy_m": (None if escape_target is None else
                                                  escape_target.round(5).tolist()),
              "fault_profile": fault_profile,
              "fault_after_move": fault_after_move if fault_applied else None,
              "pose_source_at_return": "range_recovery" if range_recovery else source,
              "route_status": route_state, "item_status": item_status,
              "memory_capture_time_s": memory_time_s,
              "memory_item_xyz_m": item_memory.round(5).tolist(),
              "away_view_xy_m": away_view.round(5).tolist(),
              "return_view_xy_m": return_view.round(5).tolist(),
              "moves": motion_count,
              "end_actual_xy_m_evaluator_only": actual.round(5).tolist(),
              "end_estimated_xy_m": estimate[:2].round(5).tolist(),
              "end_pose_error_m_evaluator_only": round(
                  float(np.linalg.norm(actual - estimate[:2])), 5),
              "minimum_proxy_gap_m_evaluator_only": round(min_proxy_gap, 5),
              "post_stop_minimum_proxy_gap_m_evaluator_only": round(post_stop_min_gap, 5),
              "post_stop_margin_breach_evaluator_only": post_stop_breach,
              "last_pose_health": last_pose_health,
              "two_view_iou": None if chosen_box is None or fresh_box is None
              else owner.iou(chosen_box, fresh_box),
              "fresh_track_status": fresh_status,
              "fresh_frame_age_s": fresh_age_s,
              "fresh_item_xyz_m": estimated_item,
              "fresh_item_error_m_evaluator_only": evaluator_item_error}
    trace.append({"stage": "memory_return_decision", "time_s": round(time_s, 4),
                  **result})
    return {"end_time_s": time_s, "frames": frames,
            "trace": trace, "summary": result}


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    policy = json.loads(joined.CONFIG.read_text())
    case = next(row for row in policy["cases"] if row["id"] == "stationary")
    declaration = OUT / "predeclared-memory-return.json"
    declaration.write_text(json.dumps({
        "schema": "robot-learning-lab.robocasa-memory-return-declaration.v1",
        "scene": case, "sources": SOURCES, "item_states": ITEM_STATES,
        "away_view_rule": "estimated route capture at step 3",
        "return_view_rule": "estimated stop pose at owner item confirmation",
        "pose_fault": "alternating hidden .58/.94 action transfer during memory route",
        "body": "17 cm kinematic proxy; historical Direct Frame remains rejected",
        "source_sha256": digest(SOURCE),
        "joined_source_sha256": digest(Path(joined.__file__)),
        "config_sha256": digest(joined.CONFIG),
        "prior_map_sha256": digest(joined.PRIOR),
    }, indent=2) + "\n")
    prior = np.load(joined.PRIOR)
    vision = owner.Vision(teach_selector="generic-proposal")
    exemplar = vision.feature([Image.open(owner.TEACH_CROP).convert("RGB")])[0]
    old_out, old_hook, old_margin = joined.OUT, joined.POST_OWNER_HOOK, route.MARGIN_M
    joined.OUT = OUT
    route.MARGIN_M = .15
    episodes = []
    try:
        for item_state in ITEM_STATES:
            for source in SOURCES:
                joined.POST_OWNER_HOOK = (
                    lambda *, source=source, item_state=item_state, **kwargs:
                    callback(source=source, item_state=item_state, **kwargs))
                response = f"memory_{source}_{item_state}"
                episode = joined.episode(case, response, True, prior, policy,
                                         vision, exemplar)
                episodes.append(episode)
                (OUT / "partial-report.json").write_text(json.dumps({
                    "declaration_sha256": digest(declaration),
                    "episodes": episodes}, indent=2) + "\n")
                memory = episode["memory_return"]
                print(source, item_state, episode["route_state"],
                      episode["outcome"],
                      None if memory is None else memory["route_status"],
                      None if memory is None else memory["item_status"], flush=True)
    finally:
        joined.OUT, joined.POST_OWNER_HOOK = old_out, old_hook
        route.MARGIN_M = old_margin
    (OUT / "report.json").write_text(json.dumps({
        "schema": "robot-learning-lab.robocasa-memory-return-result.v1",
        "declaration_sha256": digest(declaration),
        "source_sha256": digest(SOURCE),
        "joined_source_sha256": digest(Path(joined.__file__)),
        "episodes": episodes,
    }, indent=2) + "\n")
    (OUT / SOURCE.name).write_bytes(SOURCE.read_bytes())
    (OUT / Path(joined.__file__).name).write_bytes(Path(joined.__file__).read_bytes())


if __name__ == "__main__":
    main()
