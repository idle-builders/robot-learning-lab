"""Recover a lost owner range track from released person RGB-D evidence.

The actor asks for a show at the stationary range stop, compares calibrated
depth before/after the show within the range track's neighborhood, and picks
a previously known-free farther pose that bounds the changed surface under a
development owner-motion assumption. It then monitors motion during slow
recognition and requires a fresh visual/depth confirmation. All physical
translation and owner motion are scripted surrogates, not human safety.
"""

from __future__ import annotations

import math
import time
import hashlib
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor

import cv2
import mujoco
import numpy as np
from PIL import Image

import robocasa_owner_person as person
import robocasa_person_interaction as base
import robocasa_person_monitored_turn_v502 as monitored
import robocasa_standoff_turn_v507 as range_standoff
import run_robocasa_joined_owner_route_v410 as owner
import run_robocasa_layout2_multitrack_v485 as multi
from analyze_robocasa_owner_monitor_v501 import changed_beams
from run_robocasa_scan_height_v433 import height_scan
from screen_robocasa_owner_person_v489 import evaluator_box
from screen_robocasa_owner_safe_stop_v494 import aim_camera, depth_point
import owner_depth_change_component_v516 as depth_change
import robocasa_owner_reacquire_rgbd_v560 as person_reacquire
import run_owner_concurrent_reacquire_v571 as async_person


changed_points = depth_change.changed_points


SCAN = height_scan((.05, .30))
MIN_RESERVE_M = .10
OWNER_MOTION_BOUND_M = .60
BODY_RADIUS_M = .17
REQUIRED_GAP_M = .05
POSE_UNCERTAINTY_M = .03
DEPTH_UNCERTAINTY_M = .03
BACKTRACK_SPEED_M_S = .25
OWNER_APPROACH_M = .4
OWNER_APPROACH_S = 2.
SECOND_RETURN_OWNER_M = .6
SECOND_RETURN_OWNER_S = 1.5
FIRST_RETURN_OWNER_M = .3
FIRST_RETURN_OWNER_S = 1.
PRE_SHOW_RESERVE_M = .35
RETURN_TRACK_RADIUS_M = .35
RETURN_MOTION_SHIFT_M = .10
DT = .02
PROP_SCRIPT = None
PACKET_OUT: Path | None = None
DEPTH_NOISE_SIGMA_M = 0.
DEPTH_DROPOUT_FRACTION = 0.
LATE_OWNER_APPROACH_M = 0.
LATE_OWNER_APPROACH_S = 2.
REACQUIRE_RETREAT_M = .5
REACQUIRE_RESPONSE_DELAY_S = .2
REACQUIRE_OWNER_SECTOR_DEG = 35.


def _prop(model, t):
    if PROP_SCRIPT is not None:
        model.body_pos[model.body("route_obstacle").id, :2] = PROP_SCRIPT(t)


def _choose_standoff(grid, changed_xyz_est, estimate):
    return depth_change.choose_standoff(
        grid, changed_xyz_est, estimate,
        motion_bound_m=OWNER_MOTION_BOUND_M,
        body_radius_m=BODY_RADIUS_M,
        required_gap_m=REQUIRED_GAP_M,
        pose_uncertainty_m=POSE_UNCERTAINTY_M,
        depth_uncertainty_m=DEPTH_UNCERTAINTY_M,
        development_reserve_m=MIN_RESERVE_M)


def _place_owner(model, south_m):
    return monitored._present_owner(model, south_m)


def _rest_owner(model, south_m):
    torso = base.TORSO_XY + [0., -south_m]
    hand = base.REST_HAND + [0., -south_m, 0.]
    pose = person.set_owner_pose(model, torso, hand, show_can=False)
    pose["show_can"] = False
    return pose


def _show_owner(model, fraction, south_m):
    torso = base.TORSO_XY + [0., -south_m]
    hand = base.REST_HAND + fraction * (base.SHOW_HAND - base.REST_HAND)
    hand = hand + [0., -south_m, 0.]
    pose = person.set_owner_pose(model, torso, hand, show_can=True)
    direction = hand[:2] - torso
    direction /= np.linalg.norm(direction)
    model.body_pos[model.body("shown_can").id, :2] = hand[:2] + .14 * direction
    pose["can_xyz_m"] = model.body_pos[model.body("shown_can").id].tolist()
    pose["show_can"] = True
    return pose


class ReturnOwnerTrack:
    def __init__(self, center_xy):
        self.center = np.asarray(center_xy, dtype=float)
        self.first_y = None
        self.streak = 0
        self.first_alarm_s = None
        self.max_south_shift_m = 0.

    def observe(self, packet, estimated_xy, time_s):
        angles = np.linspace(-math.pi, math.pi, len(packet), endpoint=False)
        points = np.asarray([
            np.asarray(estimated_xy) + value * np.array(
                [math.cos(angles[beam]), math.sin(angles[beam])])
            for beam, value in enumerate(packet) if value is not None]).reshape(-1, 2)
        nearby = points[np.linalg.norm(points - self.center, axis=1) <
                        RETURN_TRACK_RADIUS_M]
        alarm_now = False
        shift = None
        if len(nearby) >= 2:
            self.center = np.median(nearby, axis=0)
            if self.first_y is None:
                self.first_y = float(self.center[1])
            shift = float(self.first_y - self.center[1])
            self.max_south_shift_m = max(self.max_south_shift_m, shift)
            self.streak = self.streak + 1 if shift >= RETURN_MOTION_SHIFT_M else 0
            if self.streak >= 2 and self.first_alarm_s is None:
                self.first_alarm_s = time_s
                alarm_now = True
        else:
            self.streak = 0
        return {"estimated_owner_surface_xy_m": self.center.tolist()
                if len(nearby) >= 2 else None,
                "supporting_range_returns": int(len(nearby)),
                "south_shift_from_first_packet_m": shift,
                "motion_streak": self.streak, "alarm_now": alarm_now}


def _choose_pre_show_standoff(grid, track, estimate):
    envelope = (track["radius_m"] + BODY_RADIUS_M + REQUIRED_GAP_M +
                POSE_UNCERTAINTY_M + OWNER_MOTION_BOUND_M)
    candidates = []
    for distance in (0., .1, .2, .3, .4, .5, .6):
        target = np.asarray(estimate[:2]) + [0., -distance]
        reserve = float(np.linalg.norm(target - track["center_xy_m"]) - envelope)
        cells = []
        for amount in np.linspace(0., distance, 7):
            xy = np.asarray(estimate[:2]) + [0., -amount]
            cell = grid.cell(xy)
            cells.append({"xy_m": xy.tolist(), "free": bool(grid.free[cell]),
                          "occupied": bool(grid.occupied[cell])})
        known = all(row["free"] and not row["occupied"] for row in cells)
        row = {"distance_m": distance, "target_xy_est_m": target.tolist(),
               "assumed_motion_envelope_m": envelope,
               "reserve_m": reserve, "known_free_path": known,
               "required_pre_show_reserve_m": PRE_SHOW_RESERVE_M,
               "cells": cells}
        candidates.append(row)
        if reserve >= PRE_SHOW_RESERVE_M and known:
            return row, candidates
    return None, candidates


def _known_free_south(grid, estimate, distance):
    cells = []
    for amount in np.linspace(0., distance, 11):
        xy = np.asarray(estimate[:2]) + [0., -amount]
        cell = grid.cell(xy)
        cells.append({"xy_m": xy.tolist(), "free": bool(grid.free[cell]),
                      "occupied": bool(grid.occupied[cell])})
    return all(row["free"] and not row["occupied"] for row in cells), cells


def _sector_change(a, b, bearing, half_width_degrees):
    angles = np.linspace(-math.pi, math.pi, len(a), endpoint=False)
    delta = (angles - bearing + math.pi) % (2 * math.pi) - math.pi
    return int(sum(abs(delta[index]) <= math.radians(half_width_degrees) and
                   ((old is None) != (new is None) or
                    (old is not None and new is not None and abs(old-new) >= .05))
                   for index, (old, new) in enumerate(zip(a, b))))


def owner_turn(response, model, data, renderer, actual, estimate,
               time_s, use_depth, vision, exemplar):
    moving_owner = not response.startswith("stationary")
    grid = multi.CAPTURED[-1]
    track = dict(max(grid.multi_history[-1]["tracks"], key=lambda row: row["hits"]))
    pre_target, pre_candidates = _choose_pre_show_standoff(grid, track, estimate)
    rows, frames = [{"stage": "choose_pre_show_standoff",
                     "time_s": round(time_s, 4),
                     "candidates": pre_candidates, "chosen": pre_target}], []
    if pre_target is None:
        return "pre_show_standoff_not_supported", rows, frames, None, None, time_s
    start_actual = actual.copy()
    start_estimate = estimate[:2].copy()
    pre_distance = pre_target["distance_m"]
    pre_steps = int(round(pre_distance / BACKTRACK_SPEED_M_S / DT))
    pre_start_s = time_s
    pre_track = ReturnOwnerTrack(track["center_xy_m"])
    pre_support = []

    def first_owner_south_at(t):
        return (FIRST_RETURN_OWNER_M * min(1., max(0.,
                (t - pre_start_s) / FIRST_RETURN_OWNER_S))) if moving_owner else 0.

    for index in range(1, pre_steps + 1):
        t = time_s + index * DT
        amount = min(pre_distance, BACKTRACK_SPEED_M_S * index * DT)
        rest_pose = _rest_owner(model, first_owner_south_at(t))
        actual[:] = start_actual + [0., -amount]
        estimate[:2] = start_estimate + [0., -amount]
        model.body_pos[model.body("audit_ape").id, :2] = actual
        aim_camera(model, base.CAMERA, estimate[:2], track["center_xy_m"], 20)
        model.cam_pos[model.camera(base.CAMERA).id, :2] = actual
        _prop(model, t)
        mujoco.mj_forward(model, data)
        row = {"stage": "pre_show_backtrack", "time_s": round(t, 4),
               "ape_xy_m": actual.tolist(),
               "estimated_xy_m": estimate[:2].tolist(),
               "owner_pose_evaluator": rest_pose,
               "skill_status": "kinematic_surrogate_unqualified"}
        if index % 5 == 0:
            packet = SCAN(model, data, actual, 5., np.random.default_rng(6500 + index))
            row["range_returns"] = sum(x is not None for x in packet)
            observation = pre_track.observe(packet, estimate[:2], t)
            pre_support.append(observation["supporting_range_returns"])
            row["owner_pre_show_motion_packet"] = observation
            if observation["alarm_now"]:
                row["actor_action"] = "defer_show_restage_from_owner_range_motion"
            rgb, _, _ = owner.capture(renderer, data, base.CAMERA, False)
            frames.append(owner.owner_frame(rgb, "pre-show owner standoff", None, None))
        rows.append(row)
    time_s += pre_steps * DT
    first_south = first_owner_south_at(time_s)
    blind_run = longest_blind_run = 0
    for count in pre_support:
        blind_run = blind_run + 1 if count < 2 else 0
        longest_blind_run = max(longest_blind_run, blind_run)
    early_anchor = any(count >= 2 for count in pre_support[:2])
    owner_track_healthy = early_anchor and longest_blind_run < 4
    rows.append({"stage": "pre_show_motion_assessment", "time_s": round(time_s, 4),
                 "owner_motion_seen_actor": pre_track.first_alarm_s is not None,
                 "first_motion_alarm_time_s_actor": pre_track.first_alarm_s,
                 "maximum_observed_south_shift_m": pre_track.max_south_shift_m,
                 "owner_track_early_anchor": early_anchor,
                 "longest_owner_support_gap_packets": longest_blind_run,
                 "owner_track_healthy": owner_track_healthy})
    recovered_motion = False
    if not owner_track_healthy:
        rows.append({"stage": "owner_range_track_lost",
                     "time_s": round(time_s, 4),
                     "actor_action": "defer_show_seek_person_rgbd"})
        observations = []
        captures = []
        guard_start_s = time_s
        retreat_known, retreat_cells = _known_free_south(
            grid, estimate, REACQUIRE_RETREAT_M)
        rows.append({"stage": "owner_reacquire_retreat_path",
                     "time_s": round(time_s, 4),
                     "distance_m": REACQUIRE_RETREAT_M,
                     "known_free_path": retreat_known,
                     "cells": retreat_cells})
        if LATE_OWNER_APPROACH_M and not retreat_known:
            return "owner_reacquire_no_known_free_retreat", rows, frames, None, None, time_s
        for index in range(2):
            pitch = (20, 15)[index]
            aim_camera(model, base.CAMERA, estimate[:2], track["center_xy_m"], pitch)
            model.cam_pos[model.camera(base.CAMERA).id, :2] = actual
            late_now = LATE_OWNER_APPROACH_M * min(
                1., max(0., (time_s - guard_start_s) / LATE_OWNER_APPROACH_S))
            _rest_owner(model, first_owner_south_at(time_s) + late_now)
            _prop(model, time_s)
            mujoco.mj_forward(model, data)
            rgb, depth, age = owner.capture(renderer, data, base.CAMERA, True)
            capture_s = time_s + max(.1, age)
            start_actual = actual.copy()
            start_estimate = estimate[:2].copy()
            direction = np.asarray(track["center_xy_m"]) - start_estimate
            bearing = math.atan2(direction[1], direction[0])
            baseline = None
            streak = 0
            alarm_elapsed = None
            scan_wall_times = []
            with ThreadPoolExecutor(max_workers=1) as pool:
                wall_start = time.perf_counter()
                future = pool.submit(async_person.detect_person, rgb)
                next_tick = wall_start
                next_scan_elapsed = 0.
                while True:
                    now = time.perf_counter()
                    if now < next_tick:
                        time.sleep(next_tick - now)
                        now = time.perf_counter()
                    elapsed = now - wall_start
                    t = capture_s + elapsed
                    late_south = LATE_OWNER_APPROACH_M * min(
                        1., max(0., (t - guard_start_s) / LATE_OWNER_APPROACH_S))
                    south = first_owner_south_at(t) + late_south
                    retreat = (min(REACQUIRE_RETREAT_M,
                                   BACKTRACK_SPEED_M_S * max(
                                       0., elapsed - alarm_elapsed -
                                       REACQUIRE_RESPONSE_DELAY_S))
                               if alarm_elapsed is not None else 0.)
                    actual[:] = start_actual + [0., -retreat]
                    estimate[:2] = start_estimate + [0., -retreat]
                    model.body_pos[model.body("audit_ape").id, :2] = actual
                    aim_camera(model, base.CAMERA, estimate[:2],
                               track["center_xy_m"], pitch)
                    model.cam_pos[model.camera(base.CAMERA).id, :2] = actual
                    rest_pose = _rest_owner(model, south)
                    _prop(model, t)
                    mujoco.mj_forward(model, data)
                    row = {"stage": "owner_reacquire_guard_tick",
                           "time_s": round(t, 4), "index": index,
                           "wall_elapsed_s": elapsed,
                           "ape_xy_m": actual.tolist(),
                           "estimated_xy_m": estimate[:2].tolist(),
                           "owner_pose_evaluator": rest_pose,
                           "skill_status": "kinematic_surrogate_unqualified"}
                    if elapsed >= next_scan_elapsed:
                        packet = SCAN(model, data, actual, 5.,
                                      np.random.default_rng(
                                          14000 + len(scan_wall_times)))
                        if baseline is None:
                            baseline = packet
                        changed = _sector_change(
                            baseline, packet, bearing,
                            REACQUIRE_OWNER_SECTOR_DEG)
                        streak = streak + 1 if changed >= 3 else 0
                        if streak >= 2 and alarm_elapsed is None:
                            alarm_elapsed = elapsed
                            row["actor_action"] = "invalidate_person_image_and_retract"
                        row["owner_sector_changed_beams"] = changed
                        row["owner_sector_motion_streak"] = streak
                        row["range_returns"] = sum(v is not None for v in packet)
                        scan_wall_times.append(elapsed)
                        next_scan_elapsed += .1
                        while next_scan_elapsed <= elapsed:
                            next_scan_elapsed += .1
                        if len(scan_wall_times) % 4 == 1:
                            guard_rgb, _, _ = owner.capture(
                                renderer, data, base.CAMERA, False)
                            frames.append(owner.owner_frame(
                                guard_rgb, "watching owner during person inference",
                                None, None))
                    rows.append(row)
                    if future.done() and (alarm_elapsed is None or
                                          retreat >= REACQUIRE_RETREAT_M - 1e-8):
                        break
                    if elapsed > 30.:
                        break
                    next_tick += DT
                    if next_tick < now:
                        next_tick = now + DT
                detection = future.result()
            time_s = capture_s + elapsed
            if alarm_elapsed is not None:
                rows.append({"stage": "owner_reacquire_image_cancelled",
                             "time_s": round(time_s, 4),
                             "alarm_time_s": round(capture_s + alarm_elapsed, 4),
                             "retreat_distance_m": float(np.linalg.norm(
                                 actual - start_actual)),
                             "scan_count": len(scan_wall_times),
                             "max_scan_interval_s": float(np.max(np.diff(
                                 scan_wall_times))) if len(scan_wall_times) > 1 else None,
                             "stale_detection_count_discarded": len(
                                 detection["detections"]),
                             "detector_wall_s": detection["worker_wall_s"]})
                aim_camera(model, base.CAMERA, estimate[:2],
                           track["center_xy_m"], 20)
                model.cam_pos[model.camera(base.CAMERA).id, :2] = actual
                mujoco.mj_forward(model, data)
                fresh_rgb, fresh_depth, fresh_age = owner.capture(
                    renderer, data, base.CAMERA, True)
                time_s += max(.1, fresh_age)
                healthy, fractions = (depth_change.packet_health(
                    fresh_depth, fresh_depth) if fresh_depth is not None else
                    (False, [0., 0.]))
                packet = None
                if PACKET_OUT is not None:
                    path = PACKET_OUT.with_name(
                        PACKET_OUT.stem + "-person-interrupted.npz")
                    path.parent.mkdir(parents=True, exist_ok=True)
                    np.savez_compressed(path, stale_rgb=rgb, stale_depth=depth,
                                        fresh_rgb=fresh_rgb, fresh_depth=fresh_depth)
                    packet = {"path": str(path),
                              "sha256": hashlib.sha256(path.read_bytes()).hexdigest()}
                frames.append(owner.owner_frame(
                    fresh_rgb, "fresh view after owner motion: task reset",
                    None, None))
                rows.append({"stage": "owner_post_retreat_view",
                             "time_s": round(time_s, 4),
                             "ape_xy_m": actual.tolist(),
                             "owner_pose_evaluator": rest_pose,
                             "rgb_sha256": hashlib.sha256(
                                 fresh_rgb.tobytes()).hexdigest(),
                             "central_valid_fraction": fractions[0],
                             "depth_healthy": healthy,
                             "raw_packets": packet,
                             "task_state": "owner_association_reset_no_show"})
                return ("owner_interaction_reset_after_reacquire_motion",
                        rows, frames, None, None, time_s)
            healthy, fractions = (depth_change.packet_health(depth, depth)
                                  if depth is not None else (False, [0., 0.]))
            detections = detection["detections"]
            if not healthy:
                observation = {"status": "depth_unhealthy",
                               "central_valid_fraction": fractions[0],
                               "detections": detections}
            elif not detections or detections[0]["score"] < .25:
                observation = {"status": "person_unconfirmed",
                               "central_valid_fraction": fractions[0],
                               "detections": detections}
            else:
                point = depth_point(detections[0]["box_xyxy"], depth,
                                    data, model, base.CAMERA)
                if point is None:
                    observation = {"status": "person_depth_unlocalized",
                                   "central_valid_fraction": fractions[0],
                                   "detections": detections}
                else:
                    point = np.asarray(point, dtype=float)
                    point[:2] += estimate[:2] - actual
                    observation = {"status": "person_surface_localized",
                                   "central_valid_fraction": fractions[0],
                                   "detections": detections,
                                   "point_xyz_est_m": point.tolist()}
            observation["detector_wall_s"] = detection["worker_wall_s"]
            observations.append(observation)
            captures.append((rgb, depth))
            box = (observation["detections"][0]["box_xyxy"]
                   if observation.get("detections") else None)
            frames.append(owner.owner_frame(rgb,
                          f"reacquire owner: {observation['status']}", box, None))
            rows.append({"stage": "owner_person_rgbd_reacquire",
                         "time_s": round(time_s, 4), "index": index,
                         "pitch_degrees": pitch,
                         "actor_observation": observation})
        if PACKET_OUT is not None:
            path = PACKET_OUT.with_name(PACKET_OUT.stem + "-person-reacquire.npz")
            path.parent.mkdir(parents=True, exist_ok=True)
            np.savez_compressed(path, rgb_0=captures[0][0],
                                depth_0=captures[0][1],
                                rgb_1=captures[1][0], depth_1=captures[1][1])
            rows.append({"stage": "owner_person_rgbd_packets",
                         "time_s": round(time_s, 4), "path": str(path),
                         "sha256": hashlib.sha256(path.read_bytes()).hexdigest()})
        points = [np.asarray(item["point_xyz_est_m"])
                  for item in observations
                  if item["status"] == "person_surface_localized"]
        boxes = [item["detections"][0]["box_xyxy"]
                 for item in observations if item.get("detections")]
        consistent = (len(points) == 2 and len(boxes) == 2 and
                      np.linalg.norm(points[0] - points[1]) <= .15)
        if not consistent:
            rows.append({"stage": "defer_show_for_owner_track_loss",
                         "time_s": round(time_s, 4),
                         "reason": "person_rgbd_unconfirmed"})
            return "owner_track_uncertain_before_show", rows, frames, None, None, time_s
        candidate = np.mean(points, axis=0)[:2]
        displacement = float(np.linalg.norm(candidate - track["center_xy_m"]))
        if displacement > OWNER_MOTION_BOUND_M + .15:
            rows.append({"stage": "defer_show_for_owner_track_loss",
                         "time_s": round(time_s, 4),
                         "reason": "person_beyond_association_bound",
                         "candidate_distance_from_prior_m": displacement})
            return "owner_track_uncertain_before_show", rows, frames, None, None, time_s
        surface_uncertainty = (track["radius_m"] + POSE_UNCERTAINTY_M +
                               DEPTH_UNCERTAINTY_M)
        recovered_motion = displacement > surface_uncertainty
        min_person_score = min(item["detections"][0]["score"]
                               for item in observations)
        if recovered_motion and min_person_score < .35:
            rows.append({"stage": "defer_show_for_owner_track_loss",
                         "time_s": round(time_s, 4),
                         "reason": "relocated_person_below_strong_score",
                         "candidate_distance_from_prior_m": displacement,
                         "minimum_person_score": min_person_score})
            return "owner_track_uncertain_before_show", rows, frames, None, None, time_s
        rows.append({"stage": "owner_person_rgbd_association",
                     "time_s": round(time_s, 4),
                     "new_owner_surface_xy_est_m": candidate.tolist(),
                     "distance_from_prior_track_m": displacement,
                     "surface_uncertainty_m": surface_uncertainty,
                     "restage_required": recovered_motion,
                     "minimum_person_score": min_person_score,
                     "score_gate": .35 if recovered_motion else .25})
        if recovered_motion:
            track["center_xy_m"] = candidate.tolist()
            pre_track.center = candidate.copy()
    if pre_track.first_alarm_s is not None or recovered_motion:
        if not recovered_motion:
            track["center_xy_m"] = pre_track.center.tolist()
        extra_target, extra_candidates = _choose_pre_show_standoff(grid, track, estimate)
        rows.append({"stage": "choose_pre_show_restage", "time_s": round(time_s, 4),
                     "updated_range_track_actor": track,
                     "candidates": extra_candidates, "chosen": extra_target})
        if extra_target is None:
            return "pre_show_restage_not_supported", rows, frames, None, None, time_s
        extra_distance = extra_target["distance_m"]
        extra_steps = int(round(extra_distance / BACKTRACK_SPEED_M_S / DT))
        extra_actual = actual.copy()
        extra_estimate = estimate[:2].copy()
        for index in range(1, extra_steps + 1):
            t = time_s + index * DT
            amount = min(extra_distance, BACKTRACK_SPEED_M_S * index * DT)
            rest_pose = _rest_owner(model, first_owner_south_at(t))
            actual[:] = extra_actual + [0., -amount]
            estimate[:2] = extra_estimate + [0., -amount]
            model.body_pos[model.body("audit_ape").id, :2] = actual
            aim_camera(model, base.CAMERA, estimate[:2], track["center_xy_m"], 20)
            model.cam_pos[model.camera(base.CAMERA).id, :2] = actual
            _prop(model, t)
            mujoco.mj_forward(model, data)
            row = {"stage": "pre_show_motion_restaging", "time_s": round(t, 4),
                   "ape_xy_m": actual.tolist(),
                   "estimated_xy_m": estimate[:2].tolist(),
                   "owner_pose_evaluator": rest_pose,
                   "skill_status": "kinematic_surrogate_unqualified"}
            if index % 5 == 0:
                packet = SCAN(model, data, actual, 5.,
                              np.random.default_rng(6600 + index))
                row["owner_pre_show_motion_packet"] = pre_track.observe(
                    packet, estimate[:2], t)
                rgb, _, _ = owner.capture(renderer, data, base.CAMERA, False)
                frames.append(owner.owner_frame(rgb, "re-staging before the show",
                                                None, None))
            rows.append(row)
        time_s += extra_steps * DT
    selected_pitch = None
    for pitch in (20, 15, 10):
        aim_camera(model, base.CAMERA, estimate[:2], track["center_xy_m"], pitch)
        model.cam_pos[model.camera(base.CAMERA).id, :2] = actual
        _prop(model, time_s)
        mujoco.mj_forward(model, data)
        before_rgb, before_depth, age = owner.capture(renderer, data, base.CAMERA,
                                                       use_depth)
        time_s += max(.1, age)
        healthy_before, fractions = (
            depth_change.packet_health(before_depth, before_depth)
            if before_depth is not None else (False, [0., 0.]))
        rows.append({"stage": "before_show_view_health", "time_s": round(time_s, 4),
                     "pitch_degrees": pitch,
                     "central_valid_fraction": fractions[0],
                     "depth_healthy": healthy_before})
        frames.append(owner.owner_frame(before_rgb,
                      f"checking owner depth view: {pitch} degrees", None, None))
        if healthy_before:
            selected_pitch = pitch
            break
    if selected_pitch is None:
        return "owner_pre_show_depth_view_unavailable", rows, frames, None, None, time_s
    frames.append(owner.owner_frame(before_rgb, "range stop: ask for a show",
                                    None, None))
    rows.append({"stage": "ask_owner_to_show_before_standoff",
                 "time_s": round(time_s, 4),
                 "speech_intent": "Could you show me the taught thing from there?",
                 "selected_depth_pitch_degrees": selected_pitch,
                 "range_track_actor": track})
    pose = None
    for index, fraction in enumerate(np.linspace(.2, 1., 5)):
        pose = _show_owner(model, float(fraction), first_south)
        _prop(model, time_s)
        mujoco.mj_forward(model, data)
        rgb, _, age = owner.capture(renderer, data, base.CAMERA, False)
        time_s += max(.12, age)
        frames.append(owner.owner_frame(rgb, "owner presents item", None, None))
        rows.append({"stage": "owner_reaches", "index": index,
                     "time_s": round(time_s, 4), "pose_evaluator": pose})
    after_rgb, after_depth, age = owner.capture(renderer, data, base.CAMERA,
                                                 use_depth)
    time_s += max(.1, age)
    if before_depth is None or after_depth is None:
        rows.append({"stage": "owner_change_depth_unavailable",
                     "time_s": round(time_s, 4)})
        return "owner_standoff_unknown_without_depth", rows, frames, None, None, time_s
    raw_before, raw_after = before_depth.copy(), after_depth.copy()
    if DEPTH_NOISE_SIGMA_M:
        before_depth += np.random.default_rng(51701).normal(
            0., DEPTH_NOISE_SIGMA_M, before_depth.shape)
        after_depth += np.random.default_rng(51702).normal(
            0., DEPTH_NOISE_SIGMA_M, after_depth.shape)
    if DEPTH_DROPOUT_FRACTION:
        drops = np.random.default_rng(51703).random(after_depth.shape)
        after_depth[drops < DEPTH_DROPOUT_FRACTION] = np.nan
    healthy, central_coverage = depth_change.packet_health(
        before_depth, after_depth)
    cam = model.camera(base.CAMERA).id
    camera_est = data.cam_xpos[cam].copy()
    camera_est[:2] = estimate[:2]
    if healthy:
        mask, changed, pixels = changed_points(
            before_depth, after_depth, data.cam_xmat[cam].reshape(3, 3),
            camera_est, np.asarray(track["center_xy_m"]))
    else:
        mask = np.zeros_like(after_depth, dtype=bool)
        changed = np.empty((0, 3))
    packet = None
    if PACKET_OUT is not None:
        PACKET_OUT.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(PACKET_OUT, before_depth=before_depth,
                            after_depth=after_depth, raw_before_depth=raw_before,
                            raw_after_depth=raw_after, changed_mask=mask)
        packet = {"path": str(PACKET_OUT),
                  "sha256": hashlib.sha256(PACKET_OUT.read_bytes()).hexdigest()}
    rows.append({"stage": "owner_depth_change", "time_s": round(time_s, 4),
                 "capture_pose_est_xy_m": estimate[:2].tolist(),
                 "depth_packet_healthy": healthy,
                 "central_valid_fractions": central_coverage,
                 "injected_depth_noise_sigma_m": DEPTH_NOISE_SIGMA_M,
                 "injected_after_depth_dropout_fraction": DEPTH_DROPOUT_FRACTION,
                 "raw_change_pixels": int(mask.sum()),
                 "filtered_change_points": int(len(changed)),
                 "change_point_sha256": hashlib.sha256(changed.tobytes()).hexdigest(),
                 "depth_packets": packet})
    frames.append(owner.owner_frame(after_rgb, "depth change: shown hand and item",
                                    None, None))
    if len(changed) < 50:
        return "owner_standoff_unknown_without_change", rows, frames, None, None, time_s
    target, candidates = _choose_standoff(grid, changed, estimate)
    rows.append({"stage": "choose_owner_standoff", "time_s": round(time_s, 4),
                 "candidates": candidates, "chosen": target})
    if target is None:
        return "owner_standoff_not_supported", rows, frames, None, None, time_s
    start_actual = actual.copy()
    start_estimate = estimate[:2].copy()
    distance = target["distance_m"]
    steps = int(round(distance / BACKTRACK_SPEED_M_S / DT))
    return_start_s = time_s
    return_track = ReturnOwnerTrack(track["center_xy_m"])

    def owner_south_at(t):
        return (first_south + SECOND_RETURN_OWNER_M * min(1., max(0.,
                (t - return_start_s) / SECOND_RETURN_OWNER_S))) if moving_owner else 0.

    for index in range(1, steps + 1):
        t = time_s + index * DT
        amount = min(distance, BACKTRACK_SPEED_M_S * index * DT)
        owner_south = owner_south_at(t)
        actual[:] = start_actual + [0., -amount]
        estimate[:2] = start_estimate + [0., -amount]
        model.body_pos[model.body("audit_ape").id, :2] = actual
        aim_camera(model, base.CAMERA, estimate[:2], track["center_xy_m"], 20)
        model.cam_pos[model.camera(base.CAMERA).id, :2] = actual
        moving_pose = _place_owner(model, owner_south)
        _prop(model, t)
        mujoco.mj_forward(model, data)
        row = {"stage": "standoff_backtrack", "time_s": round(t, 4),
               "ape_xy_m": actual.tolist(),
               "estimated_xy_m": estimate[:2].tolist(),
               "owner_pose_evaluator": moving_pose,
               "skill_status": "kinematic_surrogate_unqualified"}
        if index % 5 == 0:
            packet = SCAN(model, data, actual, 5., np.random.default_rng(7000 + index))
            row["range_returns"] = sum(x is not None for x in packet)
            observation = return_track.observe(packet, estimate[:2], t)
            row["owner_return_motion_packet"] = observation
            if observation["alarm_now"]:
                row["actor_action"] = "invalidate_show_pose_request_owner_pause"
                row["speech_intent"] = "Please hold still while I move back."
            rgb, _, _ = owner.capture(renderer, data, base.CAMERA, False)
            frames.append(owner.owner_frame(rgb, "returning to owner standoff", None, None))
        rows.append(row)
    time_s += steps * DT
    rows.append({"stage": "standoff_reached_after_owner_show",
                 "time_s": round(time_s, 4),
                 "ape_xy_m": actual.tolist(), "estimated_xy_m": estimate[:2].tolist()})
    rows.append({"stage": "owner_return_motion_assessment", "time_s": round(time_s, 4),
                 "owner_motion_seen_actor": return_track.first_alarm_s is not None,
                 "first_motion_alarm_time_s_actor": return_track.first_alarm_s,
                 "maximum_observed_south_shift_m": return_track.max_south_shift_m,
                 "assumed_owner_motion_bound_m": OWNER_MOTION_BOUND_M})
    if return_track.max_south_shift_m > OWNER_MOTION_BOUND_M:
        return ("owner_exceeded_standoff_motion_bound", rows, frames,
                None, None, time_s)
    captured, _, age = owner.capture(renderer, data, base.CAMERA, False)
    _, proposals, ranked, chosen, recognition_wall = monitored._score(
        vision, exemplar, captured)
    first_capture = time_s
    pending_until = first_capture + max(4., age + recognition_wall)
    steps = int(math.ceil((pending_until - first_capture) / DT))
    packet_history = []
    baseline = None
    moving_streak = 0
    motion_seen = False
    settled_streak = 0
    settled_time = None
    owner_pose = None
    for index in range(steps + 1):
        t = first_capture + index * DT
        south = (owner_south_at(t) + OWNER_APPROACH_M * min(
                 1., max(0., (t - first_capture) / OWNER_APPROACH_S))) if moving_owner else 0.
        owner_pose = _place_owner(model, south)
        _prop(model, t)
        mujoco.mj_forward(model, data)
        row = {"stage": "standoff_recognition_monitor", "time_s": round(t, 4),
               "ape_xy_m": actual.tolist(), "owner_pose_evaluator": owner_pose,
               "actor_state": "owner_motion_seen_wait" if motion_seen else "recognition_pending"}
        if index % 5 == 0:
            packet = SCAN(model, data, actual, 5., np.random.default_rng(8000 + index))
            if baseline is None:
                baseline = packet
            changed = changed_beams(baseline, packet)
            moving_streak = moving_streak + 1 if changed >= 3 else 0
            if moving_streak >= 2 and not motion_seen:
                motion_seen = True
                row["actor_action"] = "invalidate_image_observe_from_standoff"
            packet_history.append((t, packet))
            rolling = (changed_beams(packet_history[-5][1], packet)
                       if len(packet_history) >= 5 else None)
            if motion_seen and rolling is not None:
                settled_streak = settled_streak + 1 if rolling < 3 else 0
                if settled_streak >= 2 and settled_time is None:
                    settled_time = t
                    row["actor_action"] = "owner_motion_settled_from_range"
            row["range_packet"] = {"changed_from_first": changed,
                                   "changed_over_0_4s": rolling,
                                   "returns": sum(x is not None for x in packet)}
            rgb, _, _ = owner.capture(renderer, data, base.CAMERA, False)
            frames.append(owner.owner_frame(rgb,
                          "watching owner from clear standoff", None, None))
        rows.append(row)
    time_s = first_capture + steps * DT
    rows.append({"stage": "slow_recognition_result_ready", "time_s": round(time_s, 4),
                 "recognition_wall_s": recognition_wall,
                 "original_capture_time_s": round(first_capture, 4),
                 "owner_motion_seen_actor": motion_seen,
                 "owner_settled_time_s_actor": round(settled_time, 4) if settled_time else None})
    if motion_seen and settled_time is None:
        return "owner_motion_unresolved", rows, frames, recognition_wall, None, time_s
    if motion_seen:
        # Appearance/position from the first frame is stale; wait for a new one.
        captured, _, age = owner.capture(renderer, data, base.CAMERA, False)
        image, proposals, ranked, chosen, second_wall = monitored._score(
            vision, exemplar, captured)
        second_start = time_s
        second_steps = int(math.ceil(max(4., age + second_wall) / .1))
        fresh_baseline = None
        fresh_streak = 0
        for index in range(second_steps + 1):
            t = second_start + index * .1
            _prop(model, t)
            mujoco.mj_forward(model, data)
            packet = SCAN(model, data, actual, 5., np.random.default_rng(9000 + index))
            if fresh_baseline is None:
                fresh_baseline = packet
            count = changed_beams(fresh_baseline, packet)
            fresh_streak = fresh_streak + 1 if count >= 3 else 0
            rows.append({"stage": "standoff_fresh_recognition_monitor",
                         "time_s": round(t, 4), "ape_xy_m": actual.tolist(),
                         "owner_pose_evaluator": owner_pose,
                         "changed_beams_from_baseline": count})
            if fresh_streak >= 2:
                return ("owner_motion_unresolved", rows, frames,
                        recognition_wall + second_wall, None, t)
            if index % 2 == 0:
                rgb, _, _ = owner.capture(renderer, data, base.CAMERA, False)
                frames.append(owner.owner_frame(rgb, "fresh recognition with live range",
                                                None, None))
        time_s = second_start + second_steps * .1
        recognition_wall += second_wall
    else:
        image = Image.fromarray(captured)
    gt_box, pixels = evaluator_box(renderer, data, model, base.CAMERA,
                                    "shown_can_geom")
    frames.append(owner.owner_frame(captured, "taught-item proposal after standoff",
                                    chosen[0]["box"] if chosen else None, gt_box))
    rows.append({"stage": "recognize_after_standoff", "time_s": round(time_s, 4),
                 "proposal_count": len(proposals),
                 "top_similarity": float(ranked[0][1]) if ranked else None,
                 "selected_box": chosen[0]["box"] if chosen else None,
                 "selected_iou_evaluator": owner.iou(
                     chosen[0]["box"] if chosen else None, gt_box),
                 "can_pixels_evaluator": pixels})
    if chosen is None:
        return "visible_item_not_reacquired", rows, frames, recognition_wall, None, time_s
    crop = image.crop(chosen[0]["box"])
    tracker = owner.FastActor(cv2.cvtColor(np.asarray(crop), cv2.COLOR_RGB2BGR),
                              use_depth=False)
    point_error, final_track = None, None
    for index in range(3):
        _prop(model, time_s)
        mujoco.mj_forward(model, data)
        rgb, depth, capture_wall = owner.capture(renderer, data, base.CAMERA, use_depth)
        begin = time.perf_counter()
        final_track = tracker.observe(Image.fromarray(rgb), None, float(estimate[0]))
        frame_age = capture_wall + time.perf_counter() - begin
        if frame_age > owner.MAX_FRESH_AGE_S:
            final_track["status"] = "stale_packet"
            tracker.reset_continuity()
        point = None
        if final_track["box"] is not None and depth is not None:
            point = depth_point(final_track["box"], depth, data, model, base.CAMERA)
            if point is not None:
                point = np.asarray(point)
                point[:2] += estimate[:2] - actual
                point = point.tolist()
                point_error = float(np.linalg.norm(
                    np.asarray(point) - model.body_pos[model.body("shown_can").id]))
        gt_box, pixels = evaluator_box(renderer, data, model, base.CAMERA,
                                        "shown_can_geom")
        frames.append(owner.owner_frame(rgb, f"fresh owner view: {final_track['status']}",
                                        final_track["box"], gt_box))
        time_s += max(.1, frame_age)
        rows.append({"stage": "fresh_owner_view", "time_s": round(time_s, 4),
                     "actor": final_track, "depth_point_xyz_m": point,
                     "depth_point_error_evaluator_m": point_error,
                     "selected_iou_evaluator": owner.iou(final_track["box"], gt_box)})
        if final_track["status"] == "visually_reacquired_unlocalized":
            break
    if final_track is not None and final_track["status"] == "visually_reacquired_unlocalized":
        rows.append({"stage": "owner_confirms_taught_item", "time_s": round(time_s, 4),
                     "owner_reply": "yes_that_is_the_taught_can"})
        outcome = ("owner_confirmed_metric_find" if point_error is not None else
                   "owner_confirmed_visual_find")
    else:
        outcome = "unconfirmed_after_slow_recognition"
    return outcome, rows, frames, recognition_wall, point_error, time_s
