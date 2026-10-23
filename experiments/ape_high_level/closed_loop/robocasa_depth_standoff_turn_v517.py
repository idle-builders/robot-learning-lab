"""Choose owner standoff from connected, health-gated presentation depth.

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

import cv2
import mujoco
import numpy as np
from PIL import Image

import robocasa_owner_person as person
import robocasa_person_interaction as base
import robocasa_person_monitored_turn_v502 as monitored
import run_robocasa_joined_owner_route_v410 as owner
import run_robocasa_layout2_multitrack_v485 as multi
from analyze_robocasa_owner_monitor_v501 import changed_beams
from run_robocasa_scan_height_v433 import height_scan
from screen_robocasa_owner_person_v489 import evaluator_box
from screen_robocasa_owner_safe_stop_v494 import aim_camera, depth_point
import owner_depth_change_component_v516 as depth_change


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
DT = .02
PROP_SCRIPT = None
PACKET_OUT: Path | None = None
DEPTH_NOISE_SIGMA_M = 0.
DEPTH_DROPOUT_FRACTION = 0.


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


def owner_turn(response, model, data, renderer, actual, estimate,
               time_s, use_depth, vision, exemplar):
    moving_owner = not response.startswith("stationary")
    grid = multi.CAPTURED[-1]
    track = max(grid.multi_history[-1]["tracks"], key=lambda row: row["hits"])
    rows, frames = [], []
    aim_camera(model, base.CAMERA, estimate[:2], track["center_xy_m"], 20)
    model.cam_pos[model.camera(base.CAMERA).id, :2] = actual
    _prop(model, time_s)
    mujoco.mj_forward(model, data)
    before_rgb, before_depth, age = owner.capture(renderer, data, base.CAMERA,
                                                   use_depth)
    time_s += max(.1, age)
    frames.append(owner.owner_frame(before_rgb, "range stop: ask for a show",
                                    None, None))
    rows.append({"stage": "ask_owner_to_show_before_standoff",
                 "time_s": round(time_s, 4),
                 "speech_intent": "Could you show me the taught thing from there?",
                 "range_track_actor": track})
    pose = None
    for index, fraction in enumerate(np.linspace(.2, 1., 5)):
        pose = base.show_pose(model, float(fraction))
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
    for index in range(1, steps + 1):
        t = time_s + index * DT
        amount = min(distance, BACKTRACK_SPEED_M_S * index * DT)
        actual[:] = start_actual + [0., -amount]
        estimate[:2] = start_estimate + [0., -amount]
        model.body_pos[model.body("audit_ape").id, :2] = actual
        aim_camera(model, base.CAMERA, estimate[:2], track["center_xy_m"], 20)
        model.cam_pos[model.camera(base.CAMERA).id, :2] = actual
        _prop(model, t)
        mujoco.mj_forward(model, data)
        row = {"stage": "standoff_backtrack", "time_s": round(t, 4),
               "ape_xy_m": actual.tolist(),
               "estimated_xy_m": estimate[:2].tolist(),
               "owner_pose_evaluator": pose,
               "skill_status": "kinematic_surrogate_unqualified"}
        if index % 5 == 0:
            packet = SCAN(model, data, actual, 5., np.random.default_rng(7000 + index))
            row["range_returns"] = sum(x is not None for x in packet)
            rgb, _, _ = owner.capture(renderer, data, base.CAMERA, False)
            frames.append(owner.owner_frame(rgb, "returning to owner standoff", None, None))
        rows.append(row)
    time_s += steps * DT
    rows.append({"stage": "standoff_reached_after_owner_show",
                 "time_s": round(time_s, 4),
                 "ape_xy_m": actual.tolist(), "estimated_xy_m": estimate[:2].tolist()})
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
        south = (OWNER_APPROACH_M * min(1., max(0.,
                 (t - first_capture) / OWNER_APPROACH_S)) if moving_owner else 0.)
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
