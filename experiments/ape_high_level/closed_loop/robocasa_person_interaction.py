"""Owner show-and-confirm turn from a range-track-induced safe stop.

The camera aims from the localized ape pose at the range track. A human label,
item position, and segmentation box are not available to the actor. The
owner reply and arm motion are scripted; segmentation and true positions are
only used to evaluate the image/depth result afterward.
"""

from __future__ import annotations

import time
from concurrent.futures import ThreadPoolExecutor

import cv2
import mujoco
import numpy as np
from PIL import Image

import robocasa_owner_person as person
import run_robocasa_joined_owner_route_v410 as owner
import run_robocasa_layout2_multitrack_v485 as multi
import run_robocasa_moving_prop_v423 as moving
import screen_robocasa_owner_person_v489 as evaluator
from screen_robocasa_owner_safe_stop_v494 import aim_camera, depth_point


CAMERA = "owner_wide20"
TORSO_XY = np.array([1., -2.4])
REST_HAND = np.array([1.10, -2.24, .43])
SHOW_HAND = np.array([1.30, -2.80, .55])


def show_pose(model, fraction):
    hand = REST_HAND + fraction * (SHOW_HAND - REST_HAND)
    pose = person.set_owner_pose(model, TORSO_XY, hand, show_can=True)
    reach = hand[:2] - TORSO_XY
    reach /= np.linalg.norm(reach)
    model.body_pos[model.body("shown_can").id, :2] = hand[:2] + .14 * reach
    pose["can_xyz_m"] = model.body_pos[model.body("shown_can").id].tolist()
    return pose


def owner_turn(response, model, data, renderer, actual, estimate,
               time_s, use_depth, vision, exemplar, person_target_xy_m=None,
               inference_monitor=None, on_inference_capture=None):
    track = None
    if person_target_xy_m is None:
        track = max(multi.CAPTURED[-1].multi_history[-1]["tracks"],
                    key=lambda item: item["hits"])
        person_target_xy_m = track["center_xy_m"]
    aim_camera(model, CAMERA, estimate[:2], person_target_xy_m, 20)
    model.cam_pos[model.camera(CAMERA).id, :2] = actual
    mujoco.mj_forward(model, data)
    rows, frames = [], []
    before, _, capture_wall = owner.capture(renderer, data, CAMERA, use_depth)
    time_s += max(.1, capture_wall)
    source = "range_track" if track is not None else "fresh_person_rgbd"
    frames.append(owner.owner_frame(before, f"{source}: ask person to show item", None, None))
    row = {"stage": "ask_owner_to_show_from_safe_stop", "time_s": round(time_s, 3),
           "person_source": source, "person_target_xy_m": list(person_target_xy_m),
           "action": "please_show_taught_can_and_hold_still"}
    if track is not None:
        row.update({"range_track_id": track["id"],
                    "range_track_center_xy_m": track["center_xy_m"]})
    rows.append(row)
    waits = 3 if response == "ignored" else 2 if response == "delayed" else 0
    for index in range(waits):
        rgb, _, age = owner.capture(renderer, data, CAMERA, use_depth)
        time_s += max(.1, age)
        frames.append(owner.owner_frame(rgb, "waiting for owner response", None, None))
        rows.append({"stage": "owner_response_wait", "index": index,
                     "time_s": round(time_s, 3)})
    if response == "ignored":
        return "unknown_ask_for_help", rows, frames, None, None, time_s
    for index, fraction in enumerate(np.linspace(.2, 1., 5)):
        pose = show_pose(model, float(fraction))
        mujoco.mj_forward(model, data)
        rgb, _, age = owner.capture(renderer, data, CAMERA, use_depth)
        time_s += max(.12, age)
        frames.append(owner.owner_frame(rgb, "owner presents item", None, None))
        rows.append({"stage": "owner_reaches", "index": index,
                     "time_s": round(time_s, 3), "pose_evaluator": pose})
    shown_rgb, _, age = owner.capture(renderer, data, CAMERA, use_depth)
    capture_time_s = time_s + age
    if on_inference_capture is not None:
        on_inference_capture(capture_time_s)
    image = Image.fromarray(shown_rgb)
    def infer():
        start = time.perf_counter()
        proposals = vision.detect(image)
        scores = (vision.feature([image.crop(item["box"]) for item in proposals]) @ exemplar
                  if proposals else np.empty(0))
        ranked = sorted(zip(proposals, scores), key=lambda pair: float(pair[1]), reverse=True)
        return proposals, ranked, time.perf_counter() - start

    monitor_rows = []
    alarm_time_s = None
    alarm_reason = None
    if inference_monitor is None:
        proposals, ranked, recognition_wall = infer()
        inference_elapsed = recognition_wall
    else:
        # The image worker never touches MuJoCo. The main thread keeps scene
        # time and the independent near-body sensor alive while it runs.
        with ThreadPoolExecutor(max_workers=1) as pool:
            started = time.perf_counter()
            future = pool.submit(infer)
            next_tick = started
            while True:
                now = time.perf_counter()
                if now < next_tick:
                    time.sleep(next_tick - now)
                    now = time.perf_counter()
                elapsed = now - started
                sample_time_s = capture_time_s + elapsed
                observation = inference_monitor(sample_time_s)
                if isinstance(observation, dict):
                    guard_range = observation["actor_guard_range_m"]
                    monitor_rows.append({"time_s": round(sample_time_s, 4),
                                         **observation})
                    invalidate = bool(observation.get("invalidate_image", False))
                else:
                    guard_range = observation
                    monitor_rows.append({"time_s": round(sample_time_s, 4),
                                         "actor_guard_range_m": guard_range})
                    invalidate = False
                if (guard_range < moving.GUARD_CENTER_RANGE_M or invalidate) and alarm_time_s is None:
                    alarm_time_s = sample_time_s
                    alarm_reason = ("moving_range_track" if invalidate else
                                    "near_body_range")
                if future.done():
                    break
                next_tick += .02
                if next_tick < now:
                    next_tick = now + .02
            proposals, ranked, recognition_wall = future.result()
            inference_elapsed = time.perf_counter() - started
        rows.append({"stage": "owner_inference_near_body_monitor",
                     "time_s": round(capture_time_s + inference_elapsed, 4),
                     "image_capture_time_s": round(capture_time_s, 4),
                     "monitor_samples": monitor_rows,
                     "first_alarm_time_s": None if alarm_time_s is None else round(alarm_time_s, 4),
                     "first_alarm_reason": alarm_reason,
                     "worker_wall_s": recognition_wall,
                     "image_invalidated": alarm_time_s is not None})
    time_s += max(.1, age + inference_elapsed)
    chosen = ranked[0] if ranked and float(ranked[0][1]) >= owner.THRESHOLD else None
    gt_box, gt_pixels = evaluator.evaluator_box(renderer, data, model, CAMERA,
                                                 "shown_can_geom")
    frames.append(owner.owner_frame(shown_rgb, "slow taught-item match; wait for fresh view",
                                    chosen[0]["box"] if chosen else None, gt_box))
    rows.append({"stage": "recognize_after_owner_show", "time_s": round(time_s, 3),
                 "proposal_count": len(proposals),
                 "top_similarity": float(ranked[0][1]) if ranked else None,
                 "selected_box": chosen[0]["box"] if chosen else None,
                 "selected_iou_evaluator": owner.iou(
                     chosen[0]["box"] if chosen else None, gt_box),
                 "recognition_wall_s": recognition_wall,
                 "shown_visible_pixels_evaluator": gt_pixels})
    if alarm_time_s is not None:
        rows.append({"stage": "owner_inference_interrupted",
                     "time_s": round(time_s, 3),
                     "reason": alarm_reason,
                     "item_claim": "none"})
        return "owner_inference_interrupted_near_body", rows, frames, recognition_wall, None, time_s
    if response == "withdrawn":
        model.body_pos[model.body("shown_can").id] = [-10., -10., .43]
        mujoco.mj_forward(model, data)
        rows.append({"stage": "owner_withdraws_during_recognition",
                     "time_s": round(time_s, 3)})
    if chosen is None:
        return "visible_item_not_reacquired", rows, frames, recognition_wall, None, time_s
    crop = image.crop(chosen[0]["box"])
    tracker = owner.FastActor(cv2.cvtColor(np.asarray(crop), cv2.COLOR_RGB2BGR),
                              use_depth=False)
    fresh_track, point_est, point_error = None, None, None
    for index in range(3):
        rgb, depth, capture_wall = owner.capture(renderer, data, CAMERA, use_depth)
        start = time.perf_counter()
        fresh_track = tracker.observe(Image.fromarray(rgb), None, float(estimate[0]))
        age = capture_wall + time.perf_counter() - start
        if age > owner.MAX_FRESH_AGE_S:
            fresh_track["status"] = "stale_packet"
            tracker.reset_continuity()
        if fresh_track["box"] is not None and depth is not None:
            point_est = depth_point(fresh_track["box"], depth, data, model, CAMERA)
            if point_est is not None:
                point_est = np.asarray(point_est)
                point_est[:2] += estimate[:2] - actual
                point_est = point_est.tolist()
                point_error = float(np.linalg.norm(
                    np.asarray(point_est) - model.body_pos[model.body("shown_can").id]))
        gt_box, gt_pixels = evaluator.evaluator_box(renderer, data, model, CAMERA,
                                                     "shown_can_geom")
        frames.append(owner.owner_frame(rgb,
                      f"fresh owner view {index}: {fresh_track['status']}",
                      fresh_track["box"], gt_box))
        time_s += max(.1, age)
        rows.append({"stage": "fresh_owner_view", "index": index,
                     "time_s": round(time_s, 3), "frame_age_s": age,
                     "actor": fresh_track, "depth_point_xyz_m": point_est,
                     "depth_point_error_evaluator_m": point_error,
                     "selected_iou_evaluator": owner.iou(fresh_track["box"], gt_box),
                     "shown_visible_pixels_evaluator": gt_pixels})
        if fresh_track["status"] == "visually_reacquired_unlocalized":
            break
    if fresh_track is not None and fresh_track["status"] == "visually_reacquired_unlocalized":
        rows.append({"stage": "owner_confirms_taught_item", "time_s": round(time_s, 3),
                     "owner_reply": "yes_that_is_the_taught_can"})
        outcome = "owner_confirmed_metric_find" if use_depth else "owner_confirmed_visual_find"
    else:
        outcome = "unconfirmed_after_slow_recognition"
    return outcome, rows, frames, recognition_wall, point_error, time_s
