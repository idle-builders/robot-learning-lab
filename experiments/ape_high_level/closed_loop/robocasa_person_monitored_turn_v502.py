"""Owner turn with capture-time range monitoring during slow recognition.

The slow recognizer runs on a captured image; its result is withheld while
simulation time advances in 20 ms steps and 10 Hz range packets are handled.
This is a discrete-event concurrency model, not a real-time threaded runtime.
An actor-packet change asks the owner to pause. If fresh packets still show
movement 0.4 s later, the high-level controller requests a backward skill;
the episode executes a labeled kinematic surrogate to test its geometry.
"""

from __future__ import annotations

import math
import time

import cv2
import mujoco
import numpy as np
from PIL import Image

import robocasa_owner_person as person
import robocasa_person_interaction as base
import run_robocasa_joined_owner_route_v410 as owner
import run_robocasa_layout2_multitrack_v485 as multi
from analyze_robocasa_owner_monitor_v501 import changed_beams
from run_robocasa_scan_height_v433 import height_scan
from screen_robocasa_owner_person_v489 import evaluator_box
from screen_robocasa_owner_safe_stop_v494 import aim_camera, depth_point


SCAN_LOW_MID = height_scan((.05, .30))
SCAN_UPPER = height_scan((.55,))
BEAM_THRESHOLD = 3
RESPONSE_CHANGE_THRESHOLD = {"lowmid": 3, "upper": 5}
STREAK_N = 2
SCAN_S = .1
SUBSTEP_S = .02
APPROACH_M = .4
APPROACH_S = 2.
OWNER_PAUSE_DELAY_S = .2
RESPONSE_CHECK_S = .4
RETREAT_SPEED_M_S = .25
RETREAT_MAX_M = .3
MIN_RECOGNITION_S = 4.
PROP_SCRIPT = None  # Set by the episode runner for this live scenario.


def _present_owner(model, south_m):
    torso = base.TORSO_XY + [0., -south_m]
    hand = base.SHOW_HAND + [0., -south_m, 0.]
    person.set_owner_pose(model, torso, hand, show_can=True)
    direction = hand[:2] - torso
    direction /= np.linalg.norm(direction)
    model.body_pos[model.body("shown_can").id, :2] = hand[:2] + .14 * direction
    return {"torso_xy_m": torso.tolist(), "hand_xyz_m": hand.tolist(),
            "can_xyz_m": model.body_pos[model.body("shown_can").id].tolist()}


def _score(vision, exemplar, rgb):
    image = Image.fromarray(rgb)
    begin = time.perf_counter()
    proposals = vision.detect(image)
    scores = (vision.feature([image.crop(item["box"]) for item in proposals]) @ exemplar
              if proposals else np.empty(0))
    ranked = sorted(zip(proposals, scores), key=lambda pair: float(pair[1]), reverse=True)
    wall = time.perf_counter() - begin
    chosen = ranked[0] if ranked and float(ranked[0][1]) >= owner.THRESHOLD else None
    return image, proposals, ranked, chosen, wall


def _actor_frame(rgb, label, chosen=None, gt_box=None):
    return owner.owner_frame(rgb, label, chosen, gt_box)


def _advance_prop(model, time_s):
    if PROP_SCRIPT is not None:
        model.body_pos[model.body("route_obstacle").id, :2] = PROP_SCRIPT(time_s)


def owner_turn(response, model, data, renderer, actual, estimate,
               time_s, use_depth, vision, exemplar):
    approach = not response.startswith("stationary")
    monitor = ("upper" if response.endswith("upper") else
               "lowmid" if "lowmid" in response else "off")
    owner_obeys = response.startswith("approach_pause")
    scan = SCAN_UPPER if monitor == "upper" else SCAN_LOW_MID
    track = max(multi.CAPTURED[-1].multi_history[-1]["tracks"],
                key=lambda item: item["hits"])
    aim_camera(model, base.CAMERA, estimate[:2], track["center_xy_m"], 20)
    model.cam_pos[model.camera(base.CAMERA).id, :2] = actual
    mujoco.mj_forward(model, data)
    rows, frames = [], []
    before, _, age = owner.capture(renderer, data, base.CAMERA, False)
    time_s += max(.1, age)
    frames.append(_actor_frame(before, "ask owner to show taught item"))
    rows.append({"stage": "ask_owner_to_show", "time_s": round(time_s, 4),
                 "range_track_id": track["id"],
                 "range_track_center_xy_m": track["center_xy_m"]})
    for index, fraction in enumerate(np.linspace(.2, 1., 5)):
        pose = base.show_pose(model, float(fraction))
        mujoco.mj_forward(model, data)
        rgb, _, age = owner.capture(renderer, data, base.CAMERA, False)
        time_s += max(.12, age)
        frames.append(_actor_frame(rgb, "owner presents item"))
        rows.append({"stage": "owner_reaches", "index": index,
                     "time_s": round(time_s, 4), "pose_evaluator": pose})
    captured, _, age = owner.capture(renderer, data, base.CAMERA, False)
    first_image, proposals, ranked, chosen, recognition_wall = _score(
        vision, exemplar, captured)
    capture_time = time_s
    ready_time = capture_time + max(MIN_RECOGNITION_S, age + recognition_wall)
    start_xy, start_est = actual.copy(), estimate[:2].copy()
    owner_south = 0.
    alarm, alarm_packet, streak = None, None, 0
    baseline = None
    retreat_start, pause_confirmed = None, False
    response_evidence = []
    episode_end = max(ready_time, capture_time + APPROACH_S + .4)
    steps = int(math.ceil((episode_end - capture_time) / SUBSTEP_S))
    for index in range(steps + 1):
        t = capture_time + index * SUBSTEP_S
        commanded_south = (APPROACH_M * min(1., max(0., (t - capture_time) / APPROACH_S))
                           if approach else 0.)
        if owner_obeys and alarm is not None and t >= alarm + OWNER_PAUSE_DELAY_S:
            commanded_south = min(commanded_south,
                                  APPROACH_M * min(1., (alarm + OWNER_PAUSE_DELAY_S -
                                                        capture_time) / APPROACH_S))
        owner_south = commanded_south
        pose = _present_owner(model, owner_south)
        _advance_prop(model, t)
        retreat = (min(RETREAT_MAX_M, RETREAT_SPEED_M_S * max(0., t - retreat_start))
                   if retreat_start is not None else 0.)
        actual[:] = start_xy + [0., -retreat]
        estimate[:2] = start_est + [0., -retreat]
        model.body_pos[model.body("audit_ape").id, :2] = actual
        aim_camera(model, base.CAMERA, estimate[:2], track["center_xy_m"], 20)
        model.cam_pos[model.camera(base.CAMERA).id, :2] = actual
        mujoco.mj_forward(model, data)
        tick = {"stage": "recognition_pending_monitor", "time_s": round(t, 4),
                "ape_xy_m": actual.tolist(), "owner_pose_evaluator": pose,
                "owner_south_evaluator_m": owner_south,
                "actor_state": ("retreat_requested" if retreat_start is not None else
                                "pause_observed" if pause_confirmed else
                                "pause_requested" if alarm is not None else
                                "recognition_pending")}
        if index % 5 == 0 and monitor != "off":
            packet = scan(model, data, actual, 5., np.random.default_rng(5000 + index))
            if baseline is None:
                baseline = packet
            changes = changed_beams(baseline, packet)
            streak = streak + 1 if changes >= BEAM_THRESHOLD else 0
            tick["range_packet"] = {"source": monitor, "capture_time_s": round(t, 4),
                                    "changed_beams_from_baseline": changes,
                                    "returns": sum(value is not None for value in packet)}
            if alarm is None and streak >= STREAK_N:
                alarm, alarm_packet = t, packet
                tick["actor_action"] = "ask_owner_pause_and_cancel_stale_recognition"
            elif alarm is not None and retreat_start is None and not pause_confirmed:
                if t >= alarm + RESPONSE_CHECK_S - 1e-8:
                    changes_since_alarm = changed_beams(alarm_packet, packet)
                    response_evidence.append({"time_s": round(t, 4),
                                              "changed_beams_since_alarm": changes_since_alarm})
                    tick["changed_beams_since_alarm"] = changes_since_alarm
                    response_threshold = RESPONSE_CHANGE_THRESHOLD[monitor]
                    if changes_since_alarm >= response_threshold:
                        retreat_start = t
                        tick["actor_action"] = "request_backtrack_skill_unverified"
                    elif len(response_evidence) >= 2 and all(
                            entry["changed_beams_since_alarm"] < response_threshold
                            for entry in response_evidence[-2:]):
                        pause_confirmed = True
                        tick["actor_action"] = "owner_pause_observed_from_range"
        if index % 5 == 0:
            rgb, _, _ = owner.capture(renderer, data, base.CAMERA, False)
            label = f"{response}: {tick['actor_state']}"
            if tick.get("actor_action"):
                label += f"; {tick['actor_action']}"
            frames.append(_actor_frame(rgb, label))
        rows.append(tick)
    time_s = capture_time + steps * SUBSTEP_S
    rows.append({"stage": "slow_recognition_result_ready", "time_s": round(time_s, 4),
                 "recognition_wall_s": recognition_wall,
                 "original_capture_time_s": round(capture_time, 4),
                 "original_top_similarity": float(ranked[0][1]) if ranked else None,
                 "stale_due_to_owner_motion": alarm is not None,
                 "pause_requested_s": round(alarm, 4) if alarm is not None else None,
                 "pause_confirmed_actor": pause_confirmed,
                 "retreat_requested_s": round(retreat_start, 4)
                 if retreat_start is not None else None})
    if retreat_start is not None:
        return "owner_approach_retreat_requested", rows, frames, recognition_wall, None, time_s
    if alarm is not None and not pause_confirmed:
        return "owner_motion_unresolved", rows, frames, recognition_wall, None, time_s
    if alarm is not None:
        # The first image predates owner motion. Reacquire before any claim.
        captured, _, age = owner.capture(renderer, data, base.CAMERA, False)
        first_image, proposals, ranked, chosen, second_wall = _score(
            vision, exemplar, captured)
        second_capture = time_s
        second_ready = second_capture + max(MIN_RECOGNITION_S, age + second_wall)
        rows.append({"stage": "recognize_fresh_after_pause", "time_s": round(second_capture, 4),
                     "recognition_wall_s": second_wall})
        baseline2 = None
        second_steps = int(math.ceil((second_ready - second_capture) / SCAN_S))
        second_streak = 0
        for index in range(second_steps + 1):
            t = second_capture + index * SCAN_S
            _advance_prop(model, t)
            mujoco.mj_forward(model, data)
            packet = scan(model, data, actual, 5., np.random.default_rng(9000 + index))
            if baseline2 is None:
                baseline2 = packet
            count = changed_beams(baseline2, packet)
            second_streak = second_streak + 1 if count >= BEAM_THRESHOLD else 0
            rows.append({"stage": "fresh_recognition_monitor", "time_s": round(t, 4),
                         "source": monitor, "changed_beams_from_baseline": count,
                         "ape_xy_m": actual.tolist(),
                         "owner_pose_evaluator": _present_owner(model, owner_south)})
            if second_streak >= STREAK_N:
                return ("owner_motion_unresolved", rows, frames,
                        recognition_wall + second_wall, None, t)
            if index % 2 == 0:
                rgb, _, _ = owner.capture(renderer, data, base.CAMERA, False)
                frames.append(_actor_frame(rgb, "monitoring while fresh recognition runs"))
        time_s = second_capture + second_steps * SCAN_S
        recognition_wall += second_wall
    gt_box, pixels = evaluator_box(renderer, data, model, base.CAMERA,
                                    "shown_can_geom")
    frames.append(_actor_frame(captured, "recognized; wait for fresh visual track",
                               chosen[0]["box"] if chosen else None, gt_box))
    rows.append({"stage": "recognize_after_monitor", "time_s": round(time_s, 4),
                 "proposal_count": len(proposals),
                 "top_similarity": float(ranked[0][1]) if ranked else None,
                 "selected_box": chosen[0]["box"] if chosen else None,
                 "selected_iou_evaluator": owner.iou(
                     chosen[0]["box"] if chosen else None, gt_box),
                 "can_pixels_evaluator": pixels})
    if chosen is None:
        return "visible_item_not_reacquired", rows, frames, recognition_wall, None, time_s
    crop = first_image.crop(chosen[0]["box"])
    tracker = owner.FastActor(cv2.cvtColor(np.asarray(crop), cv2.COLOR_RGB2BGR),
                              use_depth=False)
    final_track, point_error = None, None
    for index in range(3):
        _advance_prop(model, time_s)
        mujoco.mj_forward(model, data)
        rgb, depth, capture_wall = owner.capture(renderer, data, base.CAMERA, use_depth)
        begin = time.perf_counter()
        final_track = tracker.observe(Image.fromarray(rgb), None, float(estimate[0]))
        age = capture_wall + time.perf_counter() - begin
        if age > owner.MAX_FRESH_AGE_S:
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
        frames.append(_actor_frame(rgb, f"fresh visual track: {final_track['status']}",
                                   final_track["box"], gt_box))
        time_s += max(.1, age)
        rows.append({"stage": "fresh_owner_view", "time_s": round(time_s, 4),
                     "actor": final_track, "depth_point_xyz_m": point,
                     "depth_point_error_evaluator_m": point_error,
                     "selected_iou_evaluator": owner.iou(final_track["box"], gt_box),
                     "can_pixels_evaluator": pixels})
        if final_track["status"] == "visually_reacquired_unlocalized":
            break
    if final_track is not None and final_track["status"] == "visually_reacquired_unlocalized":
        rows.append({"stage": "owner_confirms_taught_item", "time_s": round(time_s, 4),
                     "owner_reply": "yes_that_is_the_taught_can"})
        outcome = "owner_confirmed_metric_find" if use_depth else "owner_confirmed_visual_find"
    else:
        outcome = "unconfirmed_after_slow_recognition"
    return outcome, rows, frames, recognition_wall, point_error, time_s
