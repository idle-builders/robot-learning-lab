"""V84 fork of the fixed-clock owner runner with bounded restage reobservation.

The starting MuJoCo state is reconstructed from the saved V36 endpoint and
required to reproduce its final 72-ray packet exactly. An anonymous held-side
range watch requests space, waits for three fresh clear packets, then obtains
a new RGB-D owner demonstration while the range watch stays live. Human
responses are scripted evaluator interventions; no person ID or future pose
is given to the actor. The ape remains a kinematic proxy. A blocked meeting
step can now hold, receive a fresh range packet, and retry until the bounded
restage tick limit; the prior runner remains frozen for historical evidence.
"""

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
import hashlib
import json
import math
from pathlib import Path
import sys
import time

import cv2
import imageio.v2 as imageio
import mujoco
import numpy as np
from PIL import Image


ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / "experiments/ape_high_level/closed_loop"))

import audit_owner_inference_response as geometry  # noqa: E402
import owner_inference_response as v35  # noqa: E402
import owner_camera_inference_response as v36  # noqa: E402
import robocasa_first_return_person_reacquire_turn_v576 as turn  # noqa: E402
import robocasa_person_interaction as interaction  # noqa: E402
import run_robocasa_dynamic_suite_v439 as dynamic  # noqa: E402
import run_robocasa_joined_owner_route_v410 as owner  # noqa: E402
import run_robocasa_person_joined_v497 as joined  # noqa: E402
import run_robocasa_social_reteach_v589 as reteach  # noqa: E402
import screen_robocasa_owner_safe_stop_v494 as camera  # noqa: E402
import analyze_robocasa_post_retreat_person_roi_v588 as roi  # noqa: E402


SOURCE = ROOT / "artifacts/ape-native-route-guard/v36-camera-inference-response/report.json"
OUT = ROOT / "artifacts/ape-native-route-guard/v51-full-angle-escape"
CASES = ("complies", "ignores", "side_range_lost", "returns_during_new_show",
         "blind_return_no_contact", "blind_return_contact",
         "blind_return_flank_rgbd", "blind_return_flank_fast_response",
         "blind_return_flank_slow_response")
BLIND_RETURN = ("blind_return_no_contact", "blind_return_contact",
                "blind_return_flank_rgbd", "blind_return_flank_fast_response",
                "blind_return_flank_slow_response")
FLANK_MONITOR = ("blind_return_flank_rgbd", "blind_return_flank_fast_response",
                 "blind_return_flank_slow_response")
FLANK_RESPONSE = {
    "blind_return_flank_fast_response": {"delay_s": .1, "speed_mps": .25},
    "blind_return_flank_slow_response": {"delay_s": .5, "speed_mps": .10},
}
SECOND_RESPONSE_MAX_M = .25
BLIND_RETURN_NEAR_XY = np.array([1.456, -4.640])
HOLD_PERIOD_S = .1
HOLD_HORIZON_S = 4.
HUMAN_DELAY_S = .25
HUMAN_AWAY_MPS = .50
HUMAN_RETURN_MPS = .62
CLEAR_SIDE_RANGE_M = .70
CLEAR_STREAK = 3
SHOW_SETTLE_S = .6
SHOW_MONITOR_PERIOD_S = .1
SHOW_MAX_WALL_S = 20.
FIXED_MONITOR_TICKS = 36  # 0.0 through 3.5 seconds on the simulation clock


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def original():
    report = json.loads(SOURCE.read_text())
    source = next(row for row in report["outcomes"]
                  if row["sensor"] == "left_flank_rgbd_motion"
                  and row["condition"] == "wide_respond_fast")
    return source


def side_healthy(packet):
    angles = np.linspace(-math.pi, math.pi, len(packet), endpoint=False)
    side = [value for value, angle in zip(packet, angles)
            if abs((angle+math.pi)%(2*math.pi)-math.pi) >= math.radians(120)]
    return any(value is not None for value in side)


def lose_side(packet):
    angles = np.linspace(-math.pi, math.pi, len(packet), endpoint=False)
    return [None if abs((angle+math.pi)%(2*math.pi)-math.pi) >= math.radians(120)
            else value for value, angle in zip(packet, angles)]


def geometry_at(summary, episode, scenario, model, data):
    actual = np.asarray(summary["ape_after_episode_xy_m"])
    estimate = np.asarray(episode["final_estimated_xy_m"])
    t0 = summary["range_packets"][-1]["time_s"]
    model.body_pos[model.body("audit_ape").id, :2] = actual
    prior_hand = interaction.SHOW_HAND.copy()
    interaction.SHOW_HAND = np.asarray([1.30, -2.98, .55])
    try:
        turn._show_owner(model, 1., .4)
    finally:
        interaction.SHOW_HAND = prior_hand
    v35.two.set_distractor(model, np.asarray(summary["bystander_target_evaluator_xy_m"]))
    model.body_pos[model.body("route_obstacle").id, :2] = dynamic.motion(scenario)(t0)
    camera.aim_camera(model, reteach.CAMERA, actual, np.asarray([1., -2.8]), 15)
    camera.aim_camera(model, "owner_wide25", actual, actual+[-1., 0.], 15)
    mujoco.mj_forward(model, data)
    packet = reteach.SCAN(model, data, actual, 5.,
                         np.random.default_rng(28000+len(summary["range_packets"])-1))
    saved = summary["range_packets"][-1]["ranges_m"]
    if packet != saved:
        raise RuntimeError("V36 endpoint did not reproduce exactly")
    return actual, estimate, t0, sha(SOURCE), packet


def human_xy(case, summary, t, request_s, inference_start_s):
    target = np.asarray(summary["bystander_target_evaluator_xy_m"])
    away = v35.SIDE_START
    if case == "ignores" or t <= request_s+HUMAN_DELAY_S:
        return target.copy()
    distance = np.linalg.norm(away-target)
    if case == "parked_after_clear" and inference_start_s is not None:
        # Control for camera self-motion: finish the scripted retreat, then
        # freeze the person when the fresh owner inference starts.
        t = min(t, inference_start_s)
    fraction = min(1., HUMAN_AWAY_MPS*(t-request_s-HUMAN_DELAY_S)/distance)
    xy = target+(away-target)*fraction
    if case in ("returns_during_new_show", *BLIND_RETURN) and inference_start_s is not None:
        return_start = inference_start_s+.2
        if t > return_start:
            origin = target+(away-target)*min(
                1., HUMAN_AWAY_MPS*(return_start-request_s-HUMAN_DELAY_S)/distance)
            destination = (target+.55*(BLIND_RETURN_NEAR_XY-target)
                           if case in BLIND_RETURN else target)
            back = min(1., HUMAN_RETURN_MPS*(t-return_start)/
                       np.linalg.norm(destination-origin))
            xy = origin+(destination-origin)*back
    return xy


def observed_packet(model, data, actual, case, tick,
                    *, person_xy=None, hidden_return=False):
    if hidden_return:
        # Sensor-model fault: plausible room/parked-side returns replace the
        # near person's surfaces, without a health flag. The actor receives
        # only the resulting packet, never this evaluator-side intervention.
        v35.two.set_distractor(model, v35.SIDE_START)
        mujoco.mj_forward(model, data)
    packet = reteach.SCAN(model, data, actual, 5.,
                         np.random.default_rng(38000+tick))
    if hidden_return:
        v35.two.set_distractor(model, person_xy)
        mujoco.mj_forward(model, data)
    if case == "side_range_lost":
        packet = lose_side(packet)
    return packet


def south_range_clear(packet):
    angles = np.linspace(-math.pi, math.pi, len(packet), endpoint=False)
    south = [value for value, angle in zip(packet, angles)
             if abs((angle+math.pi/2+math.pi)%(2*math.pi)-math.pi) <=
             math.radians(30.)]
    return bool(south) and all(value is None or value >= .60 for value in south)


def response_displacement(response, response_start_s, time_s, *, path_clear=True):
    if response is None or response_start_s is None or not path_clear:
        return 0.
    return min(SECOND_RESPONSE_MAX_M, max(0., response["speed_mps"] *
               (time_s-response_start_s)))


def actor_ape_at(summary, rows, time_s):
    start = np.asarray(summary["ape_after_episode_xy_m"])
    restage = [(row["time_s"], np.asarray(row["ape_xy_m"]))
               for row in rows if row["stage"] in
               ("pre_show_restage_step", "pre_show_restage_wait")]
    if restage:
        resume = next(row["time_s"] for row in rows
                      if row["stage"] == "post_alarm_range_watch" and
                      row["actor_action"] ==
                      "resume_after_three_fresh_clear_packets")
        if time_s <= resume:
            return start
        for (left_t, left_xy), (right_t, right_xy) in zip(
                [(resume, start), *restage[:-1]], restage):
            if time_s <= right_t:
                fraction = np.clip((time_s-left_t)/(right_t-left_t), 0., 1.)
                return left_xy+fraction*(right_xy-left_xy)
        start = restage[-1][1]
    near_body_plan = next((row for row in rows if row["stage"] ==
                           "near_body_escape_plan"), None)
    if near_body_plan is not None:
        samples = [(row["time_s"], np.asarray(row["ape_xy_m"]))
                   for row in rows if row["stage"] ==
                   "second_inference_range_watch"]
        if not samples or time_s < samples[0][0]:
            return start
        for (left_t, left_xy), (right_t, right_xy) in zip(
                samples, samples[1:]):
            if time_s <= right_t:
                fraction = np.clip((time_s-left_t)/(right_t-left_t), 0., 1.)
                return left_xy+fraction*(right_xy-left_xy)
        return samples[-1][1]
    plan = next((row for row in rows if row["stage"] ==
                 "second_response_plan"), None)
    if plan is None:
        return start
    if plan.get("adaptive_controller"):
        samples = [(row["time_s"], np.asarray(row["ape_xy_m"]))
                   for row in rows if row["stage"] ==
                   "second_inference_range_watch"]
        if not samples or time_s < samples[0][0]:
            return start
        for (left_t, left_xy), (right_t, right_xy) in zip(
                samples, samples[1:]):
            if time_s <= right_t:
                fraction = np.clip((time_s-left_t)/(right_t-left_t), 0., 1.)
                return left_xy+fraction*(right_xy-left_xy)
        return samples[-1][1]
    inference = next((row for row in rows if row["stage"] ==
                      "second_inference_result"), None)
    response_start_s = inference["first_response_s"] if inference else None
    displacement = response_displacement(plan["parameters"],
                                         response_start_s, time_s,
                                         path_clear=plan["south_range_clear"])
    if inference and inference.get("response_halt_displacement_m") is not None:
        displacement = min(displacement,
                           inference["response_halt_displacement_m"])
    return start+[0., -displacement]


def prop_gap(model, data, actual):
    model.body_pos[model.body("audit_ape").id, :2] = actual
    mujoco.mj_forward(model, data)
    return float(mujoco.mj_geomDistance(
        model, data, model.geom("audit_ape_geom").id,
        model.geom("route_obstacle_geom").id, 2., None))


def frame(renderer, model, data, actual, caption, side_range, state,
          gap, item_box=None, prop_gap_m=None):
    mujoco.mj_forward(model, data)
    rgb, _, _ = owner.capture(renderer, data, reteach.CAMERA, False)
    left, _, _ = owner.capture(renderer, data, "owner_wide25", False)
    image = np.full((528, 1040, 3), (16, 22, 25), dtype=np.uint8)
    image[48:, :640] = rgb
    image[48:348, 640:] = cv2.resize(left, (400, 300))
    if item_box is not None:
        x0, y0, x1, y1 = map(int, item_box)
        cv2.rectangle(image, (x0, y0+48), (x1, y1+48), (70, 240, 100), 3)
    cv2.putText(image, caption, (10, 31), cv2.FONT_HERSHEY_SIMPLEX,
                .60, (240, 240, 240), 2)
    lines = [f"actor: {state}",
             f"left range: {side_range:.2f} m" if side_range is not None
             else "left range: unavailable",
             f"evaluator gap: {gap:.2f} m",
             "right image: left-facing RGB view"]
    if prop_gap_m is not None:
        lines.append(f"evaluator prop gap: {prop_gap_m:.2f} m")
    for index, line in enumerate(lines):
        cv2.putText(image, line, (650, 382+index*30),
                    cv2.FONT_HERSHEY_SIMPLEX, .52,
                    (230, 235, 240), 1)
    return image


def capture_show(renderer, model, data, actual, estimate, vision, exemplar,
                 case, summary, request_s, resume_s, frames, rows, side_range,
                 output_root=OUT, route_obstacle_fn=None):
    # A fresh, scripted owner demonstration starts only after observed space.
    prior_hand = interaction.SHOW_HAND.copy()
    interaction.SHOW_HAND = np.asarray([1.30, -2.98, .55])
    try:
        turn._rest_owner(model, .4)
        mujoco.mj_forward(model, data)
        before_rgb, _, _ = owner.capture(renderer, data, reteach.CAMERA, True)
        t = resume_s+SHOW_SETTLE_S
        if route_obstacle_fn is not None:
            model.body_pos[model.body("route_obstacle").id, :2] = \
                route_obstacle_fn(t)
        turn._show_owner(model, 1., .4)
        v35.two.set_distractor(model, human_xy(case, summary, t, request_s, None))
        mujoco.mj_forward(model, data)
        shown_rgb, shown_depth, _ = owner.capture(renderer, data,
                                                   reteach.CAMERA, True)
    finally:
        interaction.SHOW_HAND = prior_hand
    frames.append(frame(renderer, model, data, actual,
                        "fresh owner show after observed clearance", side_range,
                        "recognizing while monitoring", geometry.gap(
                            model, data, actual,
                            human_xy(case, summary, t, request_s, None))))
    raw_path = output_root / case / "fresh-show-rgbd.npz"
    raw_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(raw_path, before_rgb=before_rgb,
                        shown_rgb=shown_rgb, shown_depth=shown_depth)
    rows.append({"stage": "fresh_new_show_capture", "time_s": round(t, 5),
                 "raw_rgbd": {"path": str(raw_path), "sha256": sha(raw_path)},
                 "old_owner_identity": "unknown"})
    return t, before_rgb, shown_rgb, shown_depth


def finish_item(vision, exemplar, inferred, before_rgb, shown_rgb,
                shown_depth, renderer, model, data, actual, estimate, t,
                case, summary, request_s, frames, rows, side_range):
    ranked = inferred["ranked"]
    chosen = ranked[0] if ranked and ranked[0]["similarity"] >= owner.THRESHOLD else None
    if chosen is None:
        return "new_show_unconfirmed_unknown", t, None
    box = chosen["box_xyxy"]
    x0, y0, x1, y1 = map(int, box)
    patch = np.abs(shown_rgb[y0:y1,x0:x1].astype(float)-
                   before_rgb[y0:y1,x0:x1].astype(float))
    change = float(np.mean(patch)/255.) if patch.size else 0.
    local = roi.roi_support(shown_depth, box)
    rows.append({"stage": "fresh_item_candidate", "time_s": round(t, 5),
                 "box_xyxy": box, "similarity": chosen["similarity"],
                 "rgb_change_in_box": change,
                 "depth_valid_fraction": local["valid_fraction"],
                 "depth_valid_pixels": local["valid_pixels"]})
    if change < .05 or local["valid_fraction"] < .7 or local["valid_pixels"] < 50:
        return "new_show_evidence_weak", t, box
    crop = Image.fromarray(shown_rgb).crop(box)
    tracker = owner.FastActor(cv2.cvtColor(np.asarray(crop),
                                            cv2.COLOR_RGB2BGR), use_depth=False)
    points = []
    for index in range(2):
        t += .12
        v35.two.set_distractor(model, human_xy(case, summary, t, request_s, None))
        mujoco.mj_forward(model, data)
        fresh_rgb, fresh_depth, age = owner.capture(
            renderer, data, reteach.CAMERA, True)
        begin = time.perf_counter()
        track = tracker.observe(Image.fromarray(fresh_rgb), None,
                                float(estimate[0]))
        age += time.perf_counter()-begin
        fresh_box = track["box"]
        fresh_local = roi.roi_support(fresh_depth, fresh_box) if fresh_box else None
        point = (camera.depth_point(fresh_box, fresh_depth, data, model,
                                    reteach.CAMERA) if fresh_box else None)
        if point is not None:
            point = np.asarray(point)
            point[:2] += estimate[:2]-actual
        accepted = (age <= owner.MAX_FRESH_AGE_S and fresh_box is not None
                    and point is not None and fresh_local["valid_fraction"] >= .7)
        rows.append({"stage": "fresh_new_show_check", "index": index,
                     "time_s": round(t, 5), "age_s": age,
                     "actor_box_xyxy": fresh_box,
                     "depth_valid_fraction": fresh_local["valid_fraction"]
                     if fresh_local else None,
                     "point_xyz_est_m": point.tolist() if point is not None else None,
                     "accepted": accepted})
        frames.append(frame(renderer, model, data, actual,
                            f"fresh item check {index+1}/2", side_range,
                            "candidate" if accepted else "unknown",
                            geometry.gap(model, data, actual,
                                         human_xy(case, summary, t, request_s, None)),
                            fresh_box))
        if accepted:
            points.append(point)
        else:
            points.clear()
    if len(points) == 2 and np.linalg.norm(points[0]-points[1]) <= .10:
        result = np.mean(points, axis=0)
        rows.append({"stage": "new_show_metric_item_seen",
                     "time_s": round(t, 5),
                     "item_xyz_est_m": result.tolist(),
                     "teacher_identity": "unattributed_new_session",
                     "item_position_error_evaluator_m": float(np.linalg.norm(
                         result-model.body_pos[model.body("shown_can").id]))})
        return "new_show_metric_item_seen_unattributed", t, box
    return "new_show_unconfirmed_unknown", t, box


def run_case(case, source, policy, scenario, vision, exemplar, *,
             flank_probe_factory=None, flank_response=None,
             require_camera_pose=False, freeze_route_obstacle_after_show=False,
             route_obstacle_fn=None, response_range_guard_m=None,
             response_controller=None, pre_show_restage=None,
             owner_fovy_deg=None, range_transform=None,
             near_body_controller=None, task_risk_monitor=None,
             restage_wait_on_block=False,
             output_root=OUT):
    summary = source["summary"]
    model = mujoco.MjModel.from_xml_path(str(v35.moving.SCENE_XML))
    if owner_fovy_deg is not None:
        model.cam_fovy[model.camera(reteach.CAMERA).id] = owner_fovy_deg
    data = mujoco.MjData(model)
    actual, estimate, t0, source_hash, endpoint_packet = geometry_at(
        summary, source["result"]["result"]["episode"], scenario, model, data)
    renderer = mujoco.Renderer(model, width=640, height=480)
    rows, frames = [], []
    response_start_actual = actual.copy()
    response_start_estimate = estimate.copy()
    request_s = t0
    resume_s = None
    inference_start_s = None
    clear_streak = 0
    last_t = t0
    last_side = v35.side_range(endpoint_packet)
    last_packet = endpoint_packet
    try:
        rows.append({"stage": "suspend_stale_show_and_request_space",
                     "time_s": t0, "speech_intent":
                     "Please give me a little space so I can see your show again.",
                     "old_owner_identity": "unknown",
                     "checkpoint_range_packet": endpoint_packet})
        for tick in range(round(HOLD_HORIZON_S/HOLD_PERIOD_S)+1):
            t = t0+tick*HOLD_PERIOD_S
            person = human_xy(case, summary, t, request_s, None)
            v35.two.set_distractor(model, person)
            model.body_pos[model.body("route_obstacle").id, :2] = (
                route_obstacle_fn(t) if route_obstacle_fn is not None
                else dynamic.motion(scenario)(t))
            mujoco.mj_forward(model, data)
            packet = observed_packet(model, data, actual, case, tick)
            if range_transform is not None:
                packet = range_transform(packet, t, "post_alarm_range_watch")
            healthy = side_healthy(packet)
            nearest = v35.side_range(packet) if healthy else None
            clear = bool(healthy and nearest is not None and
                         nearest >= CLEAR_SIDE_RANGE_M)
            clear_streak = clear_streak+1 if clear else 0
            action = "hold_and_observe"
            if not healthy:
                action = "hold_and_request_range_recovery"
            if clear_streak >= CLEAR_STREAK:
                resume_s = t
                action = "resume_after_three_fresh_clear_packets"
            gap = geometry.gap(model, data, actual, person)
            rows.append({"stage": "post_alarm_range_watch", "time_s": round(t, 5),
                         "actor_packet": packet, "side_health": healthy,
                         "nearest_side_range_m": nearest,
                         "clear_streak": clear_streak, "actor_action": action,
                         "evaluator_person_gap_m": round(gap, 5)})
            if tick % 2 == 0 or resume_s is not None:
                frames.append(frame(renderer, model, data, actual,
                                    f"post-alarm watch +{t-t0:.1f}s",
                                    nearest, action, gap))
            last_t = t
            last_side = nearest
            last_packet = packet
            if resume_s is not None:
                break
        if resume_s is None:
            outcome = ("sensor_recovery_requested" if case == "side_range_lost"
                       else "waiting_for_person_space")
            return package(case, source, source_hash, summary, t0, last_t,
                           request_s, resume_s, inference_start_s,
                           outcome, rows, frames, model, data, actual,
                           renderer, output_root=output_root,
                           route_obstacle_fn=route_obstacle_fn)
        show_ready_s = resume_s
        if pre_show_restage is not None:
            pre_show_restage.start(actual.copy(), last_packet, resume_s)
            for tick in range(pre_show_restage.max_ticks):
                next_t = resume_s+(tick+1)*HOLD_PERIOD_S
                direction, step_m, decision = pre_show_restage.step(
                    next_t, actual.copy(), last_packet)
                if step_m <= 0.:
                    if (restage_wait_on_block and
                            decision.get("action") == "hold_step_blocked"):
                        person = human_xy(case, summary, next_t,
                                          request_s, None)
                        v35.two.set_distractor(model, person)
                        model.body_pos[model.body("route_obstacle").id, :2] = (
                            route_obstacle_fn(next_t)
                            if route_obstacle_fn is not None else
                            dynamic.motion(scenario)(next_t))
                        mujoco.mj_forward(model, data)
                        last_packet = observed_packet(model, data, actual,
                                                      case, 500+tick)
                        if range_transform is not None:
                            last_packet = range_transform(
                                last_packet, next_t,
                                "pre_show_restage_wait")
                        last_side = (v35.side_range(last_packet)
                                     if side_healthy(last_packet) else None)
                        show_ready_s = next_t
                        last_t = next_t
                        person_gap = geometry.gap(model, data,
                                                  actual, person)
                        rows.append({
                            "stage": "pre_show_restage_wait",
                            "time_s": round(next_t, 5),
                            "actor_packet": last_packet,
                            "actor_decision": decision,
                            "ape_xy_m": actual.tolist(),
                            "evaluator_person_gap_m": round(person_gap, 5),
                            "evaluator_prop_gap_m": (
                                round(prop_gap(model, data, actual), 5)
                                if route_obstacle_fn is not None else None)})
                        frames.append(frame(
                            renderer, model, data, actual,
                            f"meeting route reobserve +{next_t-resume_s:.1f}s",
                            last_side, decision["action"], person_gap,
                            prop_gap_m=(prop_gap(model, data, actual)
                                        if route_obstacle_fn is not None
                                        else None)))
                        continue
                    rows.append({"stage": "pre_show_restage_decision",
                                 "time_s": round(show_ready_s, 5),
                                 "actor_decision": decision,
                                 "ape_xy_m": actual.tolist()})
                    break
                actual[:] += np.asarray(direction)*step_m
                estimate[:2] += np.asarray(direction)*step_m
                model.body_pos[model.body("audit_ape").id, :2] = actual
                model.cam_pos[model.camera(reteach.CAMERA).id, :2] = actual
                person = human_xy(case, summary, next_t, request_s, None)
                v35.two.set_distractor(model, person)
                model.body_pos[model.body("route_obstacle").id, :2] = (
                    route_obstacle_fn(next_t) if route_obstacle_fn is not None
                    else dynamic.motion(scenario)(next_t))
                mujoco.mj_forward(model, data)
                last_packet = observed_packet(model, data, actual, case,
                                              500+tick)
                if range_transform is not None:
                    last_packet = range_transform(
                        last_packet, next_t, "pre_show_restage_step")
                last_side = (v35.side_range(last_packet)
                             if side_healthy(last_packet) else None)
                show_ready_s = next_t
                last_t = next_t
                person_gap = geometry.gap(model, data, actual, person)
                rows.append({"stage": "pre_show_restage_step",
                             "time_s": round(next_t, 5),
                             "actor_packet": last_packet,
                             "actor_decision": decision,
                             "commanded_step_m": step_m,
                             "ape_xy_m": actual.tolist(),
                             "evaluator_person_gap_m": round(person_gap, 5),
                             "evaluator_prop_gap_m": (
                                 round(prop_gap(model, data, actual), 5)
                                 if route_obstacle_fn is not None else None)})
                frames.append(frame(
                    renderer, model, data, actual,
                    f"range-checked meeting pose +{next_t-resume_s:.1f}s",
                    last_side, decision["action"], person_gap,
                    prop_gap_m=(prop_gap(model, data, actual)
                                if route_obstacle_fn is not None else None)))
            else:
                rows.append({"stage": "pre_show_restage_decision",
                             "time_s": round(show_ready_s, 5),
                             "actor_decision": {"action": "tick_limit"},
                             "ape_xy_m": actual.tolist()})
            if not pre_show_restage.ready(actual.copy()):
                return package(case, source, source_hash, summary, t0,
                               last_t, request_s, resume_s, None,
                               "no_supported_meeting_pose", rows, frames,
                               model, data, actual, renderer,
                               output_root=output_root,
                               route_obstacle_fn=route_obstacle_fn)
            camera.aim_camera(model, reteach.CAMERA, actual,
                              np.asarray([1., -2.8]), 15)
            camera.aim_camera(model, "owner_wide25", actual,
                              actual+[-1., 0.], 15)
            mujoco.mj_forward(model, data)
            response_start_actual = actual.copy()
            response_start_estimate = estimate.copy()
        t, before_rgb, shown_rgb, shown_depth = capture_show(
            renderer, model, data, actual, estimate, vision, exemplar,
            case, summary, request_s, show_ready_s, frames, rows, last_side,
            output_root=output_root,
            route_obstacle_fn=route_obstacle_fn)
        inference_start_s = t
        if route_obstacle_fn is not None:
            model.body_pos[model.body("route_obstacle").id, :2] = \
                route_obstacle_fn(inference_start_s)
            mujoco.mj_forward(model, data)
        if freeze_route_obstacle_after_show:
            model.body_pos[model.body("route_obstacle").id, :2] = \
                dynamic.motion(scenario)(inference_start_s)
            mujoco.mj_forward(model, data)
        flank_probe = None
        if case in FLANK_MONITOR or flank_probe_factory is not None:
            flank_video = output_root / case / "flank-rgbd-watch.mp4"
            flank_probe = (flank_probe_factory(flank_video)
                           if flank_probe_factory is not None else
                           v36.MotionCamera("left_flank_rgbd_motion", flank_video))
            flank_probe.start(model, data, actual, t)
        baseline = observed_packet(model, data, actual, case, 900)
        if range_transform is not None:
            baseline = range_transform(baseline, t,
                                       "second_inference_baseline")
        if near_body_controller is not None:
            near_body_controller.start(actual.copy(), t, baseline)
        risk_streak = 0
        camera_streak = 0
        alarm_s = None
        camera_alarm_s = None
        near_body_alarm_s = None
        response_plan = None
        first_response_s = None
        response_distance_m = 0.
        response_step_info = None
        response_halt_s = None
        response_halt_displacement_m = None
        response_halt_range_m = None
        previous_packet = None
        previous_packet_pose = None
        previous_packet_time_s = None
        loss_s = None
        pose_loss_s = None
        contact_s = None
        with ThreadPoolExecutor(max_workers=1) as pool:
            future = pool.submit(reteach.infer_item, vision, exemplar, shown_rgb)
            for tick in range(FIXED_MONITOR_TICKS):
                elapsed = tick*SHOW_MONITOR_PERIOD_S
                current_t = inference_start_s+elapsed
                near_body_command = None
                if response_plan is not None:
                    if (response_range_guard_m is not None and
                            response_halt_s is None and
                            previous_packet is not None):
                        observed = [value for value in previous_packet
                                    if value is not None]
                        nearest = min(observed, default=None)
                        if nearest is None or nearest < response_range_guard_m:
                            response_halt_s = current_t
                            response_halt_displacement_m = float(
                                response_start_actual[1]-actual[1])
                            response_halt_range_m = nearest
                    if (first_response_s is None and
                            response_plan["south_range_clear"] and
                            current_t >= camera_alarm_s+
                            response_plan["parameters"]["delay_s"]):
                        first_response_s = current_t
                    amount = response_displacement(
                        response_plan["parameters"], first_response_s, current_t,
                        path_clear=response_plan["south_range_clear"])
                    if response_halt_displacement_m is not None:
                        amount = min(amount, response_halt_displacement_m)
                    if response_controller is None:
                        actual[:] = response_start_actual+[0., -amount]
                        estimate[:2] = response_start_estimate[:2]+[0., -amount]
                    else:
                        step = max(0., amount-response_distance_m)
                        direction, response_step_info = response_controller.step(
                            current_t, actual.copy(), previous_packet,
                            previous_packet_pose, previous_packet_time_s, step)
                        actual[:] += np.asarray(direction)*step
                        estimate[:2] += np.asarray(direction)*step
                        response_distance_m += step
                    model.body_pos[model.body("audit_ape").id, :2] = actual
                    model.cam_pos[model.camera(reteach.CAMERA).id, :2] = actual
                elif near_body_controller is not None:
                    direction, distance, near_body_command = \
                        near_body_controller.command(current_t, actual.copy())
                    if distance > 0.:
                        actual[:] += np.asarray(direction)*distance
                        estimate[:2] += np.asarray(direction)*distance
                        model.body_pos[model.body("audit_ape").id, :2] = actual
                        model.cam_pos[model.camera(reteach.CAMERA).id, :2] = actual
                person = human_xy(case, summary, current_t,
                                  request_s, inference_start_s)
                v35.two.set_distractor(model, person)
                route_time_s = (inference_start_s if freeze_route_obstacle_after_show
                                else current_t)
                model.body_pos[model.body("route_obstacle").id, :2] = (
                    route_obstacle_fn(current_t) if route_obstacle_fn is not None
                    else dynamic.motion(scenario)(route_time_s))
                mujoco.mj_forward(model, data)
                packet = observed_packet(
                    model, data, actual, case, 1000+tick,
                    person_xy=person, hidden_return=case in BLIND_RETURN)
                if range_transform is not None:
                    packet = range_transform(
                        packet, current_t, "second_inference_range_watch")
                previous_packet = packet
                previous_packet_pose = actual.copy()
                previous_packet_time_s = current_t
                near_body_observation = (near_body_controller.observe(
                    current_t, actual.copy(), packet)
                    if near_body_controller is not None else None)
                if (near_body_observation is not None and
                        near_body_observation["alarm"] and
                        near_body_alarm_s is None):
                    near_body_alarm_s = current_t
                    rows.append({"stage": "near_body_escape_plan",
                                 "time_s": round(current_t, 5),
                                 "actor_decision": near_body_observation})
                healthy = side_healthy(packet)
                nearest = v35.side_range(packet) if healthy else None
                task_risk_decision = (task_risk_monitor(
                    current_t, actual.copy(), packet, near_body_observation)
                    if task_risk_monitor is not None else
                    {"risk": bool(healthy and nearest is not None and
                                  nearest < CLEAR_SIDE_RANGE_M),
                     "basis": "scalar_side_range"})
                risk = bool(task_risk_decision["risk"])
                risk_streak = risk_streak+1 if risk else 0
                if not healthy and loss_s is None:
                    loss_s = current_t
                if risk_streak >= 2 and alarm_s is None:
                    alarm_s = current_t
                flank_sample = (flank_probe.observe(model, data, actual, current_t)
                                if flank_probe is not None else None)
                if (require_camera_pose and flank_sample is not None and
                        not flank_sample.get("actor_pose_healthy", True) and
                        pose_loss_s is None):
                    pose_loss_s = current_t
                camera_streak = (camera_streak+1 if flank_sample and
                                 flank_sample["actor_motion_cue"] else 0)
                if camera_streak >= 2 and camera_alarm_s is None:
                    camera_alarm_s = current_t
                    response = (flank_response if flank_response is not None
                                else FLANK_RESPONSE.get(case))
                    if response is not None:
                        response_plan = {"stage": "second_response_plan",
                                         "alarm_s": current_t,
                                         "south_range_clear": south_range_clear(packet),
                                         "parameters": response,
                                         "requested": "move_south_if_clear",
                                         "adaptive_controller":
                                             response_controller is not None}
                        if response_controller is not None:
                            response_controller.on_alarm(
                                current_t, actual.copy(), flank_sample, packet)
                        rows.append(response_plan)
                gap = geometry.gap(model, data, actual, person)
                contact_packet = gap <= 0 if case == "blind_return_contact" else None
                if contact_packet and contact_s is None:
                    contact_s = current_t
                action = ("contact_detected_abort_after_overlap" if contact_s
                          else "execute_full_angle_escape" if
                          near_body_command is not None and
                          near_body_command["commanded_step_m"] > 0.
                          else "full_angle_alarm_hold" if near_body_alarm_s
                          else "hold_on_new_near_body_range" if response_halt_s
                          else "execute_scanned_south_response" if response_plan
                          and response_plan["south_range_clear"] and
                          response_displacement(response_plan["parameters"],
                                                first_response_s, current_t) > 0
                          else "flank_motion_discard_and_request_space" if camera_alarm_s
                          else "discard_stale_image_and_request_space_again" if alarm_s
                          else "discard_image_on_sensor_loss" if loss_s
                          else "discard_image_on_pose_loss" if pose_loss_s
                          else "keep_observing_during_item_inference")
                rows.append({"stage": "second_inference_range_watch",
                             "time_s": round(current_t, 5),
                             "sensor_time_s_raw": current_t,
                             "actor_packet": packet,
                             "side_health": healthy,
                             "nearest_side_range_m": nearest,
                             "task_risk_decision": task_risk_decision,
                             "risk_streak": risk_streak,
                             "flank_rgbd_packet": flank_sample,
                             "camera_motion_streak": camera_streak,
                             "ape_xy_m": actual.tolist(),
                             "response_halt_s": response_halt_s,
                             "response_halt_range_m": response_halt_range_m,
                             "response_step_info": response_step_info,
                             "near_body_observation": near_body_observation,
                             "near_body_command": near_body_command,
                             "evaluator_route_obstacle_xy_m": model.body_pos[
                                 model.body("route_obstacle").id, :2].tolist(),
                             "evaluator_prop_gap_m": (
                                 round(prop_gap(model, data, actual), 5)
                                 if route_obstacle_fn is not None else None),
                             "actor_contact_packet": contact_packet,
                             "actor_action": action,
                             "evaluator_person_gap_m": round(gap, 5)})
                if tick % 2 == 0:
                    frames.append(frame(renderer, model, data, actual,
                                        f"fresh item inference +{elapsed:.1f}s",
                                        nearest, action, gap,
                                        prop_gap_m=(prop_gap(model, data, actual)
                                                    if route_obstacle_fn is not None
                                                    else None)))
                last_t = current_t
                last_side = nearest
            inferred = future.result()
        if flank_probe is not None:
            flank_probe.close()
            rows.append({"stage": "flank_sensor_video",
                         "path": str(flank_probe.video_path),
                         "sha256": sha(flank_probe.video_path),
                         "frames": len(flank_probe.frames)})
        rows.append({"stage": "second_inference_result",
                     "time_s": round(last_t, 5),
                     "worker_wall_s": inferred["worker_wall_s"],
                     "alarm_s": alarm_s, "camera_alarm_s": camera_alarm_s,
                     "near_body_alarm_s": near_body_alarm_s,
                     "loss_s": loss_s,
                     "pose_loss_s": pose_loss_s,
                     "contact_s": contact_s,
                     "first_response_s": first_response_s,
                     "response_halt_s": response_halt_s,
                     "response_halt_displacement_m": response_halt_displacement_m,
                     "response_halt_range_m": response_halt_range_m})
        if contact_s is not None:
            outcome = "contact_detected_after_overlap"
        elif near_body_alarm_s is not None:
            outcome = "near_body_escape_owner_turn_unresolved"
        elif camera_alarm_s is not None:
            outcome = "new_show_flank_motion_unresolved"
        elif alarm_s is not None:
            outcome = "new_show_motion_unresolved_again"
        elif loss_s is not None:
            outcome = "new_show_sensor_uncertain"
        elif pose_loss_s is not None:
            outcome = "new_show_pose_uncertain"
        else:
            outcome, last_t, box = finish_item(
                vision, exemplar, inferred, before_rgb, shown_rgb,
                shown_depth, renderer, model, data, actual, estimate,
                last_t, case, summary, request_s, frames, rows, last_side)
            if box is not None:
                gap = geometry.gap(model, data, actual,
                                   human_xy(case, summary, last_t,
                                            request_s, None))
                frames.append(frame(renderer, model, data, actual,
                                    "fresh item result", last_side,
                                    outcome, gap, box))
        return package(case, source, source_hash, summary, t0, last_t,
                       request_s, resume_s, inference_start_s,
                       outcome, rows, frames, model, data, actual,
                       renderer, output_root=output_root,
                       route_obstacle_fn=route_obstacle_fn)
    finally:
        renderer.close()


def package(case, source, source_hash, summary, start_s, end_s,
            request_s, resume_s, inference_start_s, outcome, rows, frames,
            model, data, actual, renderer, *, output_root=OUT,
            route_obstacle_fn=None):
    final_person = human_xy(case, summary, end_s, request_s, inference_start_s)
    v35.two.set_distractor(model, final_person)
    if route_obstacle_fn is not None:
        model.body_pos[model.body("route_obstacle").id, :2] = \
            route_obstacle_fn(end_s)
    last_watch = next((row for row in reversed(rows)
                       if "nearest_side_range_m" in row), None)
    final_nearest = (last_watch["nearest_side_range_m"]
                     if last_watch is not None else None)
    final_frame = frame(renderer, model, data, actual,
                        f"OUTCOME: {outcome}", final_nearest,
                        outcome, geometry.gap(model, data, actual, final_person),
                        prop_gap_m=(prop_gap(model, data, actual)
                                    if route_obstacle_fn is not None else None))
    frames.extend([final_frame]*5)
    # Score motion from the first command actually issued at a sensor tick.
    # A nominal alarm+delay start would invent travel between missed ticks.
    scored = []
    for t in np.arange(start_s, end_s+.00001, .02):
        person = human_xy(case, summary, float(t), request_s, inference_start_s)
        ape = actor_ape_at(summary, rows, float(t))
        if route_obstacle_fn is not None:
            model.body_pos[model.body("route_obstacle").id, :2] = \
                route_obstacle_fn(float(t))
        person_gap = geometry.gap(model, data, ape, person)
        scored.append({"time_s": round(float(t), 5),
                       "gap_m": round(person_gap, 5),
                       "prop_gap_m": (round(prop_gap(model, data, ape), 5)
                                      if route_obstacle_fn is not None else None),
                       "ape_xy_m": ape.tolist()})
    case_out = output_root / case
    case_out.mkdir(parents=True, exist_ok=True)
    source_video = source["result"]["result"]["episode"]["video"]["path"]
    reader = imageio.get_reader(source_video)
    source_frames = [frame for frame in reader]
    reader.close()
    assert all(frame.shape == source_frames[0].shape for frame in frames)
    video = case_out / "route-through-post-alarm-outcome.mp4"
    with imageio.get_writer(video, fps=5, codec="libx264",
                            macro_block_size=None) as writer:
        for video_frame in source_frames+frames:
            writer.append_data(video_frame)
    trace = case_out / "trace.json"
    trace.write_text(json.dumps(rows, indent=2)+"\n")
    return {"case": case, "outcome": outcome,
            "source_v36_report_sha256": source_hash,
            "checkpoint_time_s": start_s,
            "request_time_s": request_s,
            "resume_time_s": resume_s,
            "second_inference_start_s": inference_start_s,
            "last_time_s": end_s,
            "checkpoint_range_packet_exact": True,
            "evaluator_motion_model": "speed_law_from_first_issued_response_tick",
            "minimum_proxy_gap_m": min(row["gap_m"] for row in scored),
            "minimum_prop_gap_m": (min(row["prop_gap_m"] for row in scored)
                                   if route_obstacle_fn is not None else None),
            "whole_episode_minimum_proxy_gap_m": min(
                summary["inference_geometry_evaluator"]["minimum_gap_m"],
                min(row["gap_m"] for row in scored)),
            "first_below_5cm_s": next((row["time_s"] for row in scored
                                       if row["gap_m"] < .05), None),
            "evaluator_gap_samples": scored,
            "trace": {"path": str(trace), "sha256": sha(trace)},
            "video": {"path": str(video), "sha256": sha(video),
                      "frames": len(source_frames)+len(frames)},
            "source_video_frames": len(source_frames)}


def main():
    source = original()
    policy = json.loads(joined.CONFIG.read_text())
    scenario = next(x for x in policy["cases"] if x["id"] == "late_crossing")
    vision = owner.Vision(teach_selector="generic-proposal")
    exemplar = vision.feature([Image.open(owner.TEACH_CROP).convert("RGB")])[0]
    results = []
    for case in CASES:
        result = run_case(case, source, policy, scenario, vision, exemplar)
        results.append(result)
        print(case, result["outcome"], result["resume_time_s"],
              result["minimum_proxy_gap_m"], flush=True)
    report = {"schema": "robot-learning-lab.ape-owner-post-alarm-recovery.v1",
              "task": "T-9v3p3n",
              "scope": "Exact saved V36 endpoint sensor checkpoint continues through range-grounded space request, scripted human comply/ignore/return or declared side-range loss, fresh owner re-show, concurrent item inference and second near-body watch. Matched blind-return cases replace near-person range surfaces with plausible distant returns; one exposes an ideal contact bit only after overlap, another monitors independent left-facing RGB-D motion, and two issue scanned-south fast/slow kinematic responses from that camera alarm. Actor sees range/RGB-D/ideal pose and optional contact bit, not evaluator person label or path. No selected body or physical response skill.",
              "source_sha256": sha(Path(__file__)),
              "v36_report_sha256": sha(SOURCE),
              "scene_sha256": sha(v35.moving.SCENE_XML),
              "current_design_pointer_sha256": sha(ROOT / "robot/ape-current-design.json"),
              "thresholds": {"clear_side_range_m": CLEAR_SIDE_RANGE_M,
                             "clear_packets": CLEAR_STREAK,
                             "hold_period_s": HOLD_PERIOD_S},
              "human_evaluator": {"delay_s": HUMAN_DELAY_S,
                                  "away_mps": HUMAN_AWAY_MPS,
                                  "return_mps": HUMAN_RETURN_MPS},
              "results": results}
    (OUT / "report.json").write_text(json.dumps(report, indent=2)+"\n")


if __name__ == "__main__":
    main()
