"""Joined two-person route, interruption, and honest new-show recovery.

After identity reset the ape requests a new demonstration, holds pose, checks
mid-height range while slow image inference runs, and uses fresh RGB plus
candidate-local depth. The result is an unattributed new item observation,
never a claim that the original owner has been identified. Human responses
and body motion remain scripted evaluator interventions.
"""

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
import hashlib
import json
import math
from pathlib import Path
import time

import cv2
import mujoco
import numpy as np
from PIL import Image

import analyze_robocasa_post_retreat_person_roi_v588 as roi
import robocasa_first_return_person_reacquire_turn_v576 as turn
import robocasa_owner_person as person
import robocasa_person_interaction as interaction
import run_robocasa_joined_owner_route_v410 as owner
import run_robocasa_layout2_route_v407 as route
import run_robocasa_person_joined_v497 as joined
import run_robocasa_layout2_multitrack_v485 as multi
import run_robocasa_two_moving_people_v586 as moving
from run_robocasa_scan_height_v433 import height_scan
from screen_robocasa_owner_safe_stop_v494 import aim_camera, depth_point


ROOT = Path(__file__).resolve().parents[3]
OUT = ROOT / "artifacts/ape-robocasa-social-reteach-v589"
CAMERA = "owner_up25"
PITCH = 15
SCAN = height_scan((.30,))
CASES = ("owner_reteach", "bystander_reteach", "ignored")


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def bystander_show(model, torso, fraction):
    torso = np.asarray(torso)
    rest = np.array([torso[0]+.10, torso[1]+.16, .43])
    shown = np.array([torso[0]+.15, torso[1]-.30, .55])
    hand = rest + fraction * (shown-rest)
    shoulder = np.array([torso[0], torso[1], 1.05])
    elbow = person.elbow_for(shoulder, hand)
    person._capsule(model, "distractor_upper_arm_geom", shoulder, elbow)
    person._capsule(model, "distractor_forearm_geom", elbow, hand)
    model.geom_pos[model.geom("distractor_hand_geom").id] = hand
    model.body_pos[model.body("replica_can").id] = hand + [0., -.14, .104]
    return hand.tolist()


def infer_item(vision, exemplar, rgb):
    started = time.perf_counter()
    image = Image.fromarray(rgb)
    proposals = vision.detect(image)
    scores = (vision.feature([image.crop(item["box"]) for item in proposals]) @ exemplar
              if proposals else np.empty(0))
    ranked = sorted(({"box_xyxy": item["box"], "similarity": float(score)}
                     for item, score in zip(proposals, scores)),
                    key=lambda row: row["similarity"], reverse=True)
    return {"ranked": ranked, "worker_wall_s": time.perf_counter()-started}


def run_case(case_name, policy, scenario, prior, vision, exemplar):
    original_turn = turn.owner_turn
    original_path = moving.distractor_xy_at
    original_out = moving.OUT
    original_margin = route.MARGIN_M
    original_hand = interaction.SHOW_HAND.copy()
    original_first_m = turn.FIRST_RETURN_OWNER_M
    original_first_s = turn.FIRST_RETURN_OWNER_S
    original_late = turn.LATE_OWNER_APPROACH_M
    original_second = turn.SECOND_RETURN_OWNER_M
    original_owner = turn.OWNER_APPROACH_M
    trial_out = OUT / case_name
    trial_out.mkdir(parents=True, exist_ok=True)

    def recovery(response, model, data, renderer, actual, estimate,
                 time_s, use_depth, vision, exemplar):
        outcome, rows, frames, previous_wall, previous_error, t = original_turn(
            response, model, data, renderer, actual, estimate,
            time_s, use_depth, vision, exemplar)
        if outcome not in ("owner_interaction_reset_after_reacquire_motion",
                           "owner_track_uncertain_before_show"):
            return outcome, rows, frames, previous_wall, previous_error, t
        if outcome == "owner_track_uncertain_before_show":
            grid = multi.CAPTURED[-1]
            known, cells = turn._known_free_south(grid, estimate, .5)
            rows.append({"stage": "new_show_checked_restage_path",
                         "time_s": round(t, 4), "known_free": known,
                         "distance_m": .5, "cells": cells})
            if not known:
                return ("new_show_no_known_free_restage", rows, frames,
                        previous_wall, previous_error, t)
            start_actual = actual.copy()
            start_estimate = estimate[:2].copy()
            angles = np.linspace(-math.pi, math.pi, 72, endpoint=False)
            south_sector = np.abs((angles+math.pi/2+math.pi)%(2*math.pi)-math.pi) <= math.radians(35.)
            for step in range(1, 101):
                t += .02
                fraction = step/100
                actual[:] = start_actual + [0., -.5*fraction]
                estimate[:2] = start_estimate + [0., -.5*fraction]
                model.body_pos[model.body("audit_ape").id, :2] = actual
                turn._prop(model, t)
                mujoco.mj_forward(model, data)
                packet = SCAN(model, data, actual, 5.,
                              np.random.default_rng(25000+step))
                south_returns = [value for value, in_sector in zip(packet, south_sector)
                                 if in_sector and value is not None]
                if south_returns and min(south_returns) < .35:
                    rows.append({"stage": "new_show_restage_blocked",
                                 "time_s": round(t, 4), "ape_xy_m": actual.tolist(),
                                 "south_nearest_range_m": min(south_returns)})
                    return ("new_show_restage_blocked", rows, frames,
                            previous_wall, previous_error, t)
                if step % 5 == 0:
                    aim_camera(model, interaction.CAMERA, estimate[:2],
                               start_estimate+[0., 1.], 20)
                    model.cam_pos[model.camera(interaction.CAMERA).id, :2] = actual
                    mujoco.mj_forward(model, data)
                    retreat_rgb, _, _ = owner.capture(
                        renderer, data, interaction.CAMERA, False)
                    frames.append(owner.owner_frame(
                        retreat_rgb, "restage before asking again", None, None))
                rows.append({"stage": "new_show_restage_motion",
                             "time_s": round(t, 4), "ape_xy_m": actual.tolist(),
                             "known_free_start_path": True})
            rows.append({"stage": "new_show_restage_complete",
                         "time_s": round(t, 4), "ape_xy_m": actual.tolist(),
                         "distance_m": float(np.linalg.norm(actual-start_actual))})
        tracks = multi.CAPTURED[-1].multi_history[-1]["tracks"]
        anonymous = np.asarray(max(tracks, key=lambda item: item["hits"])["center_xy_m"])
        # Simulator response: compliant people stop moving after the request.
        # The actor below must verify the resulting packets and item image.
        fixed_bystander = moving.distractor_xy_at(t)
        if case_name != "ignored":
            moving.distractor_xy_at = lambda _t: fixed_bystander.copy()
        rows.append({"stage": "request_new_show_after_identity_reset",
                     "time_s": round(t, 4), "speech_intent":
                     "I lost track of who was teaching me. Please hold still and show the can again.",
                     "identity_state": "old_owner_unknown_new_teacher_not_yet_seen",
                     "ape_xy_m": actual.tolist()})
        before_t = t + .2
        turn._prop(model, before_t)
        aim_camera(model, CAMERA, estimate[:2], anonymous, PITCH)
        model.cam_pos[model.camera(CAMERA).id, :2] = actual
        mujoco.mj_forward(model, data)
        before_rgb, before_depth, before_age = owner.capture(
            renderer, data, CAMERA, True)
        t = before_t + max(0., before_age-.2)
        frames.append(owner.owner_frame(before_rgb,
                      "lost identity: request a new show", None, None))
        presentation = []
        for index, fraction in enumerate(np.linspace(.2, 1., 5)):
            t += .16
            turn._prop(model, t)
            if case_name == "owner_reteach":
                turn._show_owner(model, float(fraction), .4)
            elif case_name == "bystander_reteach":
                bystander_show(model, fixed_bystander, float(fraction))
            mujoco.mj_forward(model, data)
            rgb, _, age = owner.capture(renderer, data, CAMERA, False)
            frames.append(owner.owner_frame(rgb, "waiting for a fresh show",
                                            None, None))
            packet = SCAN(model, data, actual, 5., np.random.default_rng(26000+index))
            presentation.append({"time_s": round(t, 4),
                                 "mid_height_returns": sum(x is not None for x in packet),
                                 "capture_wall_s": age})
        rows.append({"stage": "new_show_response_observation_window",
                     "time_s": round(t, 4), "samples": presentation,
                     "actor_state": "still_no_owner_identity"})
        shown_rgb, shown_depth, shown_age = owner.capture(
            renderer, data, CAMERA, True)
        t += max(.1, shown_age)
        raw = trial_out / "new-show-rgbd.npz"
        np.savez_compressed(raw, before_rgb=before_rgb,
                            before_depth=before_depth,
                            shown_rgb=shown_rgb, shown_depth=shown_depth)
        image = trial_out / "new-show.png"
        Image.fromarray(shown_rgb).save(image)
        bearing = math.atan2(anonymous[1]-estimate[1],
                             anonymous[0]-estimate[0])
        baseline = SCAN(model, data, actual, 5., np.random.default_rng(27000))
        guard_rows = []
        alarm = None
        streak = 0
        with ThreadPoolExecutor(max_workers=1) as pool:
            start = time.perf_counter()
            future = pool.submit(infer_item, vision, exemplar, shown_rgb)
            next_tick = start
            while True:
                now = time.perf_counter()
                if now < next_tick:
                    time.sleep(next_tick-now)
                    now = time.perf_counter()
                elapsed = now-start
                current_t = t+elapsed
                turn._prop(model, current_t)
                if case_name == "bystander_reteach":
                    bystander_show(model, fixed_bystander, 1.)
                mujoco.mj_forward(model, data)
                packet = SCAN(model, data, actual, 5.,
                              np.random.default_rng(28000+len(guard_rows)))
                changed = turn._sector_change(baseline, packet, bearing, 35.)
                streak = streak+1 if changed >= 3 else 0
                if streak >= 2 and alarm is None:
                    alarm = current_t
                guard_rows.append({"time_s": round(current_t, 4),
                                   "changed_owner_sector_beams": changed,
                                   "motion_streak": streak,
                                   "range_returns": sum(x is not None for x in packet)})
                if len(guard_rows) % 5 == 1:
                    live_rgb, _, _ = owner.capture(renderer, data, CAMERA, False)
                    frames.append(owner.owner_frame(
                        live_rgb, "monitoring while recognizing new show",
                        None, None))
                if future.done() or elapsed > 20.:
                    break
                next_tick += .1
                if next_tick < now:
                    next_tick = now+.1
            inferred = future.result()
        t += elapsed
        rows.append({"stage": "new_show_inference_guard",
                     "time_s": round(t, 4), "scan_count": len(guard_rows),
                     "alarm_time_s": round(alarm, 4) if alarm is not None else None,
                     "worker_wall_s": inferred["worker_wall_s"],
                     "samples": guard_rows})
        if alarm is not None:
            rows.append({"stage": "new_show_stale_image_discarded",
                         "time_s": round(t, 4), "reason": "person_sector_motion"})
            return "new_show_motion_unresolved", rows, frames, inferred["worker_wall_s"], None, t
        ranked = inferred["ranked"]
        chosen = ranked[0] if ranked and ranked[0]["similarity"] >= owner.THRESHOLD else None
        if chosen is None:
            rows.append({"stage": "new_show_not_seen", "time_s": round(t, 4),
                         "proposal_count": len(ranked),
                         "top_similarity": ranked[0]["similarity"] if ranked else None})
            frames.append(owner.owner_frame(shown_rgb,
                          "no new can shown: ask again later", None, None))
            return ("new_show_unconfirmed_unknown", rows, frames,
                    inferred["worker_wall_s"], None, t)
        box = chosen["box_xyxy"]
        x0,y0,x1,y1 = map(int, box)
        region = np.abs(shown_rgb[y0:y1,x0:x1].astype(float) -
                        before_rgb[y0:y1,x0:x1].astype(float))
        change = float(np.mean(region)/255.) if region.size else 0.
        local = roi.roi_support(shown_depth, box)
        rows.append({"stage": "new_show_candidate", "time_s": round(t, 4),
                     "box_xyxy": box, "similarity": chosen["similarity"],
                     "rgb_change_in_box": change,
                     "candidate_depth_valid_fraction": local["valid_fraction"],
                     "candidate_depth_valid_pixels": local["valid_pixels"],
                     "raw_packets": {"path": str(raw), "sha256": sha(raw)},
                     "shown_image": {"path": str(image), "sha256": sha(image)}})
        frames.append(owner.owner_frame(shown_rgb,
                      "new show candidate; old identity remains unknown", box, None))
        if change < .05 or local["valid_fraction"] < .7 or local["valid_pixels"] < 50:
            return ("new_show_evidence_weak", rows, frames,
                    inferred["worker_wall_s"], None, t)
        tracker = owner.FastActor(
            cv2.cvtColor(np.asarray(Image.fromarray(shown_rgb).crop(box)),
                         cv2.COLOR_RGB2BGR), use_depth=False)
        points = []
        # Transient renderer/tracker load can age out an otherwise valid frame.
        # Keep the original freshness gate and allow a bounded retry window.
        for index in range(10):
            t += .12
            turn._prop(model, t)
            if case_name == "bystander_reteach":
                bystander_show(model, fixed_bystander, 1.)
            mujoco.mj_forward(model, data)
            fresh_rgb, fresh_depth, capture_age = owner.capture(
                renderer, data, CAMERA, True)
            start_track = time.perf_counter()
            track = tracker.observe(Image.fromarray(fresh_rgb), None,
                                    float(estimate[0]))
            if case_name == "owner_reteach" and index < 3:
                # Evaluator stress: three transient slow tracking cycles.
                time.sleep(.27)
            age = capture_age + time.perf_counter()-start_track
            fresh_box = track["box"]
            fresh_local = (roi.roi_support(fresh_depth, fresh_box)
                           if fresh_box is not None else None)
            point = (depth_point(fresh_box, fresh_depth, data, model, CAMERA)
                     if fresh_box is not None else None)
            if point is not None:
                point = np.asarray(point)
                point[:2] += estimate[:2]-actual
            accepted = (age <= owner.MAX_FRESH_AGE_S and
                        fresh_box is not None and point is not None and
                        fresh_local["valid_fraction"] >= .7)
            if accepted:
                points.append(np.asarray(point))
            else:
                points.clear()
            rows.append({"stage": "fresh_new_show_check", "index": index,
                         "time_s": round(t, 4), "frame_age_s": age,
                         "tracker_status": track["status"],
                         "actor_box_xyxy": fresh_box,
                         "candidate_depth_valid_fraction": (
                             fresh_local["valid_fraction"] if fresh_local else None),
                         "point_xyz_est_m": point.tolist() if point is not None else None,
                         "accepted": accepted})
            frames.append(owner.owner_frame(fresh_rgb,
                          f"fresh new show {index}: {'candidate' if accepted else 'unknown'}",
                          fresh_box, None))
            if len(points) >= 2 and np.linalg.norm(points[-1]-points[-2]) <= .10:
                estimate_point = np.mean(points[-2:], axis=0)
                rows.append({"stage": "new_show_metric_item_seen",
                             "time_s": round(t, 4),
                             "item_xyz_est_m": estimate_point.tolist(),
                             "teacher_identity": "unattributed_new_session"})
                return ("new_show_metric_item_seen_unattributed", rows, frames,
                        inferred["worker_wall_s"], None, t)
        return ("new_show_unconfirmed_unknown", rows, frames,
                inferred["worker_wall_s"], None, t)

    try:
        moving.OUT = trial_out
        turn.owner_turn = recovery
        route.MARGIN_M = .15
        interaction.SHOW_HAND = np.asarray([1.30, -2.98, .55])
        turn.OWNER_APPROACH_M = 0.
        turn.FIRST_RETURN_OWNER_M = .4
        turn.FIRST_RETURN_OWNER_S = 1.
        turn.LATE_OWNER_APPROACH_M = 0.
        turn.SECOND_RETURN_OWNER_M = 0.
        result = moving.run_trial(scenario, prior, policy, vision, exemplar,
                                  bystander=True)
    finally:
        turn.owner_turn = original_turn
        moving.distractor_xy_at = original_path
        moving.OUT = original_out
        route.MARGIN_M = original_margin
        interaction.SHOW_HAND = original_hand
        turn.OWNER_APPROACH_M = original_owner
        turn.FIRST_RETURN_OWNER_M = original_first_m
        turn.FIRST_RETURN_OWNER_S = original_first_s
        turn.LATE_OWNER_APPROACH_M = original_late
        turn.SECOND_RETURN_OWNER_M = original_second
    return {"scripted_response_evaluator": case_name, "result": result}


def main():
    policy = json.loads(joined.CONFIG.read_text())
    scenario = next(x for x in policy["cases"] if x["id"] == "late_crossing")
    prior = np.load(joined.PRIOR)
    OUT.mkdir(parents=True, exist_ok=True)
    vision = owner.Vision(teach_selector="generic-proposal")
    exemplar = vision.feature([Image.open(owner.TEACH_CROP).convert("RGB")])[0]
    results = []
    for case in CASES:
        result = run_case(case, policy, scenario, prior, vision, exemplar)
        results.append(result)
        ep = result["result"]["episode"]
        print(case, ep["outcome"], ep["video"]["frames"], flush=True)
    report = {"scope": "three full furnished moving-two-person route/interrupt/re-request episodes; actor asks for a new show after owner identity reset, monitors ideal mid-range during released item inference, and can report a fresh metric item without attributing old owner identity; scripted owner/bystander response and kinematic ape",
              "source_sha256": sha(Path(__file__)),
              "base_runner_sha256": sha(Path(moving.__file__)),
              "turn_source_sha256": sha(Path(turn.__file__)),
              "taught_crop_sha256": sha(owner.TEACH_CROP),
              "scene_xml_sha256": sha(moving.SCENE_XML),
              "results": results}
    (OUT / "report.json").write_text(json.dumps(report, indent=2) + "\n")


if __name__ == "__main__":
    main()
