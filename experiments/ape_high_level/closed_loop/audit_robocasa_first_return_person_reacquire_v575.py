"""Independent 20 ms geometry/video audit for image-grounded owner recovery.

Rebuild the seeded MuJoCo room, replay route, standoff and monitor poses,
interpolate the five scripted presentation poses as an additional swept-arm
screen, and continue the prop through the full post-turn hold. Evaluator
geometry never enters the actor's packet-based interruption rule.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import imageio.v2 as imageio
import mujoco
import numpy as np

import robocasa_owner_person as person
import robocasa_person_interaction as interaction
import robocasa_first_return_person_reacquire_turn_v573 as turn
import run_robocasa_person_joined_v497 as joined
from audit_robocasa_layout2_routes_v409 import clearance, sha256
from audit_robocasa_person_joined_v498 import nearest_can_room
from run_robocasa_dynamic_suite_v439 import motion
from screen_robocasa_owner_person_v489 import gap_to_room


ROOT = Path(__file__).resolve().parents[3]
TRIAL = ROOT / "artifacts/ape-robocasa-first-return-concurrent-reacquire-v574/report.json"
OUT = ROOT / "artifacts/ape-robocasa-first-return-concurrent-reacquire-v574/audit-v575.json"
DT = .02


def audit_episode(ep):
    case = ep["scenario"]
    model = joined.scene_with_owner_and_prop(case["from_xy_m"])
    fingerprint = hashlib.sha256(b"".join(array.tobytes() for array in
        (model.body_pos, model.geom_pos, model.geom_quat,
         model.geom_size, model.geom_rgba, model.light_diffuse))).hexdigest()
    data = mujoco.MjData(model)
    prop_script = motion(case)
    ape_body = model.body("audit_ape").id
    ape_geom = model.geom("audit_ape_geom").id
    can_geom = model.geom("shown_can_geom").id
    prop_geom = model.geom("route_obstacle_geom").id
    people = {name: model.geom(name).id for name in person.PERSON_GEOMS}
    minima = {key: {"gap_m_capped_at_10cm": .1, "geom": None,
                    "time_s": None, "phase": None}
              for key in ("ape_any", "ape_person", "ape_can", "ape_prop")}
    room_min = {"gap_m_capped_at_30cm": .3, "time_s": None, "pair": None}
    can_room_min = {"gap_m_capped_at_10cm": .1, "time_s": None, "geom": None}
    samples = 0

    def owner_pose(pose):
        if pose is None:
            person.set_owner_pose(model, interaction.TORSO_XY,
                                  interaction.REST_HAND, show_can=False)
        else:
            hand = pose.get("hand_xyz_m", interaction.SHOW_HAND)
            torso = pose.get("torso_xy_m", interaction.TORSO_XY)
            showing = pose.get("show_can", True)
            person.set_owner_pose(model, torso, hand, show_can=showing)
            if showing:
                model.body_pos[model.body("shown_can").id] = pose["can_xyz_m"]

    def update_min(key, gap, geom, t, phase):
        if gap < minima[key]["gap_m_capped_at_10cm"]:
            minima[key] = {"gap_m_capped_at_10cm": gap,
                           "geom": geom, "time_s": round(t, 4), "phase": phase}

    def sample(t, xy, pose, phase, room_check=False):
        nonlocal samples, room_min, can_room_min
        model.body_pos[model.body("route_obstacle").id, :2] = prop_script(t)
        owner_pose(pose)
        gap, near = clearance(model, data, ape_body, ape_geom, np.asarray(xy))
        update_min("ape_any", gap, near, t, phase)
        for name, geom in people.items():
            d = float(mujoco.mj_geomDistance(model, data, ape_geom, geom, .1, None))
            update_min("ape_person", d, name, t, phase)
        d = float(mujoco.mj_geomDistance(model, data, ape_geom, can_geom, .1, None))
        update_min("ape_can", d, "shown_can_geom", t, phase)
        d = float(mujoco.mj_geomDistance(model, data, ape_geom, prop_geom, .1, None))
        update_min("ape_prop", d, "route_obstacle_geom", t, phase)
        if room_check:
            room, _ = gap_to_room(model, data)
            if room["gap_m_capped_at_30cm"] < room_min["gap_m_capped_at_30cm"]:
                room_min = {**room, "time_s": round(t, 4)}
            if pose is not None:
                can_room = nearest_can_room(model, data)
                if can_room["gap_m_capped_at_10cm"] < can_room_min["gap_m_capped_at_10cm"]:
                    can_room_min = {**can_room, "time_s": round(t, 4)}
        samples += 1

    route = ep["route_episode"]
    for row in route["trace"]:
        start = np.asarray(row["actual_xy_m"])
        sample(float(row["time_s"]), start, None, "route_capture", True)
        requested = row.get("requested_xy_m")
        if requested is None:
            continue
        delta = ((np.asarray(requested) - np.asarray(row["estimated_xy_m"]))
                 * (1. + .30 / 5.89))
        for index, sub in enumerate(row["motion_substeps"]):
            fraction = (index if sub["stopped_before_move"] else index + 1) / 20
            sample(float(sub["time_s"]), start + fraction * delta, None,
                   "executed_route_move")
    pre_show = [row for row in ep["trace"] if row["stage"] == "pre_show_backtrack"]
    for index, row in enumerate(pre_show):
        sample(float(row["time_s"]), row["ape_xy_m"], row["owner_pose_evaluator"],
               "pre_show_backtrack", index % 5 == 0)
    guard = [row for row in ep["trace"]
             if row["stage"] == "owner_reacquire_guard_tick"]
    previous_guard = pre_show[-1] if pre_show else None
    for index, row in enumerate(guard):
        if previous_guard is not None:
            t0, t1 = float(previous_guard["time_s"]), float(row["time_s"])
            p0, p1 = (previous_guard["owner_pose_evaluator"],
                      row["owner_pose_evaluator"])
            x0, x1 = (np.asarray(previous_guard["ape_xy_m"]),
                      np.asarray(row["ape_xy_m"]))
            for t in np.arange(t0 + DT, t1 - .001, DT):
                fraction = float(np.clip((t - t0) / (t1 - t0), 0., 1.))
                pose = {"torso_xy_m": ((1.-fraction) * np.asarray(
                            p0["torso_xy_m"]) + fraction * np.asarray(
                            p1["torso_xy_m"])).tolist(),
                        "hand_xyz_m": ((1.-fraction) * np.asarray(
                            p0["hand_xyz_m"]) + fraction * np.asarray(
                            p1["hand_xyz_m"])).tolist(),
                        "show_can": False}
                sample(float(t), (1.-fraction) * x0 + fraction * x1,
                       pose, "interpolated_reacquire_guard")
        sample(float(row["time_s"]), row["ape_xy_m"],
               row["owner_pose_evaluator"], "reacquire_guard",
               index % 5 == 0)
        previous_guard = row
    interrupted = next((row for row in ep["trace"]
                        if row["stage"] == "owner_post_retreat_view"), None)
    if interrupted is not None:
        sample(float(interrupted["time_s"]), interrupted["ape_xy_m"],
               interrupted["owner_pose_evaluator"], "post_retreat_view", True)
    restage = [row for row in ep["trace"]
               if row["stage"] == "pre_show_motion_restaging"]
    if restage and pre_show:
        held = pre_show[-1]
        for t in np.arange(float(held["time_s"]) + DT,
                           float(restage[0]["time_s"]), DT):
            sample(float(t), held["ape_xy_m"], held["owner_pose_evaluator"],
                   "person_rgbd_reacquire_hold")
    for index, row in enumerate(restage):
        sample(float(row["time_s"]), row["ape_xy_m"], row["owner_pose_evaluator"],
               "pre_show_motion_restaging", index % 5 == 0)
    last_pre = (restage or pre_show)[-1] if restage or pre_show else None
    held_route_xy = np.asarray(last_pre["ape_xy_m"] if last_pre
                               else route["end_true_xy_m"])
    reach = [row for row in ep["trace"] if row["stage"] == "owner_reaches"]
    previous_time = float(last_pre["time_s"] if last_pre
                          else route["end_time_s"])
    previous_hand = (np.asarray(last_pre["owner_pose_evaluator"]["hand_xyz_m"])
                     if last_pre else interaction.REST_HAND.copy())
    if interrupted is not None:
        held_route_xy = np.asarray(interrupted["ape_xy_m"])
        previous_time = float(interrupted["time_s"])
        previous_hand = np.asarray(
            interrupted["owner_pose_evaluator"]["hand_xyz_m"])
    ask = next((row for row in ep["trace"]
                if row["stage"] == "ask_owner_to_show_before_standoff"), None)
    if ask is not None and last_pre is not None:
        for t in np.arange(previous_time + DT, float(ask["time_s"]), DT):
            sample(float(t), held_route_xy, last_pre["owner_pose_evaluator"],
                   "pre_show_view_check")
        previous_time = float(ask["time_s"])
    for row in reach:
        t1 = float(row["time_s"])
        hand1 = np.asarray(row["pose_evaluator"]["hand_xyz_m"])
        for t in np.arange(previous_time + DT, t1, DT):
            fraction = np.clip((t - previous_time) / (t1 - previous_time), 0., 1.)
            hand = previous_hand + fraction * (hand1 - previous_hand)
            torso = np.asarray(row["pose_evaluator"]["torso_xy_m"])
            direction = hand[:2] - torso
            direction /= np.linalg.norm(direction)
            can = hand.copy()
            can[:2] += .14 * direction
            sample(float(t), held_route_xy,
                   {"torso_xy_m": torso.tolist(), "hand_xyz_m": hand.tolist(),
                    "can_xyz_m": can.tolist()}, "interpolated_owner_reach")
        sample(t1, held_route_xy, row["pose_evaluator"], "owner_reach_capture", True)
        previous_time, previous_hand = t1, hand1
    backtrack = [row for row in ep["trace"] if row["stage"] == "standoff_backtrack"]
    final_reach_pose = reach[-1]["pose_evaluator"] if reach else None
    if backtrack:
        first_backtrack = float(backtrack[0]["time_s"])
        for t in np.arange(previous_time + DT, first_backtrack, DT):
            sample(float(t), held_route_xy, final_reach_pose,
                   "presented_before_standoff")
        for index, row in enumerate(backtrack):
            sample(float(row["time_s"]), row["ape_xy_m"],
                   row["owner_pose_evaluator"], "standoff_backtrack",
                   index % 5 == 0)
        previous_time = float(backtrack[-1]["time_s"])
    monitor = [row for row in ep["trace"]
               if row["stage"] == "standoff_recognition_monitor"]
    for index, row in enumerate(monitor):
        sample(float(row["time_s"]), row["ape_xy_m"],
               row["owner_pose_evaluator"], "recognition_monitor",
               index % 5 == 0)
    final_pose = monitor[-1]["owner_pose_evaluator"] if monitor else (
        reach[-1]["pose_evaluator"] if reach else (
            interrupted["owner_pose_evaluator"] if interrupted else (
                last_pre["owner_pose_evaluator"] if last_pre else None)))
    final_xy = np.asarray(ep["final_actual_xy_m"])
    last_time = float(monitor[-1]["time_s"] if monitor else previous_time)
    for index, t in enumerate(np.arange(last_time + DT, ep["hold_until_s"] + .011, DT)):
        sample(float(t), final_xy, final_pose, "recognition_or_hold",
               index % 5 == 0)
    video = Path(ep["video"]["path"])
    reader = imageio.get_reader(video)
    frames = reader.count_frames()
    reader.close()
    packets = [row["depth_packets"] for row in ep["trace"]
               if row["stage"] == "owner_depth_change" and row.get("depth_packets")]
    depth_packet_sha_pass = all(sha256(Path(p["path"])) == p["sha256"]
                                for p in packets)
    reacquire_packets = [row for row in ep["trace"]
                         if row["stage"] == "owner_person_rgbd_packets"]
    reacquire_packets += [row["raw_packets"] for row in ep["trace"]
                          if row["stage"] == "owner_post_retreat_view" and
                          row.get("raw_packets")]
    reacquire_packet_sha_pass = all(sha256(Path(p["path"])) == p["sha256"]
                                    for p in reacquire_packets)
    return {"scenario_id": case["id"], "response": ep["response"],
            "outcome": ep["outcome"],
            "scene_fingerprint_pass": fingerprint == ep["scene_fingerprint"],
            "nearest": minima, "person_room_nearest": room_min,
            "can_room_nearest": can_room_min,
            "ape_5cm_margin_pass": minima["ape_any"]["gap_m_capped_at_10cm"] >= .05,
            "person_room_5cm_pass": room_min["gap_m_capped_at_30cm"] >= .05,
            "can_room_nonoverlap_pass": can_room_min["gap_m_capped_at_10cm"] >= 0.,
            "video_sha_pass": sha256(video) == ep["video"]["sha256"],
            "video_frames_pass": frames == ep["video"]["frames"],
            "depth_packet_sha_pass": depth_packet_sha_pass,
            "reacquire_packet_sha_pass": reacquire_packet_sha_pass,
            "geometry_samples": samples}


def main():
    report = json.loads(TRIAL.read_text())
    cases = []
    for ep in report["episodes"]:
        result = audit_episode(ep)
        cases.append(result)
        print(result["response"], result["outcome"],
              "ape gap", result["nearest"]["ape_any"],
              "room", result["person_room_nearest"], flush=True)
    output = {"source_sha256": sha256(Path(__file__)),
              "trial_report_sha256": sha256(TRIAL),
              "trial_source_sha_pass": report["source_sha256"] == sha256(
                  ROOT / "experiments/ape_high_level/closed_loop/run_robocasa_first_return_person_reacquire_v574.py"),
              "turn_source_sha_pass": report["depth_standoff_turn_source_sha256"] == sha256(
                  Path(turn.__file__)),
              "depth_change_source_sha_pass": report["depth_change_source_sha256"] == sha256(
                  Path(turn.changed_points.__code__.co_filename)),
              "person_source_sha_pass": report["person_source_sha256"] == sha256(
                  Path(person.__file__)),
              "person_reacquire_source_sha_pass": report["person_reacquire_source_sha256"] == sha256(
                  Path(turn.person_reacquire.__file__)),
              "async_person_source_sha_pass": report["async_person_source_sha256"] == sha256(
                  Path(turn.async_person.__file__)),
              "person_detector_weights_sha_pass": report["person_detector_weights_sha256"] == sha256(
                  turn.person_reacquire.DETECTOR / "model.safetensors"),
              "all_videos_pass": all(row["video_sha_pass"] and row["video_frames_pass"]
                                     and row["depth_packet_sha_pass"]
                                     and row["reacquire_packet_sha_pass"]
                                     for row in cases),
              "scope": "independent MuJoCo signed distances across matched ideal/first-five-owner-sector-blackout 30cm owner-range episodes with two-image person RGB-D recovery before show, stationary and 40cm moving owner, possible restaging/view/show/second return and full late hold including detector latency; 20ms kinematic surrogate and source/video/raw-depth/model hashes",
              "cases": cases}
    OUT.write_text(json.dumps(output, indent=2) + "\n")


if __name__ == "__main__":
    main()
