"""Independent 20 ms full-horizon body/media replay for V713's active view."""

from __future__ import annotations

from bisect import bisect_right
import hashlib
import json
from pathlib import Path

import imageio.v2 as imageio
import mujoco
import numpy as np

from audit_robocasa_layout2_routes_v409 import clearance, sha256
import robocasa_owner_person as person
import robocasa_person_interaction as interaction
import run_robocasa_layout2_active_view_v713 as trial
import run_robocasa_layout2_route_v407 as route
import run_robocasa_layout2_side_prop_v701 as side


SOURCE = Path(__file__)
REPORT = trial.OUT / "report.json"
OUT = trial.OUT / "audit-v714.json"
DT = .02


def pose_events(ep):
    events = [(0., route.START.copy())]
    for row in ep["route_episode"]["trace"]:
        start = np.asarray(row["actual_xy_m"])
        events.append((float(row["time_s"]), start))
        if row["motion_substeps"]:
            delta = ((np.asarray(row["requested_xy_m"]) -
                      np.asarray(row["estimated_xy_m"])) * (1. + .30 / 5.89))
            for index, sub in enumerate(row["motion_substeps"]):
                fraction = (index if sub["stopped_before_move"] else index + 1) / 20
                events.append((float(sub["time_s"]), start + fraction * delta))
    before_correction = np.asarray(ep["route_episode"]["end_true_xy_m"])
    events.append((float(ep["route_episode"]["end_time_s"]), before_correction))
    for row in ep["trace"]:
        if row.get("stage") == "person_active_view_correction" and row["motion_substeps"]:
            delta = np.asarray(row["actor_command_delta_xy_m"]) * (1. + .30 / 5.89)
            for index, sub in enumerate(row["motion_substeps"]):
                fraction = (index if sub["actor_stopped_before_move"] else index + 1) / 20
                events.append((float(sub["time_s"]), before_correction + fraction * delta))
            before_correction = events[-1][1]
        if row.get("stage") == "memory_route_decision":
            current = np.asarray(row["evaluator_actual_xy_m"])
            events.append((float(row["time_s"]), current))
            for sub in row["motion_substeps"]:
                if "evaluator_actual_xy_m" in sub:
                    current = np.asarray(sub["evaluator_actual_xy_m"])
                events.append((float(sub["time_s"]), current.copy()))
        if row.get("stage") in ("memory_viewpoint_reached", "memory_post_stop_hold"):
            events.append((float(row["time_s"]), np.asarray(row["evaluator_actual_xy_m"])))
        if row.get("stage") == "owner_inference_near_body_monitor":
            for sample in row["monitor_samples"]:
                if "actual_xy_m" in sample:
                    events.append((float(sample["time_s"]),
                                   np.asarray(sample["actual_xy_m"])))
        if row.get("stage") == "interrupted_monitor_continuation":
            events.append((float(row["time_s"]), np.asarray(row["actual_xy_m"])))
    events.append((float(ep["hold_until_s"]), np.asarray(ep["final_actual_xy_m"])))
    events.sort(key=lambda pair: pair[0])
    return events


def main():
    report = json.loads(REPORT.read_text())
    ep = report["episode"]
    model = person.scene_with_person(side.START, layout_id=2, style_id=1,
                                     clutter_mode=0)
    torso = interaction.TORSO_XY + [-.3, 0.]
    rest = interaction.REST_HAND + [-.3, 0., 0.]
    person.set_owner_pose(model, torso, rest, show_can=False)
    fingerprint = hashlib.sha256(b"".join(array.tobytes() for array in
        (model.body_pos, model.geom_pos, model.geom_quat,
         model.geom_size, model.geom_rgba, model.light_diffuse))).hexdigest()
    data = mujoco.MjData(model)
    ape_body = model.body("audit_ape").id
    ape_geom = model.geom("audit_ape_geom").id
    prop_body = model.body("route_obstacle").id
    can_body = model.body("shown_can").id
    events = pose_events(ep)
    event_times = [row[0] for row in events]
    reach = [row for row in ep["trace"] if row.get("stage") == "owner_reaches"]
    ask = next(row for row in ep["trace"] if row.get("stage") ==
               "ask_owner_to_show_from_safe_stop")
    hand_events = [(float(ask["time_s"]), rest)] + [
        (float(row["time_s"]), np.asarray(row["pose_evaluator"]["hand_xyz_m"]))
        for row in reach]
    rows = []
    for t in np.arange(0., ep["hold_until_s"] + DT / 2, DT):
        xy = events[max(0, bisect_right(event_times, t + 1.e-5) - 1)][1]
        if t < hand_events[0][0]:
            person.set_owner_pose(model, torso, rest, show_can=False)
        else:
            interval = min(len(hand_events) - 1,
                           max(0, bisect_right([event[0] for event in hand_events], t) - 1))
            if interval == len(hand_events) - 1:
                hand = hand_events[-1][1]
            else:
                t0, h0 = hand_events[interval]
                t1, h1 = hand_events[interval + 1]
                alpha = np.clip((t - t0) / (t1 - t0), 0., 1.)
                hand = h0 + alpha * (h1 - h0)
            person.set_owner_pose(model, torso, hand, show_can=True)
            direction = (hand[:2] - torso) / np.linalg.norm(hand[:2] - torso)
            model.body_pos[can_body, :2] = hand[:2] + .14 * direction
        prop = side.prop_at(float(t))
        model.body_pos[prop_body, :2] = prop
        gap, nearest = clearance(model, data, ape_body, ape_geom, xy)
        prop_gap = float(np.linalg.norm(xy - prop) - .17 - .24)
        rows.append({"time_s": round(float(t), 4),
                     "signed_scene_gap_m_capped_at_10cm": gap,
                     "nearest_geom": nearest,
                     "signed_prop_gap_m": prop_gap})
    minimum = min(rows, key=lambda row: row["signed_scene_gap_m_capped_at_10cm"])
    first_breach = next((row for row in rows if row["signed_scene_gap_m_capped_at_10cm"] < .05), None)
    video = Path(ep["video"]["path"])
    reader = imageio.get_reader(video)
    frames = reader.count_frames()
    reader.close()
    packet_rows = ep["route_episode"]["trace"]
    correction = next(row for row in ep["trace"] if row.get("stage") ==
                      "person_active_view_correction")
    first_person = next(row for row in ep["trace"] if row.get("stage") ==
                        "fresh_person_rgbd_before_owner_request")
    second_person = next(row for row in ep["trace"] if row.get("stage") ==
                         "fresh_person_rgbd_after_active_view")
    result = {"schema": "robot-learning-lab.layout2-active-view-audit.v1",
              "source_sha256": sha256(SOURCE),
              "trial_report_sha256": sha256(REPORT),
              "scene_fingerprint_pass": fingerprint == ep["scene_fingerprint"],
              "20ms_full_horizon_samples": len(rows),
              "first_below_5cm": first_breach,
              "minimum_signed_scene_gap": minimum,
              "minimum_signed_prop_gap_m": min(row["signed_prop_gap_m"] for row in rows),
              "route_packets_72_beams": len(packet_rows) and all(
                  len(row["range_72_fused_m"]) == 72 for row in packet_rows),
              "active_view_substeps_20": len(correction["motion_substeps"]) == 20,
              "active_view_guard_clear": all(
                  not sub["actor_stopped_before_move"]
                  for sub in correction["motion_substeps"]),
              "first_person_status": first_person["actor"]["status"],
              "second_person_status": second_person["actor"]["status"],
              "owner_find": ep["outcome"],
              "moving_return_status": ep["memory_return"]["route_status"],
              "video_sha_pass": sha256(video) == ep["video"]["sha256"],
              "video_frames_pass": frames == ep["video"]["frames"],
              "video_frames": frames,
              "scope": "Independent 20ms rebuilt layout2 scene through the full 23s side-prop path, including route, active-view move, scripted owner reach and held response; 50cm cylinder, not articulated-body safety."}
    OUT.write_text(json.dumps(result, indent=2) + "\n")
    print(result, flush=True)


if __name__ == "__main__":
    main()
