"""Independent 20 ms geometry and media audit of capture-triggered V716."""

from __future__ import annotations

from bisect import bisect_right
import hashlib
import json
from pathlib import Path

import imageio.v2 as imageio
import mujoco
import numpy as np

from audit_robocasa_layout2_active_view_v714 import pose_events
from audit_robocasa_layout2_routes_v409 import clearance, sha256
import robocasa_owner_person as person
import robocasa_person_interaction as interaction
import run_robocasa_layout2_owner_inference_monitor_v715 as monitor
import run_robocasa_layout2_owner_inference_trigger_v716 as trial
import run_robocasa_layout2_side_prop_v701 as side


SOURCE = Path(__file__)
REPORT = trial.OUT / "report.json"
OUT = trial.OUT / "audit-v717.json"


def prop_at(t, capture):
    fraction = np.clip((t - capture - .15) / 2.5, 0., 1.)
    return side.START + fraction * (monitor.END - side.START)


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
    reaches = [row for row in ep["trace"] if row.get("stage") == "owner_reaches"]
    ask = next(row for row in ep["trace"] if row.get("stage") ==
               "ask_owner_to_show_from_safe_stop")
    hand_events = [(float(ask["time_s"]), rest)] + [
        (float(row["time_s"]), np.asarray(row["pose_evaluator"]["hand_xyz_m"]))
        for row in reaches]
    hand_times = [event[0] for event in hand_events]
    capture = report["capture_trigger_time_s"]
    samples = []
    for t in np.arange(0., ep["hold_until_s"] + .01, .02):
        xy = events[max(0, bisect_right(event_times, t + 1.e-5) - 1)][1]
        if t < hand_times[0]:
            person.set_owner_pose(model, torso, rest, show_can=False)
        else:
            idx = min(len(hand_events) - 1,
                      max(0, bisect_right(hand_times, t) - 1))
            if idx == len(hand_events) - 1:
                hand = hand_events[-1][1]
            else:
                t0, h0 = hand_events[idx]
                t1, h1 = hand_events[idx + 1]
                hand = h0 + np.clip((t - t0) / (t1 - t0), 0., 1.) * (h1 - h0)
            person.set_owner_pose(model, torso, hand, show_can=True)
            direction = (hand[:2] - torso) / np.linalg.norm(hand[:2] - torso)
            model.body_pos[can_body, :2] = hand[:2] + .14 * direction
        prop = prop_at(t, capture)
        model.body_pos[prop_body, :2] = prop
        gap, nearest = clearance(model, data, ape_body, ape_geom, xy)
        samples.append({"time_s": round(float(t), 4),
                        "scene_gap_m_capped_at_10cm": gap,
                        "nearest_geom": nearest,
                        "prop_gap_m": float(np.linalg.norm(xy - prop) - .17 - .24)})
    first_margin = next((row for row in samples if row["scene_gap_m_capped_at_10cm"] < .05), None)
    first_contact = next((row for row in samples if row["scene_gap_m_capped_at_10cm"] < 0.), None)
    lowest = min(samples, key=lambda row: row["scene_gap_m_capped_at_10cm"])
    guard = next(row for row in ep["trace"] if row.get("stage") ==
                 "owner_inference_near_body_monitor")
    video = Path(ep["video"]["path"])
    reader = imageio.get_reader(video)
    frames = reader.count_frames()
    reader.close()
    result = {"schema": "robot-learning-lab.layout2-owner-inference-audit.v1",
              "source_sha256": sha256(SOURCE),
              "trial_report_sha256": sha256(REPORT),
              "scene_fingerprint_pass": fingerprint == ep["scene_fingerprint"],
              "full_horizon_20ms_samples": len(samples),
              "capture_time_s": capture,
              "actor_first_alarm_time_s": guard["first_alarm_time_s"],
              "actor_image_invalidated": guard["image_invalidated"],
              "first_below_5cm": first_margin,
              "first_contact": first_contact,
              "minimum_scene_gap": lowest,
              "minimum_prop_gap_m": min(row["prop_gap_m"] for row in samples),
              "video_sha_pass": sha256(video) == ep["video"]["sha256"],
              "video_frames_pass": frames == ep["video"]["frames"],
              "video_frames": frames,
              "scope": "Independent full-horizon 20ms geometric reconstruction of V716's observed route, active view, owner show and held capture-triggered prop. The actor alarm is raw range, but it has no body-qualified escape skill."}
    OUT.write_text(json.dumps(result, indent=2) + "\n")
    print(result, flush=True)


if __name__ == "__main__":
    main()
