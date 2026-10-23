"""Independent packet, timing and geometry replay of the V569 retreat screen."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import imageio.v2 as imageio
import mujoco
import numpy as np

import robocasa_owner_person as person
import robocasa_person_interaction as interaction
import run_robocasa_person_joined_v497 as joined
from analyze_robocasa_owner_monitor_v501 import changed_beams
from run_robocasa_dynamic_suite_v439 import motion
from run_robocasa_scan_height_v433 import height_scan
from audit_robocasa_layout2_routes_v409 import clearance


ROOT = Path(__file__).resolve().parents[3]
SOURCE = ROOT / "artifacts/ape-robocasa-first-return-person-reacquire-v566/report.json"
TRIAL = ROOT / "artifacts/ape-robocasa-owner-inference-retreat-v569/report.json"
OUT = TRIAL.with_name("audit-v570.json")
SCAN = height_scan((.30,))


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def audit_case(case, scene):
    model = joined.scene_with_owner_and_prop(scene["from_xy_m"])
    data = mujoco.MjData(model)
    prop = motion(scene)
    ape = model.geom("audit_ape_geom").id
    people = [model.geom(name).id for name in person.PERSON_GEOMS]
    baseline = None
    streak = 0
    alarm = None
    max_gap_error = 0.
    packet_match = True
    path_match = True
    first_below = None
    minimum_gap = 1.
    minimum_scene_gap = {"gap_m_capped_at_10cm": .1,
                         "time_s": None, "geom": None}
    held = np.asarray(case["held_ape_xy_m"])
    for index, row in enumerate(case["trace"]):
        t = float(row["time_s"])
        xy = np.asarray(row["ape_xy_m"])
        model.body_pos[model.body("audit_ape").id, :2] = xy
        south = .4 + row["additional_owner_south_m"]
        person.set_owner_pose(model, interaction.TORSO_XY + [0., -south],
                              interaction.REST_HAND + [0., -south, 0.],
                              show_can=False)
        model.body_pos[model.body("route_obstacle").id, :2] = prop(t)
        mujoco.mj_forward(model, data)
        gap = min(float(mujoco.mj_geomDistance(model, data, ape, geom, 1., None))
                  for geom in people)
        scene_gap, scene_geom = clearance(
            model, data, model.body("audit_ape").id, ape, xy)
        if scene_gap < minimum_scene_gap["gap_m_capped_at_10cm"]:
            minimum_scene_gap = {"gap_m_capped_at_10cm": scene_gap,
                                 "time_s": round(t, 4), "geom": scene_geom}
        max_gap_error = max(max_gap_error,
                            abs(gap - row["ape_person_gap_evaluator_m"]))
        minimum_gap = min(minimum_gap, gap)
        if gap < .05 and first_below is None:
            first_below = round(t, 4)
        if index % 5 == 0:
            packet = SCAN(model, data, xy, 5., np.random.default_rng(13000 + index))
            if baseline is None:
                baseline = packet
            changed = changed_beams(baseline, packet)
            streak = streak + 1 if changed >= 3 else 0
            if streak >= 2 and alarm is None:
                alarm = round(t, 4)
            packet_match &= (changed == row["range_actor_changed_beams"] and
                             streak == row["range_actor_motion_streak"])
        expected = held.copy()
        if case["scripted_retreat"] and alarm is not None:
            distance = min(.5, .25 * max(0., t - alarm - .2))
            expected += [0., -distance]
        path_match &= bool(np.linalg.norm(xy - expected) < 1e-8)
    video = case.get("video")
    video_pass = True
    if video:
        reader = imageio.get_reader(video["path"])
        frames = reader.count_frames()
        reader.close()
        video_pass = (frames == video["frames"] and
                      sha(video["path"]) == video["sha256"])
    return {"additional_owner_approach_m": case["additional_owner_approach_m"],
            "scripted_retreat": case["scripted_retreat"],
            "range_packet_and_streak_pass": packet_match,
            "first_alarm_time_pass": alarm == case["first_range_motion_alarm_s"],
            "retreat_timing_and_speed_pass": path_match,
            "maximum_gap_replay_error_m": max_gap_error,
            "minimum_ape_person_gap_m": minimum_gap,
            "minimum_ape_scene_gap": minimum_scene_gap,
            "ape_scene_5cm_margin_pass": minimum_scene_gap["gap_m_capped_at_10cm"] >= .05,
            "first_5cm_crossing_pass": first_below == case["first_person_gap_below_5cm_s"],
            "video_pass": video_pass, "geometry_samples": len(case["trace"])}


def main():
    trial = json.loads(TRIAL.read_text())
    source = json.loads(SOURCE.read_text())
    scene = next(ep["scenario"] for ep in source["episodes"]
                 if ep["response"] == "early_owner_mid_sector_blackout")
    cases = [audit_case(case, scene) for case in trial["cases"]]
    output = {"source_sha256": sha(Path(__file__)),
              "trial_report_sha256": sha(TRIAL),
              "trial_source_sha_pass": trial["source_sha256"] == sha(
                  ROOT / "experiments/ape_high_level/closed_loop/screen_owner_approach_retreat_v569.py"),
              "source_episode_sha_pass": trial["source_episode_sha256"] == sha(SOURCE),
              "known_free_path_pass": (
                  trial["source_actor_known_free_retreat_path"]["known_free_path"] and
                  trial["source_actor_known_free_retreat_path"]["distance_m"] >= .5),
              "scope": "independent 20ms MuJoCo person distance, 10Hz ideal 30cm range packet change and .2s-delayed .25m/s surrogate retreat replay; no actual concurrent runtime, human compliance or native body response",
              "cases": cases}
    OUT.write_text(json.dumps(output, indent=2) + "\n")
    for case in cases:
        print(case, flush=True)


if __name__ == "__main__":
    main()
