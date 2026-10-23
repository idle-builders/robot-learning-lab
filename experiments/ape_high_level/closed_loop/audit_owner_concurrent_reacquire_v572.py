"""Independent sensor-schedule and full-geometry replay for V571."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import mujoco
import numpy as np

import robocasa_owner_person as person
import robocasa_person_interaction as interaction
import run_robocasa_person_joined_v497 as joined
from analyze_robocasa_owner_monitor_v501 import changed_beams
from audit_robocasa_layout2_routes_v409 import clearance
from run_robocasa_dynamic_suite_v439 import motion
from run_robocasa_scan_height_v433 import height_scan


ROOT = Path(__file__).resolve().parents[3]
SOURCE = ROOT / "artifacts/ape-robocasa-first-return-person-reacquire-v566/report.json"
TRIAL = ROOT / "artifacts/ape-robocasa-concurrent-owner-reacquire-v571/report.json"
OUT = TRIAL.with_name("audit-v572.json")
SCAN = height_scan((.30,))


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def scene_at(model, data, prop, held, alarm_elapsed, elapsed):
    t = 4.8 + elapsed
    extra = .8 * min(1., elapsed / 2.)
    amount = min(.5, .25 * max(0., elapsed - alarm_elapsed - .2))
    xy = held + [0., -amount]
    model.body_pos[model.body("audit_ape").id, :2] = xy
    model.body_pos[model.body("route_obstacle").id, :2] = prop(t)
    person.set_owner_pose(model, interaction.TORSO_XY + [0., -.4 - extra],
                          interaction.REST_HAND + [0., -.4 - extra, 0.],
                          show_can=False)
    mujoco.mj_forward(model, data)
    return xy


def main():
    trial = json.loads(TRIAL.read_text())
    source = json.loads(SOURCE.read_text())
    ep = next(item for item in source["episodes"]
              if item["response"] == "early_owner_mid_sector_blackout")
    held = np.asarray(next(row for row in reversed(ep["trace"])
                           if row["stage"] == "pre_show_backtrack")["ape_xy_m"])
    alarm_elapsed = trial["range_alarm_time_s"] - 4.8
    model = joined.scene_with_owner_and_prop(ep["scenario"]["from_xy_m"])
    data = mujoco.MjData(model)
    prop = motion(ep["scenario"])
    ape_body = model.body("audit_ape").id
    ape_geom = model.geom("audit_ape_geom").id
    people = [model.geom(name).id for name in person.PERSON_GEOMS]
    baseline = None
    streak = 0
    packet_match = True
    trace_pose_error = 0.
    scan_elapsed = []
    for row in trial["trace"]:
        elapsed = row["elapsed_wall_s"]
        xy = scene_at(model, data, prop, held, alarm_elapsed, elapsed)
        trace_pose_error = max(trace_pose_error,
                               float(np.linalg.norm(xy - row["ape_xy_m"])))
        if "range_changed_beams" in row:
            packet = SCAN(model, data, xy, 5.,
                          np.random.default_rng(14000 + len(scan_elapsed)))
            if baseline is None:
                baseline = packet
            changed = changed_beams(baseline, packet)
            streak = streak + 1 if changed >= 3 else 0
            packet_match &= (changed == row["range_changed_beams"] and
                             streak == row["range_motion_streak"])
            scan_elapsed.append(elapsed)
    intervals = np.diff(scan_elapsed)
    minimum_person = {"gap_m": 1., "time_s": None}
    minimum_scene = {"gap_m_capped_at_10cm": .1,
                     "geom": None, "time_s": None}
    count = 0
    for elapsed in np.arange(0., trial["detector_finished_wall_s"] + .01, .02):
        xy = scene_at(model, data, prop, held, alarm_elapsed, float(elapsed))
        gap = min(float(mujoco.mj_geomDistance(model, data, ape_geom, geom,
                                               1., None)) for geom in people)
        if gap < minimum_person["gap_m"]:
            minimum_person = {"gap_m": gap,
                              "time_s": round(4.8 + elapsed, 4)}
        room_gap, near = clearance(model, data, ape_body, ape_geom, xy)
        if room_gap < minimum_scene["gap_m_capped_at_10cm"]:
            minimum_scene = {"gap_m_capped_at_10cm": room_gap,
                             "geom": near, "time_s": round(4.8 + elapsed, 4)}
        count += 1
    image_hashes_pass = all(sha(item["path"]) == item["sha256"]
                            for item in (trial["before_image"], trial["after_image"]))
    output = {"source_sha256": sha(Path(__file__)),
              "trial_report_sha256": sha(TRIAL),
              "trial_source_sha_pass": trial["source_sha256"] == sha(
                  ROOT / "experiments/ape_high_level/closed_loop/run_owner_concurrent_reacquire_v571.py"),
              "source_episode_sha_pass": trial["source_episode_sha256"] == sha(SOURCE),
              "model_weights_sha_pass": trial["person_detector_weights_sha256"] == sha(
                  ROOT / ".cache/ape-high-level/models/grounding-dino-tiny/model.safetensors"),
              "image_hashes_pass": image_hashes_pass,
              "range_packet_replay_pass": packet_match,
              "scan_count_pass": len(scan_elapsed) == trial["range_scans"],
              "scan_interval_max_pass": abs(float(np.max(intervals)) - trial["range_scan_wall_interval_max_s"]) < 1e-8,
              "trace_pose_max_error_m": trace_pose_error,
              "minimum_ape_person_gap": minimum_person,
              "minimum_ape_scene_gap": minimum_scene,
              "ape_5cm_margin_pass": minimum_scene["gap_m_capped_at_10cm"] >= .05,
              "independent_geometry_samples": count,
              "scope": "independent 20ms continuous scripted owner/ape/prop MuJoCo replay across observed wall-clock detector duration, plus each recorded actor range-scan time/seed and source/image/model hashes; still kinematic retreat and ideal range"}
    OUT.write_text(json.dumps(output, indent=2) + "\n")
    print(output, flush=True)


if __name__ == "__main__":
    main()
