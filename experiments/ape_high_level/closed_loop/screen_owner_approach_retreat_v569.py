"""Screen a range-triggered surrogate retreat during slow person inference.

The owner trajectory belongs only to the simulated scene. The potential alarm
uses fresh mid-height range packets; MuJoCo person distance is evaluator-only.
The retreat is a scripted counterfactual, not a native escape skill or safety proof.
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
import run_robocasa_joined_owner_route_v410 as owner
import run_robocasa_person_joined_v497 as joined
from analyze_robocasa_owner_monitor_v501 import changed_beams
from run_robocasa_dynamic_suite_v439 import motion
from run_robocasa_scan_height_v433 import height_scan
from screen_robocasa_owner_safe_stop_v494 import aim_camera


ROOT = Path(__file__).resolve().parents[3]
SOURCE = ROOT / "artifacts/ape-robocasa-first-return-person-reacquire-v566/report.json"
OUT = ROOT / "artifacts/ape-robocasa-owner-inference-retreat-v569"
DT = .02
SCAN = height_scan((.30,))
ADDITIONAL_APPROACH_M = .8
APPROACH_DURATION_S = 2.
CHANGE_BEAMS = 3
CONSECUTIVE_PACKETS = 2
RESPONSE_DELAY_S = .2
RETREAT_SPEED_M_S = .25
RETREAT_DISTANCE_M = .5


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def evaluate(ep, extra_m, start_s, inference_end_s, retreat, video):
    model = joined.scene_with_owner_and_prop(ep["scenario"]["from_xy_m"])
    data = mujoco.MjData(model)
    renderer = mujoco.Renderer(model, height=480, width=640) if video else None
    prop = motion(ep["scenario"])
    held = next(row for row in reversed(ep["trace"])
                if row["stage"] == "pre_show_backtrack")
    held_xy = np.asarray(held["ape_xy_m"])
    actual = held_xy.copy()
    model.body_pos[model.body("audit_ape").id, :2] = actual
    # The forward camera framing is for the diagnostic video only. It is not
    # consulted by the range alarm or geometry evaluator.
    aim_camera(model, interaction.CAMERA, actual, actual + [0., 1.], 20)
    ape_geom = model.geom("audit_ape_geom").id
    person_geoms = [model.geom(name).id for name in person.PERSON_GEOMS]
    baseline = None
    streak = 0
    alarm_s = None
    first_gap_below_5cm_s = None
    min_gap = {"gap_m": 1., "time_s": None, "person_geom": None}
    rows, frames = [], []
    n = int(round((inference_end_s - start_s) / DT))
    for index in range(n + 1):
        t = start_s + index * DT
        if retreat and alarm_s is not None:
            amount = min(RETREAT_DISTANCE_M,
                         RETREAT_SPEED_M_S * max(0., t - alarm_s - RESPONSE_DELAY_S))
            actual[:] = held_xy + [0., -amount]
        model.body_pos[model.body("audit_ape").id, :2] = actual
        aim_camera(model, interaction.CAMERA, actual, actual + [0., 1.], 20)
        extra = extra_m * min(1., max(0., (t - start_s) / APPROACH_DURATION_S))
        south = .4 + extra
        person.set_owner_pose(model, interaction.TORSO_XY + [0., -south],
                              interaction.REST_HAND + [0., -south, 0.],
                              show_can=False)
        model.body_pos[model.body("route_obstacle").id, :2] = prop(t)
        mujoco.mj_forward(model, data)
        gaps = [(float(mujoco.mj_geomDistance(model, data, ape_geom, geom, 1., None)),
                 model.geom(geom).name) for geom in person_geoms]
        gap, nearest = min(gaps)
        if gap < min_gap["gap_m"]:
            min_gap = {"gap_m": gap, "time_s": round(t, 4),
                       "person_geom": nearest}
        if gap < .05 and first_gap_below_5cm_s is None:
            first_gap_below_5cm_s = round(t, 4)
        row = {"time_s": round(t, 4), "additional_owner_south_m": extra,
               "ape_xy_m": actual.tolist(),
               "ape_person_gap_evaluator_m": gap}
        if index % 5 == 0:
            packet = SCAN(model, data, actual, 5., np.random.default_rng(13000 + index))
            if baseline is None:
                baseline = packet
            changed = changed_beams(baseline, packet)
            streak = streak + 1 if changed >= CHANGE_BEAMS else 0
            if streak >= CONSECUTIVE_PACKETS and alarm_s is None:
                alarm_s = round(t, 4)
            row["range_actor_changed_beams"] = changed
            row["range_actor_motion_streak"] = streak
            row["range_actor_alarm_now"] = alarm_s == round(t, 4)
        rows.append(row)
        if renderer is not None and index % 10 == 0:
            rgb, _, _ = owner.capture(renderer, data, interaction.CAMERA, False)
            frames.append(owner.owner_frame(
                rgb, f"owner moves during detector wait: t={t:.1f}s, gap={gap:.2f}m",
                None, None))
    result = {"additional_owner_approach_m": extra_m,
              "scripted_retreat": retreat,
              "retreat_delay_s": RESPONSE_DELAY_S if retreat else None,
              "retreat_speed_m_s": RETREAT_SPEED_M_S if retreat else None,
              "retreat_distance_requested_m": RETREAT_DISTANCE_M if retreat else None,
              "retreat_distance_executed_m": float(np.linalg.norm(actual - held_xy)),
              "owner_approach_duration_s": APPROACH_DURATION_S,
              "held_ape_xy_m": held_xy.tolist(),
              "first_range_motion_alarm_s": alarm_s,
              "first_person_gap_below_5cm_s": first_gap_below_5cm_s,
              "alarm_lead_to_5cm_s": (first_gap_below_5cm_s - alarm_s
                                       if alarm_s is not None and
                                       first_gap_below_5cm_s is not None else None),
              "minimum_ape_person_gap": min_gap,
              "range_rule": {"height_m": .30, "packet_hz": 10,
                             "changed_beams": CHANGE_BEAMS,
                             "consecutive_packets": CONSECUTIVE_PACKETS},
              "samples": len(rows), "trace": rows}
    if renderer is not None:
        imageio.mimsave(video, frames, fps=5, codec="libx264", quality=7)
        result["video"] = {"path": str(video), "sha256": sha(video),
                           "frames": len(frames)}
        renderer.close()
    return result


def main():
    report = json.loads(SOURCE.read_text())
    ep = next(item for item in report["episodes"]
              if item["response"] == "early_owner_mid_sector_blackout")
    first = next(row for row in ep["trace"]
                 if row["stage"] == "owner_person_rgbd_reacquire")
    first_result_s = float(first["time_s"])
    OUT.mkdir(parents=True, exist_ok=True)
    restage = next(row for row in ep["trace"]
                   if row["stage"] == "choose_pre_show_restage")["chosen"]
    assert restage["distance_m"] >= RETREAT_DISTANCE_M
    assert restage["known_free_path"]
    cases = [evaluate(ep, 0., 4.8, first_result_s, False, None),
             evaluate(ep, ADDITIONAL_APPROACH_M, 4.8, first_result_s,
                      False, OUT / "held-owner-approach.mp4"),
             evaluate(ep, ADDITIONAL_APPROACH_M, 4.8, first_result_s,
                      True, OUT / "range-triggered-retreat.mp4")]
    output = {"source_sha256": sha(Path(__file__)),
              "source_episode_sha256": sha(SOURCE),
              "source_actor_known_free_retreat_path": restage,
              "first_detector_result_time_s": first_result_s,
              "scope": "same held first-return pose and extra owner motion as V568; compare no response with a scripted .2s delayed .25m/s, .5m retreat along actor-known-free corridor after hypothetical concurrent 10Hz mid-range packet alarm; 20ms MuJoCo person gap, no native policy or real concurrency",
              "cases": cases}
    (OUT / "report.json").write_text(json.dumps(output, indent=2) + "\n")
    for case in cases:
        print(case["additional_owner_approach_m"], case["scripted_retreat"],
              case["first_range_motion_alarm_s"],
              case["first_person_gap_below_5cm_s"],
              case["minimum_ape_person_gap"])


if __name__ == "__main__":
    main()
