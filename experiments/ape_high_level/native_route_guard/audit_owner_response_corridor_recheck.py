"""Audit V40's observed range rechecks, two hazards and complete media."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import cv2
import mujoco
import numpy as np

import audit_owner_inference_response as geometry
import owner_response_corridor_recheck as trial
import render_owner_response_corridor_recheck as comparison_renderer


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def frames(path, shape):
    cap = cv2.VideoCapture(str(path))
    assert cap.isOpened(), path
    count = 0
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        assert frame is not None and frame.shape == shape
        count += 1
    cap.release()
    return count


def check_result(item, summary, source, scenario):
    prop_case, mode, run = item["prop_case"], item["mode"], item["run"]
    config = trial.PROP_CASES[prop_case]
    rows = json.loads(Path(run["trace"]["path"]).read_text())
    assert sha(run["trace"]["path"]) == run["trace"]["sha256"]
    assert sha(run["video"]["path"]) == run["video"]["sha256"]
    assert frames(run["video"]["path"], (528, 1040, 3)) == run["video"]["frames"]
    assert run["video"]["frames"] > run["source_video_frames"]
    model = mujoco.MjModel.from_xml_path(str(trial.v38.v35.moving.SCENE_XML))
    data = mujoco.MjData(model)
    start, _, t0, source_hash, checkpoint_packet = trial.v38.geometry_at(
        summary, source["result"]["result"]["episode"], scenario, model, data)
    assert run["checkpoint_range_packet_exact"]
    assert run["checkpoint_time_s"] == t0
    assert run["source_v36_report_sha256"] == source_hash
    assert rows[0]["checkpoint_range_packet"] == checkpoint_packet
    assert rows[0]["old_owner_identity"] == "unknown"
    hold = [row for row in rows if row["stage"] == "post_alarm_range_watch"]
    streak = 0
    resume = None
    for row in hold:
        healthy = trial.v38.side_healthy(row["actor_packet"])
        nearest = trial.v38.v35.side_range(row["actor_packet"]) if healthy else None
        assert row["side_health"] == healthy
        assert row["nearest_side_range_m"] == nearest
        clear = healthy and nearest is not None and nearest >= trial.v38.CLEAR_SIDE_RANGE_M
        streak = streak+1 if clear else 0
        assert row["clear_streak"] == streak
        if streak >= trial.v38.CLEAR_STREAK and resume is None:
            resume = row["time_s"]
            assert row["actor_action"] == "resume_after_three_fresh_clear_packets"
    assert resume == run["resume_time_s"]
    fresh = next(row for row in rows if row["stage"] == "fresh_new_show_capture")
    assert fresh["time_s"] > resume
    assert sha(fresh["raw_rgbd"]["path"]) == fresh["raw_rgbd"]["sha256"]
    watch = [row for row in rows if row["stage"] == "second_inference_range_watch"]
    inference = next(row for row in rows if row["stage"] == "second_inference_result")
    sensor_video = next(row for row in rows if row["stage"] == "flank_sensor_video")
    assert sha(sensor_video["path"]) == sensor_video["sha256"]
    assert frames(sensor_video["path"], (480, 640, 3)) == sensor_video["frames"]
    assert sensor_video["frames"] == len(watch)+1
    camera_streak = 0
    camera_alarm = expected_halt = None
    plan = next(row for row in rows if row["stage"] == "second_response_plan")
    previous_packet = None
    for row in watch:
        t = row["time_s"]
        packet = row["actor_packet"]
        assert len(packet) == 72
        assert row["side_health"] == trial.v38.side_healthy(packet)
        expected_prop = trial.prop_at(t, t0, config)
        assert np.linalg.norm(np.asarray(row["evaluator_route_obstacle_xy_m"])-
                              expected_prop) < 1e-5
        sensor = row["flank_rgbd_packet"]
        assert sensor is not None and sensor["actor_pose_healthy"]
        relative = t-run["second_inference_start_s"]
        assert abs(sensor["joint_head_yaw_deg"]-trial.v39.head_yaw(relative)) < .00011
        assert abs(sensor["imu_body_yaw_deg"]-trial.v39.body_yaw(relative)) < .00011
        assert sensor["imu_age_s"] == 0.
        cue = (sensor["actor_cue_usable"] and
               sensor["rgb_changed_pixels"] >= trial.v39.RGB_MOTION_PIXELS and
               sensor["depth_changed_pixels"] >= trial.v39.DEPTH_MOTION_PIXELS)
        assert sensor["actor_motion_cue"] == cue
        camera_streak = camera_streak+1 if cue else 0
        assert row["camera_motion_streak"] == camera_streak
        if camera_streak >= 2 and camera_alarm is None:
            camera_alarm = t
        if (mode == "recheck_065" and expected_halt is None and
                camera_alarm is not None and t > camera_alarm and
                previous_packet is not None):
            measured = [value for value in previous_packet if value is not None]
            nearest_previous = min(measured, default=None)
            if nearest_previous is None or nearest_previous < trial.RANGE_RECHECK_M:
                expected_halt = t
                assert row["actor_action"] == "hold_on_new_near_body_range"
                assert (inference["response_halt_range_m"] is None or
                        abs(inference["response_halt_range_m"]-
                            nearest_previous) < 1e-8)
        previous_packet = packet
        model.body_pos[model.body("route_obstacle").id, :2] = expected_prop
        person = trial.v38.human_xy("blind_return_flank_fast_response", summary,
                                     t, run["request_time_s"],
                                     run["second_inference_start_s"])
        ape = np.asarray(row["ape_xy_m"])
        assert abs(geometry.gap(model, data, ape, person)-
                   row["evaluator_person_gap_m"]) < .00011
        assert abs(trial.v38.prop_gap(model, data, ape)-
                   row["evaluator_prop_gap_m"]) < .00011
    assert camera_alarm is not None
    assert abs(inference["camera_alarm_s"]-camera_alarm) < .00011
    assert inference["alarm_s"] is None
    assert inference["pose_loss_s"] is None
    assert abs(plan["alarm_s"]-camera_alarm) < .00011
    assert plan["south_range_clear"]
    assert plan["parameters"] == trial.v38.FLANK_RESPONSE[
        "blind_return_flank_fast_response"]
    first = inference["first_response_s"]
    assert first is not None and first >= camera_alarm+plan["parameters"]["delay_s"]
    assert (inference["response_halt_s"] is None) == (expected_halt is None)
    if expected_halt is not None:
        assert abs(inference["response_halt_s"]-expected_halt) < .00011
    else:
        assert inference["response_halt_displacement_m"] is None
    assert item["camera_alarm_s"] == inference["camera_alarm_s"]
    assert item["first_response_s"] == first
    assert item["response_halt_s"] == inference["response_halt_s"]
    samples = run["evaluator_gap_samples"]
    for row in samples:
        t = row["time_s"]
        distance = min(trial.v38.SECOND_RESPONSE_MAX_M,
                       max(0., plan["parameters"]["speed_mps"]*(t-first)))
        if inference["response_halt_displacement_m"] is not None:
            distance = min(distance, inference["response_halt_displacement_m"])
        ape = start+[0., -distance]
        assert np.linalg.norm(np.asarray(row["ape_xy_m"])-ape) < .00011
        model.body_pos[model.body("route_obstacle").id, :2] = \
            trial.prop_at(t, t0, config)
        person = trial.v38.human_xy("blind_return_flank_fast_response", summary,
                                     t, run["request_time_s"],
                                     run["second_inference_start_s"])
        assert abs(geometry.gap(model, data, ape, person)-row["gap_m"]) < .00011
        assert abs(trial.v38.prop_gap(model, data, ape)-
                   row["prop_gap_m"]) < .00011
    assert run["minimum_proxy_gap_m"] == min(row["gap_m"] for row in samples)
    assert run["minimum_prop_gap_m"] == min(row["prop_gap_m"] for row in samples)
    assert run["whole_episode_minimum_proxy_gap_m"] == min(
        summary["inference_geometry_evaluator"]["minimum_gap_m"],
        run["minimum_proxy_gap_m"])
    assert run["outcome"] == "new_show_flank_motion_unresolved"
    assert not any(row["stage"] == "new_show_metric_item_seen" for row in rows)
    return {"prop_case": prop_case, "mode": mode,
            "camera_alarm_s": camera_alarm,
            "first_response_s": first,
            "response_halt_s": expected_halt,
            "minimum_person_gap_m": run["minimum_proxy_gap_m"],
            "minimum_prop_gap_m": run["minimum_prop_gap_m"],
            "whole_episode_minimum_person_gap_m": run[
                "whole_episode_minimum_proxy_gap_m"],
            "video_frames": run["video"]["frames"]}


def main():
    report = json.loads((trial.OUT / "report.json").read_text())
    assert report["schema"] == "robot-learning-lab.ape-owner-corridor-recheck.v1"
    assert report["source_sha256"] == sha(trial.__file__)
    assert report["v38_source_sha256"] == sha(trial.v38.__file__)
    assert report["v39_source_sha256"] == sha(trial.v39.__file__)
    assert report["scene_sha256"] == sha(trial.v38.v35.moving.SCENE_XML)
    assert report["v36_report_sha256"] == sha(trial.v38.SOURCE)
    assert report["current_design_pointer_sha256"] == sha(
        trial.ROOT / "robot/ape-current-design.json")
    assert report["range_recheck_m"] == trial.RANGE_RECHECK_M
    assert report["prop_cases"] == trial.PROP_CASES
    assert len(report["results"]) == len(trial.PROP_CASES)*len(trial.MODES)
    source = trial.v38.original()
    summary = source["summary"]
    policy = json.loads(trial.v38.joined.CONFIG.read_text())
    scenario = next(x for x in policy["cases"] if x["id"] == "late_crossing")
    rows = [check_result(item, summary, source, scenario)
            for item in report["results"]]
    assert [(row["prop_case"], row["mode"]) for row in rows] == [
        (case, mode) for case in trial.PROP_CASES for mode in trial.MODES]
    comparison = json.loads((trial.OUT / "comparison.json").read_text())
    assert comparison["schema"] == "robot-learning-lab.ape-owner-corridor-comparisons.v1"
    assert comparison["report_sha256"] == sha(trial.OUT / "report.json")
    assert comparison["source_sha256"] == sha(comparison_renderer.__file__)
    assert set(comparison["videos"]) == set(trial.PROP_CASES)
    assert set(comparison["overhead_videos"]) == set(trial.PROP_CASES)
    lookup = {(item["prop_case"], item["mode"]): item
              for item in report["results"]}
    for case, video in comparison["videos"].items():
        assert sha(video["path"]) == video["sha256"]
        assert frames(video["path"], (390, 1280, 3)) == video["frames"]
        assert video["left_source_video_sha256"] == lookup[
            (case, "open_loop")]["run"]["video"]["sha256"]
        assert video["right_source_video_sha256"] == lookup[
            (case, "recheck_065")]["run"]["video"]["sha256"]
    for video in comparison["overhead_videos"].values():
        assert video["kind"] == "evaluator_replay_from_20ms_gap_samples"
        assert sha(video["path"]) == video["sha256"]
        assert frames(video["path"], (680, 1280, 3)) == video["frames"]
        assert video["frames"] == len(np.arange(22.0, 24.81, .05))
    audit = {"passed": True, "rows": rows,
             "comparison_videos": comparison["videos"],
             "overhead_videos": comparison["overhead_videos"],
             "scope": "Six matched owner-route continuations from an exact 72-ray checkpoint. Independently checks fresh camera/IMU packets, preceding-packet range-stop causality, first command and halt timing, person and moving-prop 20ms gaps, complete and paired videos and source hashes. Scripts, ideal sensors and kinematic response remain unqualified."}
    (trial.OUT / "audit.json").write_text(json.dumps(audit, indent=2)+"\n")
    print(json.dumps(audit, indent=2))


if __name__ == "__main__":
    main()
