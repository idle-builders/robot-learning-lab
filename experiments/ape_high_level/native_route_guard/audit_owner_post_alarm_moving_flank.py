"""Independent packet, pose timing, geometry and media audit for V39."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import cv2
import mujoco
import numpy as np

import audit_owner_inference_response as geometry
import owner_post_alarm_moving_flank as trial
import render_owner_post_alarm_moving_flank as comparison_renderer


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def decoded_frames(path, height, width):
    capture = cv2.VideoCapture(str(path))
    assert capture.isOpened(), path
    count = 0
    while True:
        okay, image = capture.read()
        if not okay:
            break
        assert image is not None and image.shape == (height, width, 3)
        count += 1
    capture.release()
    return count


def audit_result(item, summary, source, scenario):
    mode, case, run = item["mode"], item["case"], item["run"]
    rows = json.loads(Path(run["trace"]["path"]).read_text())
    assert sha(run["trace"]["path"]) == run["trace"]["sha256"]
    assert sha(run["video"]["path"]) == run["video"]["sha256"]
    assert decoded_frames(run["video"]["path"], 528, 1040) == run["video"]["frames"]
    assert run["video"]["frames"] > run["source_video_frames"]
    model = mujoco.MjModel.from_xml_path(str(trial.v38.v35.moving.SCENE_XML))
    data = mujoco.MjData(model)
    start, _, t0, source_hash, packet = trial.v38.geometry_at(
        summary, source["result"]["result"]["episode"], scenario, model, data)
    assert run["checkpoint_range_packet_exact"]
    assert run["checkpoint_time_s"] == t0
    assert run["source_v36_report_sha256"] == source_hash
    assert rows[0]["checkpoint_range_packet"] == packet
    assert rows[0]["old_owner_identity"] == "unknown"
    hold = [row for row in rows if row["stage"] == "post_alarm_range_watch"]
    streak = 0
    resume = None
    for row in hold:
        healthy = trial.v38.side_healthy(row["actor_packet"])
        nearest = (trial.v38.v35.side_range(row["actor_packet"])
                   if healthy else None)
        assert len(row["actor_packet"]) == 72
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
    assert fresh["old_owner_identity"] == "unknown"
    assert sha(fresh["raw_rgbd"]["path"]) == fresh["raw_rgbd"]["sha256"]
    watch = [row for row in rows if row["stage"] == "second_inference_range_watch"]
    frozen_route = trial.v38.dynamic.motion(scenario)(run["second_inference_start_s"])
    assert all(np.linalg.norm(np.asarray(row["evaluator_route_obstacle_xy_m"])-
                              frozen_route) < 1e-8 for row in watch)
    inference = next(row for row in rows if row["stage"] == "second_inference_result")
    sensor_video = next(row for row in rows if row["stage"] == "flank_sensor_video")
    assert sha(sensor_video["path"]) == sensor_video["sha256"]
    assert decoded_frames(sensor_video["path"], 480, 640) == sensor_video["frames"]
    assert sensor_video["frames"] == len(watch)+1
    risk_streak = camera_streak = 0
    range_alarm = camera_alarm = pose_loss = None
    for row in watch:
        t = row["time_s"]
        packet = row["actor_packet"]
        healthy = trial.v38.side_healthy(packet)
        nearest = trial.v38.v35.side_range(packet) if healthy else None
        assert len(packet) == 72
        assert row["side_health"] == healthy
        assert row["nearest_side_range_m"] == nearest
        risk = healthy and nearest is not None and nearest < trial.v38.CLEAR_SIDE_RANGE_M
        risk_streak = risk_streak+1 if risk else 0
        assert row["risk_streak"] == risk_streak
        if risk_streak >= 2 and range_alarm is None:
            range_alarm = t
        sensor = row["flank_rgbd_packet"]
        assert sensor is not None and abs(sensor["time_s"]-t) < .00011
        relative_s = t-run["second_inference_start_s"]
        joint = sensor["joint_head_yaw_deg"]
        imu = sensor["imu_body_yaw_deg"]
        if mode == "raw_rgbd":
            assert joint is None
        else:
            assert joint is not None
            assert abs(joint-trial.head_yaw(relative_s)) < .00011
            assert abs(sensor["joint_capture_time_s"]-t) < .00011
        if mode in ("raw_rgbd", "joint_only", "imu_lost"):
            assert imu is None and sensor["imu_capture_time_s"] is None
        else:
            assert imu is not None
            lag = trial.IMU_LAG_S if mode in ("imu_lag_100ms", "imu_lag_gated") else 0.
            assert abs(sensor["imu_capture_time_s"]-(t-lag)) < .00011
            assert abs(sensor["imu_age_s"]-lag) < .00011
            assert abs(imu-trial.body_yaw(max(0., relative_s-lag))) < .00011
        healthy_pose = mode not in ("imu_lag_gated", "imu_lost")
        assert sensor["actor_pose_healthy"] == healthy_pose
        assert abs(sensor["camera_translation_from_start_est_m"]-
                   np.linalg.norm(np.asarray(row["ape_xy_m"])-start)) < .00011
        usable = healthy_pose and sensor["camera_translation_from_start_est_m"] <= .005
        assert sensor["actor_cue_usable"] == usable
        cue = (usable and sensor["rgb_changed_pixels"] >= trial.RGB_MOTION_PIXELS
               and sensor["depth_changed_pixels"] >= trial.DEPTH_MOTION_PIXELS)
        assert sensor["actor_motion_cue"] == cue
        camera_streak = camera_streak+1 if cue else 0
        assert row["camera_motion_streak"] == camera_streak
        if camera_streak >= 2 and camera_alarm is None:
            camera_alarm = t
        if not healthy_pose and pose_loss is None:
            pose_loss = t
    assert (inference["alarm_s"] is None) == (range_alarm is None)
    if range_alarm is not None:
        assert abs(inference["alarm_s"]-range_alarm) < .00011
    assert (inference["camera_alarm_s"] is None) == (camera_alarm is None)
    if camera_alarm is not None:
        assert abs(inference["camera_alarm_s"]-camera_alarm) < .00011
    assert (inference["pose_loss_s"] is None) == (pose_loss is None)
    if pose_loss is not None:
        assert abs(inference["pose_loss_s"]-pose_loss) < .00011
    assert item["camera_alarm_s"] == inference["camera_alarm_s"]
    assert item["first_response_s"] == inference["first_response_s"]
    assert item["pose_loss_s"] == inference["pose_loss_s"]
    plans = [row for row in rows if row["stage"] == "second_response_plan"]
    assert len(plans) == int(camera_alarm is not None)
    plan = plans[0] if plans else None
    first_command = inference["first_response_s"]
    if plan:
        assert abs(plan["alarm_s"]-camera_alarm) < .00011
        alarm_packet = next(row["actor_packet"] for row in watch
                            if abs(row["time_s"]-camera_alarm) < .00011)
        assert plan["south_range_clear"] == trial.v38.south_range_clear(alarm_packet)
        assert plan["parameters"] == trial.v38.FLANK_RESPONSE[
            "blind_return_flank_fast_response"]
        if plan["south_range_clear"] and first_command is not None:
            assert first_command >= camera_alarm+plan["parameters"]["delay_s"]
            assert any(abs(row["time_s"]-first_command) < .00011 for row in watch)
    else:
        assert first_command is None
    samples = run["evaluator_gap_samples"]
    assert run["evaluator_motion_model"] == "speed_law_from_first_issued_response_tick"
    for row in samples:
        t = row["time_s"]
        displacement = (min(trial.v38.SECOND_RESPONSE_MAX_M,
                            max(0., plan["parameters"]["speed_mps"]*(t-first_command)))
                        if plan and first_command is not None else 0.)
        ape = start+[0., -displacement]
        person = trial.v38.human_xy(case, summary, t, run["request_time_s"],
                                     run["second_inference_start_s"])
        assert np.linalg.norm(np.asarray(row["ape_xy_m"])-ape) < .00011
        assert abs(geometry.gap(model, data, ape, person)-row["gap_m"]) < .00011
    assert run["minimum_proxy_gap_m"] == min(row["gap_m"] for row in samples)
    assert run["first_below_5cm_s"] == next((row["time_s"] for row in samples
                                             if row["gap_m"] < .05), None)
    assert run["whole_episode_minimum_proxy_gap_m"] == min(
        summary["inference_geometry_evaluator"]["minimum_gap_m"],
        run["minimum_proxy_gap_m"])
    metric = [row for row in rows if row["stage"] == "new_show_metric_item_seen"]
    if camera_alarm is not None:
        assert run["outcome"] == "new_show_flank_motion_unresolved"
        assert not metric
    elif pose_loss is not None:
        assert run["outcome"] == "new_show_pose_uncertain"
        assert not metric
    elif range_alarm is not None:
        assert run["outcome"] == "new_show_motion_unresolved_again"
        assert not metric
    elif run["outcome"] == "new_show_metric_item_seen_unattributed":
        assert len(metric) == 1
        assert metric[0]["teacher_identity"] == "unattributed_new_session"
    return {"mode": mode, "case": case,
            "outcome": run["outcome"],
            "camera_alarm_s": camera_alarm,
            "first_response_s": first_command,
            "pose_loss_s": pose_loss,
            "whole_episode_minimum_proxy_gap_m": run[
                "whole_episode_minimum_proxy_gap_m"],
            "video_frames": run["video"]["frames"],
            "sensor_frames": sensor_video["frames"]}


def main():
    report = json.loads((trial.OUT / "report.json").read_text())
    assert report["schema"] == "robot-learning-lab.ape-owner-moving-flank.v1"
    assert report["source_sha256"] == sha(trial.__file__)
    assert report["v38_source_sha256"] == sha(trial.v38.__file__)
    assert report["v37_source_sha256"] == sha(trial.v37.__file__)
    assert report["scene_sha256"] == sha(trial.v38.v35.moving.SCENE_XML)
    assert report["v36_report_sha256"] == sha(trial.v38.SOURCE)
    assert report["current_design_pointer_sha256"] == sha(
        trial.ROOT / "robot/ape-current-design.json")
    assert len(report["results"]) == len(trial.MODES)*len(trial.CASES)
    assert report["parameters"]["route_obstacle_frozen_during_inference"]
    source = trial.v38.original()
    summary = source["summary"]
    policy = json.loads(trial.v38.joined.CONFIG.read_text())
    scenario = next(x for x in policy["cases"] if x["id"] == "late_crossing")
    rows = [audit_result(item, summary, source, scenario)
            for item in report["results"]]
    assert [(row["mode"], row["case"]) for row in rows] == [
        (mode, case) for mode in trial.MODES for case in trial.CASES]
    comparison = json.loads((trial.OUT / "comparison.json").read_text())
    assert comparison["schema"] == "robot-learning-lab.ape-owner-moving-flank-comparison.v1"
    assert comparison["report_sha256"] == sha(trial.OUT / "report.json")
    assert comparison["source_sha256"] == sha(comparison_renderer.__file__)
    assert set(comparison["videos"]) == set(comparison_renderer.PAIRS)
    for name, video in comparison["videos"].items():
        assert sha(video["path"]) == video["sha256"]
        assert decoded_frames(video["path"], 390, 1280) == video["frames"]
        assert ((video["left"]["mode"], video["left"]["case"]),
                (video["right"]["mode"], video["right"]["case"])) == \
            comparison_renderer.PAIRS[name]
    audit = {"passed": True, "rows": rows,
             "comparison_videos": comparison["videos"],
             "scope": "Twelve matched owner-route continuations; exact source checkpoint, timed raw/joint/IMU packets, cue and pose-health causality, first issued response timing, independent 20ms proxy gaps and decoded route/sensor and paired videos. Rotation-only camera model and kinematic body remain unqualified."}
    (trial.OUT / "audit.json").write_text(json.dumps(audit, indent=2)+"\n")
    print(json.dumps(audit, indent=2))


if __name__ == "__main__":
    main()
