"""Audit V38 checkpoint, range-driven recovery and complete videos."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import cv2
import mujoco
import numpy as np

import audit_owner_inference_response as geometry
import owner_post_alarm_recovery as trial


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def video_frames(path):
    capture = cv2.VideoCapture(str(path))
    assert capture.isOpened(), path
    count = 0
    while True:
        okay, image = capture.read()
        if not okay:
            break
        assert image is not None and image.shape == (528, 1040, 3)
        count += 1
    capture.release()
    return count


def sensor_video_frames(path):
    capture = cv2.VideoCapture(str(path))
    assert capture.isOpened(), path
    count = 0
    while True:
        okay, image = capture.read()
        if not okay:
            break
        assert image is not None and image.shape[0] > 0 and image.shape[1] > 0
        count += 1
    capture.release()
    return count


def audit_case(result, summary, scenario):
    case = result["case"]
    rows = json.loads(Path(result["trace"]["path"]).read_text())
    assert sha(result["trace"]["path"]) == result["trace"]["sha256"]
    assert sha(result["video"]["path"]) == result["video"]["sha256"]
    assert video_frames(result["video"]["path"]) == result["video"]["frames"]
    assert result["video"]["frames"] > result["source_video_frames"]
    model = mujoco.MjModel.from_xml_path(str(trial.v35.moving.SCENE_XML))
    data = mujoco.MjData(model)
    actual, estimate, checkpoint_s, source_hash, checkpoint_packet = \
        trial.geometry_at(summary, trial.original()["result"]["result"]["episode"],
                          scenario, model, data)
    assert result["checkpoint_range_packet_exact"]
    assert result["checkpoint_time_s"] == checkpoint_s
    assert result["source_v36_report_sha256"] == source_hash
    assert rows[0]["checkpoint_range_packet"] == checkpoint_packet
    assert rows[0]["old_owner_identity"] == "unknown"
    hold = [row for row in rows if row["stage"] == "post_alarm_range_watch"]
    assert hold and abs(hold[0]["time_s"]-checkpoint_s) < .00001
    clear_streak = 0
    resume = None
    for row in hold:
        packet = row["actor_packet"]
        assert len(packet) == 72
        healthy = trial.side_healthy(packet)
        nearest = trial.v35.side_range(packet) if healthy else None
        assert healthy == row["side_health"]
        assert nearest == row["nearest_side_range_m"]
        clear = (healthy and nearest is not None and
                 nearest >= trial.CLEAR_SIDE_RANGE_M)
        clear_streak = clear_streak+1 if clear else 0
        assert clear_streak == row["clear_streak"]
        if clear_streak >= trial.CLEAR_STREAK and resume is None:
            resume = row["time_s"]
            assert row["actor_action"] == "resume_after_three_fresh_clear_packets"
        elif not healthy:
            assert row["actor_action"] == "hold_and_request_range_recovery"
        else:
            assert row["actor_action"] == "hold_and_observe"
    assert resume == result["resume_time_s"]
    fresh = [row for row in rows if row["stage"] == "fresh_new_show_capture"]
    assert bool(fresh) == (resume is not None)
    if fresh:
        assert fresh[0]["time_s"] > resume
        assert fresh[0]["old_owner_identity"] == "unknown"
        assert sha(fresh[0]["raw_rgbd"]["path"]) == fresh[0]["raw_rgbd"]["sha256"]
    watch = [row for row in rows if row["stage"] == "second_inference_range_watch"]
    assert bool(watch) == (resume is not None)
    risk_streak = 0
    camera_streak = 0
    second_alarm = None
    camera_alarm = None
    for row in watch:
        healthy = trial.side_healthy(row["actor_packet"])
        nearest = trial.v35.side_range(row["actor_packet"]) if healthy else None
        assert row["side_health"] == healthy
        assert row["nearest_side_range_m"] == nearest
        risk = healthy and nearest is not None and nearest < trial.CLEAR_SIDE_RANGE_M
        risk_streak = risk_streak+1 if risk else 0
        assert row["risk_streak"] == risk_streak
        if risk_streak >= 2 and second_alarm is None:
            second_alarm = row["time_s"]
        flank = row["flank_rgbd_packet"]
        assert (flank is not None) == (case in trial.FLANK_MONITOR)
        camera_streak = (camera_streak+1 if flank and flank["actor_motion_cue"]
                         else 0)
        assert row["camera_motion_streak"] == camera_streak
        if camera_streak >= 2 and camera_alarm is None:
            camera_alarm = row["time_s"]
    inference = next((row for row in rows
                      if row["stage"] == "second_inference_result"), None)
    if watch:
        assert inference is not None
        assert (inference["alarm_s"] is None) == (second_alarm is None)
        if second_alarm is not None:
            assert abs(inference["alarm_s"]-second_alarm) < .00011
        assert (inference["camera_alarm_s"] is None) == (camera_alarm is None)
        if camera_alarm is not None:
            assert abs(inference["camera_alarm_s"]-camera_alarm) < .00011
    plans = [row for row in rows if row["stage"] == "second_response_plan"]
    assert bool(plans) == (case in trial.FLANK_RESPONSE)
    plan = plans[0] if plans else None
    response_start = inference["first_response_s"] if inference else None
    if plan:
        assert abs(plan["alarm_s"]-camera_alarm) < .00011
        alarm_packet = next(row["actor_packet"] for row in watch
                            if abs(row["time_s"]-camera_alarm) < .00011)
        assert plan["south_range_clear"] == trial.south_range_clear(alarm_packet)
        assert plan["parameters"] == trial.FLANK_RESPONSE[case]
        if plan["south_range_clear"]:
            assert response_start is not None
            assert response_start >= camera_alarm+plan["parameters"]["delay_s"]
            assert any(abs(row["time_s"]-response_start) < .00011
                       for row in watch)
        else:
            assert response_start is None
    else:
        assert response_start is None
    start_xy = np.asarray(summary["ape_after_episode_xy_m"])
    for row in watch:
        displacement = (min(trial.SECOND_RESPONSE_MAX_M,
                            max(0., plan["parameters"]["speed_mps"] *
                                (row["time_s"]-response_start)))
                        if plan and response_start is not None else 0.)
        assert np.linalg.norm(np.asarray(row["ape_xy_m"])-
                              (start_xy+[0., -displacement])) < .00011
    sensor_videos = [row for row in rows if row["stage"] == "flank_sensor_video"]
    assert bool(sensor_videos) == (case in trial.FLANK_MONITOR)
    if sensor_videos:
        assert sha(sensor_videos[0]["path"]) == sensor_videos[0]["sha256"]
        assert sensor_video_frames(sensor_videos[0]["path"]) == sensor_videos[0]["frames"]
    samples = result["evaluator_gap_samples"]
    assert result["evaluator_motion_model"] == "speed_law_from_first_issued_response_tick"
    assert samples and result["minimum_proxy_gap_m"] == min(
        row["gap_m"] for row in samples)
    assert result["first_below_5cm_s"] == next((
        row["time_s"] for row in samples if row["gap_m"] < .05), None)
    if case in ("blind_return_no_contact", "blind_return_contact",
                "blind_return_flank_rgbd"):
        assert result["minimum_proxy_gap_m"] < 0
        assert result["first_below_5cm_s"] is not None
    elif case == "blind_return_flank_fast_response":
        assert result["minimum_proxy_gap_m"] >= .05
    elif case == "blind_return_flank_slow_response":
        assert 0 < result["minimum_proxy_gap_m"] < .05
    else:
        assert result["minimum_proxy_gap_m"] >= .05
    assert result["whole_episode_minimum_proxy_gap_m"] == min(
        summary["inference_geometry_evaluator"]["minimum_gap_m"],
        result["minimum_proxy_gap_m"])
    for row in samples:
        person = trial.human_xy(case, summary, row["time_s"],
                                result["request_time_s"],
                                result["second_inference_start_s"])
        displacement = (min(trial.SECOND_RESPONSE_MAX_M,
                            max(0., plan["parameters"]["speed_mps"] *
                                (row["time_s"]-response_start)))
                        if plan and response_start is not None else 0.)
        ape = start_xy+[0., -displacement]
        assert np.linalg.norm(ape-np.asarray(row["ape_xy_m"])) < .00011
        assert abs(geometry.gap(model, data, ape, person)-row["gap_m"]) < .00011
    metric = [row for row in rows if row["stage"] == "new_show_metric_item_seen"]
    if case == "complies":
        assert result["outcome"] == "new_show_metric_item_seen_unattributed"
        assert len(metric) == 1
        assert metric[0]["teacher_identity"] == "unattributed_new_session"
        checks = [row for row in rows if row["stage"] == "fresh_new_show_check"]
        assert len(checks) == 2 and all(row["accepted"] for row in checks)
        assert np.linalg.norm(np.asarray(checks[0]["point_xyz_est_m"])-
                              np.asarray(checks[1]["point_xyz_est_m"])) <= .10
        assert metric[0]["item_position_error_evaluator_m"] < .05
        assert second_alarm is None
    elif case != "blind_return_no_contact":
        assert not metric
    if case == "returns_during_new_show":
        assert second_alarm is not None
        assert result["outcome"] == "new_show_motion_unresolved_again"
        assert second_alarm < inference["time_s"]
    if case == "ignores":
        assert result["outcome"] == "waiting_for_person_space"
        assert resume is None
    if case == "side_range_lost":
        assert result["outcome"] == "sensor_recovery_requested"
        assert resume is None
        assert all(not row["side_health"] for row in hold)
    if case in trial.BLIND_RETURN:
        assert resume is not None
        assert second_alarm is None
        assert inference["loss_s"] is None
        assert all(row["nearest_side_range_m"] >= trial.CLEAR_SIDE_RANGE_M
                   for row in watch)
        if case == "blind_return_no_contact":
            first_overlap = next(row["time_s"] for row in samples
                                 if row["gap_m"] <= 0)
            assert inference["contact_s"] is None
            assert all(row["actor_contact_packet"] is None for row in watch)
            assert result["outcome"] == "new_show_metric_item_seen_unattributed"
            assert len(metric) == 1 and metric[0]["time_s"] > first_overlap
        elif case == "blind_return_contact":
            first_overlap = next(row["time_s"] for row in samples
                                 if row["gap_m"] <= 0)
            assert result["outcome"] == "contact_detected_after_overlap"
            assert not metric
            assert inference["contact_s"] >= first_overlap
            assert any(row["actor_contact_packet"] for row in watch)
        else:
            assert inference["contact_s"] is None
            assert camera_alarm is not None and camera_alarm < inference["time_s"]
            assert result["outcome"] == "new_show_flank_motion_unresolved"
            assert not metric
            assert all(row["actor_contact_packet"] is None for row in watch)
    return {"case": case, "outcome": result["outcome"],
            "resume_time_s": resume,
            "second_alarm_s": second_alarm,
            "camera_alarm_s": camera_alarm,
            "first_response_s": response_start,
            "contact_s": inference["contact_s"] if inference else None,
            "minimum_proxy_gap_m": result["minimum_proxy_gap_m"],
            "whole_episode_minimum_proxy_gap_m": result[
                "whole_episode_minimum_proxy_gap_m"],
            "video_frames": result["video"]["frames"]}


def main():
    report = json.loads((trial.OUT / "report.json").read_text())
    assert report["schema"] == "robot-learning-lab.ape-owner-post-alarm-recovery.v1"
    assert report["source_sha256"] == sha(trial.__file__)
    assert report["v36_report_sha256"] == sha(trial.SOURCE)
    assert report["scene_sha256"] == sha(trial.v35.moving.SCENE_XML)
    assert report["current_design_pointer_sha256"] == sha(
        trial.ROOT / "robot/ape-current-design.json")
    assert len(report["results"]) == len(trial.CASES)
    source = trial.original()
    summary = source["summary"]
    policy = json.loads(trial.joined.CONFIG.read_text())
    scenario = next(x for x in policy["cases"] if x["id"] == "late_crossing")
    rows = [audit_case(result, summary, scenario) for result in report["results"]]
    assert [row["case"] for row in rows] == list(trial.CASES)
    audit = {"passed": True, "rows": rows,
             "scope": "Exact V36 sensor checkpoint, post-alarm packet decisions, fresh item checks, independent 20ms proxy-gap score from first issued response tick, full concatenated videos and flank sensor video. Blind-return range faults, camera alarm, kinematic fast/slow responses and an ideal contact bit compare avoidance versus post-contact abort. Human response and ape motion remain scripted/kinematic."}
    (trial.OUT / "audit.json").write_text(json.dumps(audit, indent=2)+"\n")
    print(json.dumps(audit, indent=2))


if __name__ == "__main__":
    main()
