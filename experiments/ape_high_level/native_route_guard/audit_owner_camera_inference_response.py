"""Recount V36 camera decisions, complete episodes, videos and proxy gaps."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import cv2
import mujoco
import numpy as np

import audit_owner_inference_response as base
import owner_camera_inference_response as trial


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def frames(path):
    cap = cv2.VideoCapture(str(path))
    assert cap.isOpened(), path
    count = 0
    while True:
        good, frame = cap.read()
        if not good:
            break
        assert frame is not None and frame.size
        count += 1
    cap.release()
    return count


def audit_outcome(outcome):
    sensor, condition = outcome["sensor"], outcome["condition"]
    summary = outcome["summary"]
    episode = outcome["result"]["result"]["episode"]
    assert summary["condition"] == condition
    assert summary["inference_sensor"] == sensor
    assert summary["outcome"] == episode["outcome"]
    assert summary["wait_choice"]["action"] == (
        "hold_current_pose" if condition == "wide_no_entry"
        else "move_south_then_show")
    assert not any(row.get("teacher_identity") == "confirmed_old_owner"
                   for row in episode["trace"])
    probes = summary["inference_probe_samples"]
    changes = summary["range_changes"]
    packets = summary["range_packets"]
    guard = next(row for row in episode["trace"]
                 if row["stage"] == "new_show_inference_guard")
    assert len(probes) == len(changes) == len(packets) == len(guard["samples"]) > 0
    streak = 0
    alarm = None
    for probe, change, packet, sample in zip(
            probes, changes, packets, guard["samples"]):
        assert abs(probe["time_s"]-packet["time_s"]) < 1e-8
        assert abs(change["time_s"]-packet["time_s"]) < 1e-8
        assert probe["actor_motion_cue"] == (
            probe["rgb_changed_pixels"] >= trial.MOTION_PIXELS and
            probe["depth_changed_pixels"] >= trial.MOTION_PIXELS)
        used = 3 if probe["actor_motion_cue"] else 0
        assert change["actor_used_changed"] == used
        assert sample["changed_owner_sector_beams"] == used
        streak = streak+1 if used >= 3 else 0
        assert sample["motion_streak"] == streak
        if streak >= 2 and alarm is None:
            alarm = sample["time_s"]
    assert alarm == guard["alarm_time_s"] == summary["guard_alarm_time_s"]
    if alarm is None:
        assert summary["actor_alarm_s"] is None
        assert summary["first_action_s"] is None
        assert not summary["response_path"]
        assert summary["outcome"] == "new_show_metric_item_seen_unattributed"
    else:
        assert abs(summary["actor_alarm_s"]-alarm) < .00011
        assert summary["response_plan"]["known_free"]
        assert summary["first_action_s"] is not None
        assert summary["outcome"] == "new_show_motion_unresolved"
        assert any(row["stage"] == "new_show_stale_image_discarded"
                   for row in episode["trace"])
        for row in summary["response_path"]:
            assert abs(row["displacement_m"] -
                       base.expected_displacement(summary, row["time_s"])) < .00002
    geometry = summary["inference_geometry_evaluator"]
    model = mujoco.MjModel.from_xml_path(str(trial.v35.moving.SCENE_XML))
    data = mujoco.MjData(model)
    target = np.asarray(summary["bystander_target_evaluator_xy_m"])
    for row in geometry["samples"]:
        t = row["time_s"]
        if condition == "wide_no_entry":
            bystander = np.asarray(summary["bystander_stage_start_evaluator_xy_m"])
        elif condition == "wide_buffered_static":
            bystander = trial.v35.SIDE_START
        else:
            fraction = min(1., trial.v35.APPROACH_MPS *
                           (t-summary["worker_start_s"]) /
                           np.linalg.norm(target-trial.v35.SIDE_START))
            bystander = trial.v35.SIDE_START + fraction*(target-trial.v35.SIDE_START)
        assert abs(base.gap(model, data, row["ape_xy_m"], bystander)-
                   row["gap_m"]) < .00011
    assert geometry["minimum_gap_m"] == min(
        row["gap_m"] for row in geometry["samples"])
    assert geometry["minimum_gap_m"] >= .05
    sensor_video = summary["sensor_video"]
    episode_video = episode["video"]
    assert sha(sensor_video["path"]) == sensor_video["sha256"]
    assert frames(sensor_video["path"]) == sensor_video["frames"] == len(probes)+1
    assert sha(episode_video["path"]) == episode_video["sha256"]
    assert frames(episode_video["path"]) == episode_video["frames"]
    route_hash = hashlib.sha256(json.dumps(
        episode["route_episode"]["trace"], sort_keys=True).encode()).hexdigest()
    return {"sensor": sensor, "condition": condition,
            "outcome": summary["outcome"],
            "alarm_after_motion_start_s": (None if alarm is None else
                round(alarm-summary["worker_start_s"], 3)),
            "first_action_after_alarm_s": (None if alarm is None else
                round(summary["first_action_s"]-summary["actor_alarm_s"], 3)),
            "minimum_proxy_gap_m": geometry["minimum_gap_m"],
            "sensor_frames": sensor_video["frames"],
            "episode_frames": episode_video["frames"],
            "route_hash": route_hash}


def main():
    report = json.loads((trial.OUT / "report.json").read_text())
    assert report["schema"] == "robot-learning-lab.ape-owner-camera-inference-response.v1"
    assert report["source_sha256"] == sha(trial.__file__)
    assert report["v35_base_source_sha256"] == sha(trial.v35.__file__)
    assert report["scene_sha256"] == sha(trial.v35.moving.SCENE_XML)
    assert report["current_design_pointer_sha256"] == sha(
        trial.ROOT / "robot/ape-current-design.json")
    assert len(report["outcomes"]) == 6
    rows = [audit_outcome(outcome) for outcome in report["outcomes"]]
    assert len({row["route_hash"] for row in rows}) == 1
    side_packets = [outcome["summary"]["wait_choice"]["range_packet"]
                    for outcome in report["outcomes"]
                    if outcome["condition"] != "wide_no_entry"]
    assert all(packet == side_packets[0] for packet in side_packets)
    for sensor in trial.SENSORS:
        by = {row["condition"]: row for row in rows if row["sensor"] == sensor}
        assert set(by) == set(trial.CASES)
        assert by["wide_respond_fast"]["alarm_after_motion_start_s"] is not None
        assert by["wide_buffered_static"]["alarm_after_motion_start_s"] is None
        assert by["wide_no_entry"]["alarm_after_motion_start_s"] is None
    comparison = json.loads((trial.OUT / "comparison.json").read_text())
    assert comparison["report_sha256"] == sha(trial.OUT / "report.json")
    assert sha(comparison["video"]["path"]) == comparison["video"]["sha256"]
    assert frames(comparison["video"]["path"]) == comparison["video"]["frames"]
    audit = {"passed": True, "rows": rows,
             "comparison": comparison["video"],
             "scope": "Timed RGB-D motion packet decisions, route equality, complete owner and sensor videos, and recomputed kinematic person gaps; does not qualify a physical camera or body action."}
    (trial.OUT / "audit.json").write_text(json.dumps(audit, indent=2)+"\n")
    print(json.dumps(audit, indent=2))


if __name__ == "__main__":
    main()
