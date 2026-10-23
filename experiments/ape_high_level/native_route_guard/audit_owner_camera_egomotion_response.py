"""Audit V37 moving-camera packet causality, false alarms and full episodes."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import mujoco
import numpy as np

import audit_owner_camera_inference_response as camera_audit
import audit_owner_inference_response as base
import owner_camera_egomotion_response as trial


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def audit_outcome(outcome):
    mode, condition = outcome["mode"], outcome["condition"]
    summary = outcome["summary"]
    episode = outcome["result"]["result"]["episode"]
    assert summary["condition"] == condition
    assert summary["inference_sensor"] == f"owner_rgbd_{mode}"
    assert summary["outcome"] == episode["outcome"]
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
        assert abs(probe["simulated_head_yaw_deg"]) <= trial.SWAY_DEG+.00001
        assert probe["reported_yaw_bias_deg"] == (
            trial.BIAS_DEG if mode == "yaw_biased_half_degree" else 0.)
        cue = (probe["rgb_changed_pixels"] >= trial.RGB_MOTION_PIXELS and
               probe["depth_changed_pixels"] >= trial.DEPTH_MOTION_PIXELS)
        assert probe["actor_motion_cue"] == cue
        used = 3 if cue else 0
        assert change["actor_used_changed"] == used
        assert sample["changed_owner_sector_beams"] == used
        streak = streak+1 if cue else 0
        assert sample["motion_streak"] == streak
        if streak >= 2 and alarm is None:
            alarm = sample["time_s"]
    assert alarm == guard["alarm_time_s"] == summary["guard_alarm_time_s"]
    if alarm is not None:
        assert abs(alarm-summary["actor_alarm_s"]) < .00011
        assert summary["outcome"] == "new_show_motion_unresolved"
        assert summary["response_plan"]["known_free"]
        assert summary["first_action_s"] is not None if condition == "wide_respond_fast" else summary["first_action_s"] is None
        assert any(row["stage"] == "new_show_stale_image_discarded"
                   for row in episode["trace"])
    else:
        assert summary["actor_alarm_s"] is None
        assert summary["outcome"] == "new_show_metric_item_seen_unattributed"
        assert summary["first_action_s"] is None
    geometry = summary["inference_geometry_evaluator"]
    model = mujoco.MjModel.from_xml_path(str(trial.v36.v35.moving.SCENE_XML))
    data = mujoco.MjData(model)
    target = np.asarray(summary["bystander_target_evaluator_xy_m"])
    for row in geometry["samples"]:
        t = row["time_s"]
        if condition == "wide_no_entry":
            bystander = np.asarray(summary["bystander_stage_start_evaluator_xy_m"])
        elif condition == "wide_buffered_static":
            bystander = trial.v36.v35.SIDE_START
        else:
            fraction = min(1., trial.v36.v35.APPROACH_MPS *
                           (t-summary["worker_start_s"]) /
                           np.linalg.norm(target-trial.v36.v35.SIDE_START))
            bystander = trial.v36.v35.SIDE_START + fraction*(target-trial.v36.v35.SIDE_START)
        assert abs(base.gap(model, data, row["ape_xy_m"], bystander)-
                   row["gap_m"]) < .00011
    assert geometry["minimum_gap_m"] == min(
        row["gap_m"] for row in geometry["samples"])
    assert geometry["minimum_gap_m"] >= .05
    for video in (summary["sensor_video"], episode["video"]):
        assert sha(video["path"]) == video["sha256"]
        assert camera_audit.frames(video["path"]) == video["frames"]
    assert summary["sensor_video"]["frames"] == len(probes)+1
    route_hash = hashlib.sha256(json.dumps(
        episode["route_episode"]["trace"], sort_keys=True).encode()).hexdigest()
    return {"mode": mode, "condition": condition,
            "outcome": summary["outcome"],
            "alarm_after_motion_start_s": (None if alarm is None else
                round(alarm-summary["worker_start_s"], 3)),
            "minimum_proxy_gap_m": geometry["minimum_gap_m"],
            "sensor_frames": summary["sensor_video"]["frames"],
            "episode_frames": episode["video"]["frames"],
            "route_hash": route_hash}


def main():
    report = json.loads((trial.OUT / "report.json").read_text())
    assert report["schema"] == "robot-learning-lab.ape-owner-camera-egomotion-response.v1"
    assert report["source_sha256"] == sha(trial.__file__)
    assert report["v36_source_sha256"] == sha(trial.v36.__file__)
    assert report["v35_source_sha256"] == sha(trial.v36.v35.__file__)
    assert report["scene_sha256"] == sha(trial.v36.v35.moving.SCENE_XML)
    assert report["current_design_pointer_sha256"] == sha(
        trial.ROOT / "robot/ape-current-design.json")
    assert len(report["outcomes"]) == 9
    rows = [audit_outcome(outcome) for outcome in report["outcomes"]]
    assert len({row["route_hash"] for row in rows}) == 1
    side_packets = [outcome["summary"]["wait_choice"]["range_packet"]
                    for outcome in report["outcomes"]
                    if outcome["condition"] != "wide_no_entry"]
    assert all(packet == side_packets[0] for packet in side_packets)
    for mode in trial.MODES:
        by = {row["condition"]: row for row in rows if row["mode"] == mode}
        assert set(by) == set(trial.CASES)
        assert by["wide_respond_fast"]["alarm_after_motion_start_s"] is not None
        for condition in ("wide_buffered_static", "wide_no_entry"):
            assert (by[condition]["alarm_after_motion_start_s"] is None) == (
                mode == "yaw_compensated")
    comparison = json.loads((trial.OUT / "comparison.json").read_text())
    assert comparison["report_sha256"] == sha(trial.OUT / "report.json")
    for video in comparison["videos"].values():
        assert sha(video["path"]) == video["sha256"]
        assert camera_audit.frames(video["path"]) == video["frames"]
    audit = {"passed": True, "rows": rows,
             "comparison_videos": comparison["videos"],
             "scope": "Nine complete owner episodes with moving-camera RGB-D cue, ideal or erroneous relative-yaw compensation, packet decisions, independent proxy gaps and decoded media. Optical/shell and physical response qualification remain open."}
    (trial.OUT / "audit.json").write_text(json.dumps(audit, indent=2)+"\n")
    print(json.dumps(audit, indent=2))


if __name__ == "__main__":
    main()
