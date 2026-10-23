"""Audit V45 sensor inputs, pre-show commands and full swept geometry."""

from __future__ import annotations

import json
from pathlib import Path

import mujoco
import numpy as np

import audit_owner_joint_hazard_response as prior
import audit_owner_scanned_meeting_pose as v44_audit
import owner_sensor_fault_world_stress as trial


def check(item, report, source, scenario):
    run = item["run"]
    config = trial.PROFILES[item["profile"]]
    assert prior.sha(run["trace"]["path"]) == run["trace"]["sha256"]
    assert prior.sha(run["video"]["path"]) == run["video"]["sha256"]
    assert prior.video_frames(run["video"]["path"], (528, 1040, 3)) == \
        run["video"]["frames"]
    rows = json.loads(Path(run["trace"]["path"]).read_text())
    summary = source["summary"]
    model = mujoco.MjModel.from_xml_path(str(trial.v38.v35.moving.SCENE_XML))
    data = mujoco.MjData(model)
    start, _, t0, source_hash, endpoint = trial.v38.geometry_at(
        summary, source["result"]["result"]["episode"], scenario,
        model, data)
    assert t0 == report["checkpoint_s"] == run["checkpoint_time_s"]
    assert source_hash == run["source_v36_report_sha256"]
    assert rows[0]["checkpoint_range_packet"] == endpoint
    resume = next(row for row in rows if row["stage"] ==
                  "post_alarm_range_watch" and row["actor_action"] ==
                  "resume_after_three_fresh_clear_packets")
    assert abs(resume["time_s"]-run["resume_time_s"]) < 1e-5
    steps = [row for row in rows if row["stage"] ==
             "pre_show_restage_step"]
    assert len(steps) == item["restage_steps"]
    assert steps
    scores = v44_audit.independent_candidates(resume["actor_packet"], start)
    choice = max((name for name, score in scores.items() if score["eligible"]),
                 key=lambda name: scores[name]["robust_score_m"])
    assert choice == item["restage_choice"]
    previous = resume
    previous_pose = np.asarray(start)
    for row in steps:
        pose = np.asarray(row["ape_xy_m"])
        step_m = row["commanded_step_m"]
        assert abs(row["time_s"]-previous["time_s"]-
                   trial.v38.HOLD_PERIOD_S) < 1e-5
        assert np.linalg.norm(pose-(previous_pose+
                                  trial.v44.CANDIDATES[choice]*step_m)) < 1e-8
        points = trial.v41.JointHazardResponse.endpoints(
            previous["actor_packet"], previous_pose)
        minimum = float(np.min(np.linalg.norm(points-pose, axis=1)))
        assert abs(minimum-row["actor_decision"][
            "minimum_endpoint_gap_m"]) < 1e-8
        assert minimum >= trial.v44.MIN_ENDPOINT_GAP_M
        previous, previous_pose = row, pose
    if config.get("range_fault") == "side_lost":
        assert len(steps) == 1
        assert run["outcome"] == "no_supported_meeting_pose"
        assert run["second_inference_start_s"] is None
        assert not any(row["stage"] == "fresh_new_show_capture" for row in rows)
        packet = steps[-1]["actor_packet"]
        assert not trial.v38.side_healthy(packet)
        assert next(row for row in rows if row["stage"] ==
                    "pre_show_restage_decision")["actor_decision"][
                        "action"] == "hold_scan_unhealthy"
    watch = [row for row in rows if row["stage"] ==
             "second_inference_range_watch"]
    if watch:
        sensor = next(row for row in rows if row["stage"] ==
                      "flank_sensor_video")
        assert prior.sha(sensor["path"]) == sensor["sha256"]
        assert prior.video_frames(sensor["path"], (480, 640, 3)) == \
            sensor["frames"]
        packets = [row["flank_rgbd_packet"] for row in watch if row[
            "flank_rgbd_packet"] is not None]
        assert packets
        if config["flank"] == "raw_rgbd":
            assert all(packet["joint_head_yaw_deg"] is None and
                       packet["imu_body_yaw_deg"] is None
                       for packet in packets)
        elif config["flank"] == "imu_lag_gated":
            assert all(packet["imu_age_s"] >= .1-.0001 and
                       not packet["actor_pose_healthy"]
                       for packet in packets)
        else:
            assert all(packet["actor_pose_healthy"] for packet in packets)
    timeline = [(t0, np.asarray(start)),
                (resume["time_s"], np.asarray(start))]
    timeline.extend((row["time_s"], np.asarray(row["ape_xy_m"]))
                    for row in steps+watch)
    times = np.asarray([row[0] for row in timeline])
    poses = np.asarray([row[1] for row in timeline])
    response_plan = next((row for row in rows if row["stage"] ==
                          "second_response_plan"), None)
    inference = next((row for row in rows if row["stage"] ==
                      "second_inference_result"), None)
    person_min, prop_min = float("inf"), float("inf")
    for sample in run["evaluator_gap_samples"]:
        t = sample["time_s"]
        if response_plan is not None and not response_plan["adaptive_controller"] \
                and inference["first_response_s"] is not None and \
                t >= inference["first_response_s"]:
            speed = response_plan["parameters"]["speed_mps"]
            travel = min(.25, max(0., speed*(t-inference["first_response_s"])))
            if not response_plan["south_range_clear"]:
                travel = 0.
            if inference["response_halt_displacement_m"] is not None:
                travel = min(travel,
                             inference["response_halt_displacement_m"])
            ape = np.asarray(steps[-1]["ape_xy_m"])+[0., -travel]
        else:
            ape = np.asarray([np.interp(t, times, poses[:, axis])
                              for axis in range(2)])
        assert np.linalg.norm(ape-sample["ape_xy_m"]) < .00011
        person = trial.v38.human_xy(
            item["case"], summary, t, run["request_time_s"],
            run["second_inference_start_s"])
        model.body_pos[model.body("route_obstacle").id, :2] = \
            trial.v41.v40.prop_at(t, t0,
                                  trial.PROP_CONFIGS[config["prop"]])
        person_gap = trial.v38.geometry.gap(model, data, ape, person)
        prop_gap = trial.v38.prop_gap(model, data, ape)
        assert abs(person_gap-sample["gap_m"]) < .00011
        assert abs(prop_gap-sample["prop_gap_m"]) < .00011
        person_min = min(person_min, person_gap)
        prop_min = min(prop_min, prop_gap)
    assert abs(person_min-run["minimum_proxy_gap_m"]) < .00011
    assert abs(prop_min-run["minimum_prop_gap_m"]) < .00011
    return {"profile": item["profile"], "case": item["case"],
            "outcome": run["outcome"], "choice": choice,
            "minimum_person_gap_m": round(person_min, 5),
            "minimum_prop_gap_m": round(prop_min, 5),
            "above_provisional_5cm_both": person_min >= .05 and prop_min >= .05,
            "restage_steps": len(steps),
            "full_video_frames": run["video"]["frames"]}


def main():
    report = json.loads((trial.OUT / "report.json").read_text())
    assert report["schema"] == \
        "robot-learning-lab.ape-owner-sensor-fault-world-stress.v1"
    assert report["source_sha256"] == prior.sha(trial.__file__)
    assert report["v38_source_sha256"] == prior.sha(trial.v38.__file__)
    assert report["v39_source_sha256"] == prior.sha(trial.v39.__file__)
    assert report["v44_source_sha256"] == prior.sha(trial.v44.__file__)
    assert report["scene_sha256"] == prior.sha(trial.v38.v35.moving.SCENE_XML)
    assert report["v36_report_sha256"] == prior.sha(trial.v38.SOURCE)
    assert report["current_design_pointer_sha256"] == prior.sha(
        trial.ROOT / "robot/ape-current-design.json")
    source = trial.v38.original()
    policy = json.loads(trial.v38.joined.CONFIG.read_text())
    scenario = next(row for row in policy["cases"] if row["id"] ==
                    "late_crossing")
    checked = [check(item, report, source, scenario)
               for item in report["results"]]
    audit = {"passed": True, "rows": checked,
             "scope": "Independent candidate and prior-packet step replay, 20 ms MuJoCo person/prop gap recomputation, selected sensor-health checks, full scene/flank video decoding. Historical kinematic proxy and scripted faults; no physical safety qualification."}
    path = trial.OUT / "audit.json"
    path.write_text(json.dumps(audit, indent=2)+"\n")
    print(path)


if __name__ == "__main__":
    main()
