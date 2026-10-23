"""Independent V47 scan-history, executed-step and full-horizon audit."""

from __future__ import annotations

import json
from pathlib import Path

import mujoco
import numpy as np

import audit_owner_joint_hazard_response as prior
import owner_history_meeting_pose as trial


def check(item, report, source, scenario):
    run = item["run"]
    rows = json.loads(Path(run["trace"]["path"]).read_text())
    assert prior.sha(run["trace"]["path"]) == run["trace"]["sha256"]
    assert prior.sha(run["video"]["path"]) == run["video"]["sha256"]
    assert prior.video_frames(run["video"]["path"], (528, 1040, 3)) == \
        run["video"]["frames"]
    model = mujoco.MjModel.from_xml_path(str(trial.v38.v35.moving.SCENE_XML))
    data = mujoco.MjData(model)
    summary = source["summary"]
    start, _, t0, source_hash, endpoint = trial.v38.geometry_at(
        summary, source["result"]["result"]["episode"], scenario,
        model, data)
    assert t0 == report["checkpoint_s"] == run["checkpoint_time_s"]
    assert source_hash == run["source_v36_report_sha256"]
    assert rows[0]["checkpoint_range_packet"] == endpoint
    hold = [row for row in rows if row["stage"] == "post_alarm_range_watch"]
    resume = next(row for row in hold if row["actor_action"] ==
                  "resume_after_three_fresh_clear_packets")
    assert hold[-1] == resume and len(hold) >= 2
    # Re-derive the trigger from the actor's stationary packets, without
    # using the prop or person trajectory.
    previous_packet = hold[-2]["actor_packet"]
    current_packet = hold[-1]["actor_packet"]
    assert all(0 <= ray["bearing_deg"] <= 30 and
               ray["previous_m"]-ray["current_m"] >= .12 and
               ray["current_m"] <= 1.5
               for ray in item["novel_forward_returns"])
    expected = trial.novel_forward_returns(previous_packet, current_packet)
    assert expected == item["novel_forward_returns"]
    score = trial.v44.endpoint_score(current_packet, start, trial.NORTHEAST)
    assert score["eligible"]
    assert item["choice"] == "northeast"
    steps = [row for row in rows if row["stage"] == "pre_show_restage_step"]
    assert len(steps) == item["restage_steps"] and steps
    old_pose = np.asarray(start)
    old_row = resume
    for row in steps:
        pose = np.asarray(row["ape_xy_m"])
        distance = row["commanded_step_m"]
        assert abs(row["time_s"]-old_row["time_s"]-
                   trial.v38.HOLD_PERIOD_S) < 1e-5
        assert 0 < distance <= trial.v44.SPEED_MPS*trial.v38.HOLD_PERIOD_S+1e-9
        assert np.linalg.norm(pose-(old_pose+trial.NORTHEAST*distance)) < 1e-8
        assert row["actor_decision"]["novel_forward_returns"] == expected
        assert row["actor_decision"]["range_history_only"]
        points = trial.v41.JointHazardResponse.endpoints(
            old_row["actor_packet"], old_pose)
        minimum = float(np.min(np.linalg.norm(points-pose, axis=1)))
        assert abs(minimum-row["actor_decision"][
            "minimum_endpoint_gap_m"]) < 1e-8
        assert minimum >= trial.v44.MIN_ENDPOINT_GAP_M
        old_row, old_pose = row, pose
    assert np.linalg.norm(old_pose-start) >= trial.v44.TARGET_M-.001
    watch = [row for row in rows if row["stage"] ==
             "second_inference_range_watch"]
    if watch:
        sensor = next(row for row in rows if row["stage"] ==
                      "flank_sensor_video")
        assert prior.sha(sensor["path"]) == sensor["sha256"]
        assert prior.video_frames(sensor["path"], (480, 640, 3)) == \
            sensor["frames"]
        packets = [row["flank_rgbd_packet"] for row in watch
                   if row["flank_rgbd_packet"] is not None]
        assert packets and all(packet["actor_pose_healthy"] for packet in packets)
    assert run["second_inference_start_s"] is not None
    timeline = [(t0, np.asarray(start)), (resume["time_s"], np.asarray(start))]
    timeline.extend((row["time_s"], np.asarray(row["ape_xy_m"]))
                    for row in steps+watch)
    times = np.asarray([row[0] for row in timeline])
    poses = np.asarray([row[1] for row in timeline])
    person_min, prop_min = float("inf"), float("inf")
    config = report["prop_configs"][item["prop_case"]]
    for sample in run["evaluator_gap_samples"]:
        t = sample["time_s"]
        ape = np.asarray([np.interp(t, times, poses[:, axis])
                          for axis in range(2)])
        assert np.linalg.norm(ape-sample["ape_xy_m"]) < .00011
        person = trial.v38.human_xy(item["case"], summary, t,
                                    run["request_time_s"],
                                    run["second_inference_start_s"])
        model.body_pos[model.body("route_obstacle").id, :2] = \
            trial.v41.v40.prop_at(t, t0, config)
        person_gap = trial.v38.geometry.gap(model, data, ape, person)
        prop_gap = trial.v38.prop_gap(model, data, ape)
        assert abs(person_gap-sample["gap_m"]) < .00011
        assert abs(prop_gap-sample["prop_gap_m"]) < .00011
        person_min = min(person_min, person_gap)
        prop_min = min(prop_min, prop_gap)
    assert abs(person_min-run["minimum_proxy_gap_m"]) < .00011
    assert abs(prop_min-run["minimum_prop_gap_m"]) < .00011
    return {"prop_case": item["prop_case"], "case": item["case"],
            "outcome": run["outcome"],
            "minimum_person_gap_m": round(person_min, 5),
            "minimum_prop_gap_m": round(prop_min, 5),
            "above_provisional_5cm_both": person_min >= .05 and prop_min >= .05,
            "full_video_frames": run["video"]["frames"]}


def main():
    report = json.loads((trial.OUT / "report.json").read_text())
    assert report["schema"] == "robot-learning-lab.ape-owner-history-meeting-pose.v1"
    for key, path in (("source_sha256", trial.__file__),
                      ("v38_source_sha256", trial.v38.__file__),
                      ("v41_source_sha256", trial.v41.__file__),
                      ("v44_source_sha256", trial.v44.__file__),
                      ("v45_report_sha256", trial.v45.OUT / "report.json"),
                      ("scene_sha256", trial.v38.v35.moving.SCENE_XML),
                      ("v36_report_sha256", trial.v38.SOURCE),
                      ("current_design_pointer_sha256",
                       trial.ROOT / "robot/ape-current-design.json")):
        assert report[key] == prior.sha(path), key
    source = trial.v38.original()
    policy = json.loads(trial.v38.joined.CONFIG.read_text())
    scenario = next(x for x in policy["cases"] if x["id"] == "late_crossing")
    checked = [check(item, report, source, scenario)
               for item in report["results"]]
    audit = {"passed": True, "rows": checked,
             "scope": "Stationary actor-packet history trigger, prior-packet executed commands, 20 ms independent MuJoCo person/prop gaps, and complete scene/flank video decoding. One authored scene and kinematic proxy; no physical safety qualification."}
    path = trial.OUT / "audit.json"
    path.write_text(json.dumps(audit, indent=2)+"\n")
    print(path)


if __name__ == "__main__":
    main()
