"""Independently replay V43 commands and remeasure person/prop separation."""

from __future__ import annotations

import json
from pathlib import Path

import mujoco
import numpy as np

import audit_owner_joint_hazard_response as prior
import owner_pre_show_meeting_pose as trial


def check(item, source, scenario, checkpoint_s):
    run = item["run"]
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
    assert t0 == checkpoint_s == run["checkpoint_time_s"]
    assert source_hash == run["source_v36_report_sha256"]
    assert rows[0]["checkpoint_range_packet"] == endpoint
    restage = [row for row in rows if row["stage"] ==
               "pre_show_restage_step"]
    resume = next(row for row in rows if row["stage"] ==
                  "post_alarm_range_watch" and row["actor_action"] ==
                  "resume_after_three_fresh_clear_packets")
    assert abs(resume["time_s"]-run["resume_time_s"]) < 1e-5
    if item["mode"] == "restage":
        assert len(restage) == 18
        assert item["restage_steps"] == len(restage)
        previous = resume
        previous_pose = np.asarray(start)
        for row in restage:
            assert abs(row["time_s"]-previous["time_s"]-trial.TICK_S) < 1e-5
            assert previous["actor_packet"] is not None
            assert trial.v38.side_healthy(previous["actor_packet"])
            step = row["commanded_step_m"]
            assert 0. < step <= trial.SPEED_MPS*trial.TICK_S+1e-9
            current_pose = np.asarray(row["ape_xy_m"])
            assert np.linalg.norm(current_pose-(previous_pose+
                                  trial.WEST*step)) < 1e-8
            endpoints = trial.v41.JointHazardResponse.endpoints(
                previous["actor_packet"], previous_pose)
            nearest = float(np.min(np.linalg.norm(
                endpoints-current_pose, axis=1)))
            assert abs(nearest-row["actor_decision"][
                "minimum_endpoint_gap_m"]) < 1e-8
            assert nearest >= trial.MIN_ENDPOINT_GAP_M
            assert row["actor_decision"]["action"] == \
                "step_west_from_fresh_scan"
            previous, previous_pose = row, current_pose
        assert trial.RangeCheckedMeetingPose().max_ticks >= len(restage)
        assert abs(float(start[0]-previous_pose[0])-trial.TARGET_M) < 1e-8
        assert abs(run["second_inference_start_s"]-
                   (restage[-1]["time_s"]+trial.v38.SHOW_SETTLE_S)) < 1e-5
    else:
        assert not restage
        assert abs(run["second_inference_start_s"]-
                   (resume["time_s"]+trial.v38.SHOW_SETTLE_S)) < 1e-5
    watch = [row for row in rows if row["stage"] ==
             "second_inference_range_watch"]
    assert watch
    timeline = [(t0, np.asarray(start)),
                (resume["time_s"], np.asarray(start))]
    timeline.extend((row["time_s"], np.asarray(row["ape_xy_m"]))
                    for row in restage+watch)
    times = np.asarray([row[0] for row in timeline])
    poses = np.asarray([row[1] for row in timeline])
    person_min, prop_min = float("inf"), float("inf")
    for sample in run["evaluator_gap_samples"]:
        t = sample["time_s"]
        ape = np.asarray([np.interp(t, times, poses[:, axis])
                          for axis in range(2)])
        assert np.linalg.norm(ape-sample["ape_xy_m"]) < .00011
        person = trial.v38.human_xy(
            item["case"], summary, t, run["request_time_s"],
            run["second_inference_start_s"])
        model.body_pos[model.body("route_obstacle").id, :2] = \
            trial.v41.v40.prop_at(t, checkpoint_s,
                                  trial.v41.PROP_CASES[item["prop_case"]])
        person_gap = trial.v38.geometry.gap(model, data, ape, person)
        prop_gap = trial.v38.prop_gap(model, data, ape)
        assert abs(person_gap-sample["gap_m"]) < .00011
        assert abs(prop_gap-sample["prop_gap_m"]) < .00011
        person_min = min(person_min, person_gap)
        prop_min = min(prop_min, prop_gap)
    assert abs(person_min-run["minimum_proxy_gap_m"]) < .00011
    assert abs(prop_min-run["minimum_prop_gap_m"]) < .00011
    sensor = next(row for row in rows if row["stage"] ==
                  "flank_sensor_video")
    assert prior.sha(sensor["path"]) == sensor["sha256"]
    assert prior.video_frames(sensor["path"], (480, 640, 3)) == \
        sensor["frames"]
    if item["case"] == "parked_after_clear" and item["mode"] == "baseline":
        assert run["outcome"] == "new_show_metric_item_seen_unattributed"
    return {"prop_case": item["prop_case"], "case": item["case"],
            "mode": item["mode"], "outcome": run["outcome"],
            "minimum_person_gap_m": round(person_min, 5),
            "minimum_prop_gap_m": round(prop_min, 5),
            "above_provisional_5cm_both": (person_min >= .05 and
                                           prop_min >= .05),
            "restage_steps": len(restage),
            "full_video_frames": run["video"]["frames"]}


def main():
    report = json.loads((trial.OUT / "report.json").read_text())
    assert report["schema"] == \
        "robot-learning-lab.ape-owner-pre-show-meeting-pose.v1"
    assert report["source_sha256"] == prior.sha(trial.__file__)
    assert report["v38_source_sha256"] == prior.sha(trial.v38.__file__)
    assert report["v41_source_sha256"] == prior.sha(trial.v41.__file__)
    assert report["scene_sha256"] == prior.sha(trial.v38.v35.moving.SCENE_XML)
    assert report["v36_report_sha256"] == prior.sha(trial.v38.SOURCE)
    assert report["current_design_pointer_sha256"] == prior.sha(
        trial.ROOT / "robot/ape-current-design.json")
    source = trial.v38.original()
    policy = json.loads(trial.v38.joined.CONFIG.read_text())
    scenario = next(x for x in policy["cases"] if x["id"] == "late_crossing")
    checkpoint_s = source["summary"]["range_packets"][-1]["time_s"]
    checked = [check(item, source, scenario, checkpoint_s)
               for item in report["results"]]
    audit = {"passed": True, "rows": checked,
             "scope": "Replay each restage command from the preceding saved range packet, reconstruct the commanded pose history, remeasure 20 ms MuJoCo person/prop gaps, and decode every complete scene and flank video. Clearance is a provisional proxy criterion, not physical qualification."}
    path = trial.OUT / "audit.json"
    path.write_text(json.dumps(audit, indent=2)+"\n")
    print(path)


if __name__ == "__main__":
    main()
