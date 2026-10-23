"""Independent matched-clock packet, policy, geometry and video audit for V54."""

from __future__ import annotations

import json
import math
from pathlib import Path

import mujoco
import numpy as np

import audit_owner_joint_hazard_response as prior
import audit_owner_low_prop_scan_planes as v46_audit
import owner_fixed_clock_pair as trial


def score_item(item, report, source, scenario):
    run = item["run"]
    rows = json.loads(Path(run["trace"]["path"]).read_text())
    assert prior.sha(run["trace"]["path"]) == run["trace"]["sha256"]
    assert prior.sha(run["video"]["path"]) == run["video"]["sha256"]
    assert prior.video_frames(run["video"]["path"], (528, 1040, 3)) == \
        run["video"]["frames"]
    flank = next(row for row in rows if row["stage"] == "flank_sensor_video")
    assert prior.sha(flank["path"]) == flank["sha256"]
    assert prior.video_frames(flank["path"], (480, 640, 3)) == flank["frames"]
    model = mujoco.MjModel.from_xml_path(str(trial.runner.v35.moving.SCENE_XML))
    data = mujoco.MjData(model)
    prop_geom = model.geom("route_obstacle_geom").id
    prop_body = model.body("route_obstacle").id
    model.geom_size[prop_geom, 1] = trial.v51.v46.PROP_HALF_HEIGHT_M
    model.body_pos[prop_body, 2] = trial.v51.v46.PROP_CENTER_Z_M
    summary = source["summary"]
    start, _, t0, source_hash, endpoint = trial.runner.geometry_at(
        summary, source["result"]["result"]["episode"], scenario,
        model, data)
    assert t0 == report["checkpoint_s"] == run["checkpoint_time_s"]
    assert source_hash == run["source_v36_report_sha256"]
    v46_audit.same_packet(rows[0]["checkpoint_range_packet"], endpoint)
    scanner = trial.v51.v46.plane_packet(trial.runner.observed_packet,
                                          (.05, .30))
    end_xy = trial.v50.PATHS[item["path"]]
    watch = [row for row in rows if row["stage"] ==
             "second_inference_range_watch"]
    steps = [row for row in rows if row["stage"] == "pre_show_restage_step"]
    assert len(watch) == report["monitor_ticks"] and steps
    assert all(abs(row["sensor_time_s_raw"]-(
        run["second_inference_start_s"]+index*report["monitor_period_s"])) <
        1e-8 for index, row in enumerate(watch))
    old_pose = np.asarray(steps[-1]["ape_xy_m"])
    old_packet, old_time, alarm_direction = None, None, None
    alarm_index, commanded_m = None, 0.
    track_center, track_t = None, None
    nearest_streak = 0
    for index, row in enumerate(watch):
        t = row["sensor_time_s_raw"]
        pose = np.asarray(row["ape_xy_m"])
        person = trial.runner.human_xy(item["case"], summary, t,
                                       run["request_time_s"],
                                       run["second_inference_start_s"])
        trial.runner.v35.two.set_distractor(model, person)
        model.body_pos[prop_body, :2] = trial.v50.prop_at(t, t0, end_xy)
        model.body_pos[model.body("audit_ape").id, :2] = pose
        mujoco.mj_forward(model, data)
        observed = scanner(model, data, pose, item["case"], 1000+index,
                           person_xy=person, hidden_return=False)
        v46_audit.same_packet(observed, row["actor_packet"])
        valid = [(float(value), i) for i, value in enumerate(observed)
                 if value is not None]
        near, ray = min(valid)
        score = row["near_body_observation"]
        if item["policy"] == "side_only":
            assert score is None
        elif item["policy"] == "nearest_always":
            assert score["valid_rays"] == len(valid)
            assert abs(score["nearest_range_m"]-near) < .000011
            nearest_streak = nearest_streak+1 if near < trial.v51.ALARM_RANGE_M else 0
            assert score["risk_streak"] == nearest_streak
            if alarm_index is None and nearest_streak >= 2:
                alarm_index = index
                bearing = math.radians(-180.+5.*ray)
                alarm_direction = -np.array([math.cos(bearing),
                                             math.sin(bearing)])
        else:
            assert score["valid_rays"] == len(valid)
            assert abs(score["nearest_range_m"]-near) < .000011
            band = trial.v52.CLUSTER_RANGE_BAND_M
            width = trial.v52.CLUSTER_HALF_WIDTH_RAYS
            indexes = [i for i in range(max(0, ray-width),
                                        min(72, ray+width+1))
                       if observed[i] is not None and observed[i] <= near+band]
            cloud = np.asarray([pose+observed[i]*np.array([
                math.cos(math.radians(-180.+5*i)),
                math.sin(math.radians(-180.+5*i))]) for i in indexes])
            center = np.mean(cloud, axis=0)
            assert np.linalg.norm(center-score["cluster_center_xy_m"]) < 1e-8
            prediction, future = None, None
            if track_center is not None:
                delta = center-track_center
                dt = t-track_t
                if (0. < dt <= trial.v52.MAX_TRACK_AGE_S and
                        np.linalg.norm(delta) <= trial.v52.MAX_ASSOCIATION_STEP_M):
                    velocity = delta/dt
                    speed2 = float(np.dot(velocity, velocity))
                    if speed2 >= trial.v52.MIN_TRACK_SPEED_MPS**2:
                        tau = float(np.clip(-np.dot(center-pose, velocity)/
                                            speed2, 0.,
                                            trial.v52.PREDICTION_HORIZON_S))
                        future = center+velocity*tau
                        prediction = float(np.linalg.norm(future-pose))
            if prediction is None:
                assert score["predicted_center_to_surface_m"] is None
            else:
                assert abs(score["predicted_center_to_surface_m"]-
                           prediction) < .000011
                if (alarm_index is None and
                        prediction < trial.v52.HELD_CENTER_CLEARANCE_M):
                    alarm_index = index
                    alarm_direction = (pose-future)/np.linalg.norm(pose-future)
            track_center, track_t = center, t
        if score is not None:
            assert score["alarm"] == (alarm_index is not None)
            if alarm_index is not None:
                assert np.linalg.norm(score["escape_direction"]-
                                      alarm_direction) < 1e-8
        command = row["near_body_command"]
        if item["policy"] == "side_only":
            assert command is None
            assert np.linalg.norm(pose-old_pose) < 1e-8
        else:
            assert command is not None
            step = command["commanded_step_m"]
            if step > 0.:
                assert alarm_index is not None and index > alarm_index
                assert step <= trial.v51.MAX_STEP_M+1e-9
                assert step <= trial.v51.ESCAPE_SPEED_MPS*(t-old_time)+1e-9
                assert t-old_time <= trial.v51.MAX_PACKET_AGE_S+1e-9
                assert np.linalg.norm(pose-(old_pose+alarm_direction*step)) < 1e-8
                endpoints = trial.v51.v41.JointHazardResponse.endpoints(
                    old_packet, old_pose)
                nearest_endpoint = float(np.min(np.linalg.norm(
                    endpoints-pose, axis=1)))
                assert nearest_endpoint >= trial.v51.STEP_ENDPOINT_GAP_M
                assert abs(command["minimum_endpoint_gap_m"]-
                           nearest_endpoint) < 1e-8
                commanded_m += step
            else:
                assert np.linalg.norm(pose-old_pose) < 1e-8
        old_pose, old_packet, old_time = pose, observed, t
    assert (alarm_index is not None) == (item["near_body_alarm_s"] is not None)
    if alarm_index is not None:
        assert abs(watch[alarm_index]["time_s"]-item["near_body_alarm_s"]) < 1e-5
    assert abs(commanded_m-item["near_body_commanded_m"]) < 1e-8
    assert commanded_m <= trial.v51.MAX_TRAVEL_M+1e-9
    timeline = [(t0, np.asarray(start)),
                (run["resume_time_s"], np.asarray(start))]
    timeline.extend((row["time_s"], np.asarray(row["ape_xy_m"]))
                    for row in steps+watch)
    times = np.asarray([t for t, _ in timeline])
    poses = np.asarray([xy for _, xy in timeline])
    person_min, prop_min = float("inf"), float("inf")
    for sample in run["evaluator_gap_samples"]:
        t = sample["time_s"]
        ape = np.asarray([np.interp(t, times, poses[:, axis])
                          for axis in range(2)])
        assert np.linalg.norm(ape-sample["ape_xy_m"]) < .00011
        person = trial.runner.human_xy(item["case"], summary, t,
                                       run["request_time_s"],
                                       run["second_inference_start_s"])
        model.body_pos[prop_body, :2] = trial.v50.prop_at(t, t0, end_xy)
        person_gap = trial.runner.geometry.gap(model, data, ape, person)
        prop_gap = trial.runner.prop_gap(model, data, ape)
        assert abs(person_gap-sample["gap_m"]) < .00011
        assert abs(prop_gap-sample["prop_gap_m"]) < .00011
        person_min, prop_min = min(person_min, person_gap), min(prop_min, prop_gap)
    assert abs(person_min-run["minimum_proxy_gap_m"]) < .00011
    assert abs(prop_min-run["minimum_prop_gap_m"]) < .00011
    return {"policy": item["policy"], "path": item["path"],
            "outcome": run["outcome"],
            "alarm_s": item["near_body_alarm_s"],
            "minimum_person_gap_m": round(person_min, 5),
            "minimum_prop_gap_m": round(prop_min, 5),
            "commanded_m": round(commanded_m, 5),
            "video_frames": run["video"]["frames"]}


def main():
    path = trial.OUT / "report.json"
    report = json.loads(path.read_text())
    assert report["schema"] == "robot-learning-lab.ape-owner-fixed-clock-pair.v1"
    for key, target in (("source_sha256", trial.__file__),
                        ("runner_source_sha256", trial.runner.__file__),
                        ("v50_report_sha256", trial.v50.OUT / "report.json"),
                        ("v51_report_sha256", trial.v51.OUT / "report.json"),
                        ("v52_report_sha256", trial.v52.OUT / "report.json"),
                        ("scene_sha256", trial.runner.v35.moving.SCENE_XML),
                        ("v36_report_sha256", trial.runner.SOURCE),
                        ("current_design_pointer_sha256",
                         trial.ROOT / "robot/ape-current-design.json")):
        assert report[key] == prior.sha(target), key
    fixture = json.loads((trial.v50.OUT / "audit.json").read_text())
    assert fixture["passed"] and all(
        row["path_fixture_clear"] for row in fixture[
            "prop_static_fixture_clearance"].values())
    source = trial.runner.original()
    policy = json.loads(trial.runner.joined.CONFIG.read_text())
    scenario = next(x for x in policy["cases"] if x["id"] == "late_crossing")
    scored = [score_item(item, report, source, scenario)
              for item in report["results"]]
    paired = []
    for name in trial.POLICIES:
        left, right = [next(item for item in report["results"]
                            if item["policy"] == name and item["path"] == path)
                       for path in trial.v50.PATHS]
        left_rows = json.loads(Path(left["run"]["trace"]["path"]).read_text())
        right_rows = json.loads(Path(right["run"]["trace"]["path"]).read_text())
        left_watch = [row for row in left_rows if row["stage"] ==
                      "second_inference_range_watch"]
        right_watch = [row for row in right_rows if row["stage"] ==
                       "second_inference_range_watch"]
        common = [(a, b) for a, b in zip(left_watch, right_watch)
                  if a["sensor_time_s_raw"] <= report["identical_path_through_s"]]
        assert common
        for a, b in common:
            assert abs(a["sensor_time_s_raw"]-b["sensor_time_s_raw"]) < 1e-8
            assert np.linalg.norm(np.asarray(a["ape_xy_m"])-
                                  b["ape_xy_m"]) < 1e-8
            v46_audit.same_packet(a["actor_packet"], b["actor_packet"])
            assert a["actor_action"] == b["actor_action"]
            assert a["near_body_observation"] == b["near_body_observation"]
            assert a["near_body_command"] == b["near_body_command"]
        paired.append({"policy": name, "matching_inference_packets_before_turn":
                       len(common)})
    audit = {"passed": True, "rows": scored, "paired_history": paired,
             "scope": "Same 100 ms logical sensor clock through 3.5 s of owner inference; exact matched pre-turn packets and actions for each outward/inward policy pair; independent dual-height packet render, policy decision and prior-packet command replay; 20 ms person/prop geometry, scene videos and fixture-clear V50 path. This isolates logical timing but is not measured threaded compute latency or physical ape motion."}
    out = trial.OUT / "audit.json"
    out.write_text(json.dumps(audit, indent=2)+"\n")
    print(out)


if __name__ == "__main__":
    main()
