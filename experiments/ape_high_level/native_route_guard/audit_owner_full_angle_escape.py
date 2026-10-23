"""Independent packet, command, geometry and video audit for V51."""

from __future__ import annotations

import json
import math
from pathlib import Path

import mujoco
import numpy as np

import audit_owner_joint_hazard_response as prior
import audit_owner_low_prop_scan_planes as v46_audit
import owner_full_angle_escape as trial


def check(item, report, source, scenario):
    run = item["run"]
    rows = json.loads(Path(run["trace"]["path"]).read_text())
    assert prior.sha(run["trace"]["path"]) == run["trace"]["sha256"]
    assert prior.sha(run["video"]["path"]) == run["video"]["sha256"]
    assert prior.video_frames(run["video"]["path"], (528, 1040, 3)) == \
        run["video"]["frames"]
    sensor = next(row for row in rows if row["stage"] == "flank_sensor_video")
    assert prior.sha(sensor["path"]) == sensor["sha256"]
    assert prior.video_frames(sensor["path"], (480, 640, 3)) == \
        sensor["frames"]
    model = mujoco.MjModel.from_xml_path(str(trial.runner.v35.moving.SCENE_XML))
    data = mujoco.MjData(model)
    prop_geom = model.geom("route_obstacle_geom").id
    prop_body = model.body("route_obstacle").id
    model.geom_size[prop_geom, 1] = trial.v46.PROP_HALF_HEIGHT_M
    model.body_pos[prop_body, 2] = trial.v46.PROP_CENTER_Z_M
    summary = source["summary"]
    start, _, t0, source_hash, endpoint = trial.runner.geometry_at(
        summary, source["result"]["result"]["episode"], scenario,
        model, data)
    assert t0 == report["checkpoint_s"] == run["checkpoint_time_s"]
    assert source_hash == run["source_v36_report_sha256"]
    v46_audit.same_packet(rows[0]["checkpoint_range_packet"], endpoint)
    scanner = trial.v46.plane_packet(trial.runner.observed_packet, (.05, .30))
    end_xy = trial.v50.PATHS[item["path"]]
    hold = [row for row in rows if row["stage"] == "post_alarm_range_watch"]
    assert hold[-1]["actor_action"] == \
        "resume_after_three_fresh_clear_packets"
    for index, row in enumerate(hold):
        t = row["time_s"]
        person = trial.runner.human_xy(item["case"], summary, t,
                                       run["request_time_s"], None)
        trial.runner.v35.two.set_distractor(model, person)
        model.body_pos[prop_body, :2] = trial.v50.prop_at(t, t0, end_xy)
        model.body_pos[model.body("audit_ape").id, :2] = start
        mujoco.mj_forward(model, data)
        v46_audit.same_packet(scanner(model, data, start, item["case"],
                                       index), row["actor_packet"])
    steps = [row for row in rows if row["stage"] == "pre_show_restage_step"]
    assert steps
    old_pose, old_row = np.asarray(start), hold[-1]
    direction = trial.v47.v44.CANDIDATES["southwest"]
    for index, row in enumerate(steps):
        t = row["time_s"]
        pose = np.asarray(row["ape_xy_m"])
        assert np.linalg.norm(pose-(old_pose+direction*
                                  row["commanded_step_m"])) < 1e-8
        person = trial.runner.human_xy(item["case"], summary, t,
                                       run["request_time_s"], None)
        trial.runner.v35.two.set_distractor(model, person)
        model.body_pos[prop_body, :2] = trial.v50.prop_at(t, t0, end_xy)
        model.body_pos[model.body("audit_ape").id, :2] = pose
        mujoco.mj_forward(model, data)
        v46_audit.same_packet(scanner(model, data, pose, item["case"],
                                       500+index), row["actor_packet"])
        old_pose, old_row = pose, row
    watch = [row for row in rows if row["stage"] ==
             "second_inference_range_watch"]
    assert watch
    risk_streak = 0
    alarm_index = None
    alarm = next(row for row in rows if row["stage"] ==
                 "near_body_escape_plan")
    escape_direction = np.asarray(alarm["actor_decision"]["escape_direction"])
    assert abs(np.linalg.norm(escape_direction)-1.) < 1e-8
    old_pose = np.asarray(steps[-1]["ape_xy_m"])
    old_packet, old_time = None, None
    commanded_m = 0.
    fixture_gap_min = float("inf")
    for index, row in enumerate(watch):
        t = row["sensor_time_s_raw"]
        assert abs(t-row["time_s"]) <= .0000051
        pose = np.asarray(row["ape_xy_m"])
        person = trial.runner.human_xy(item["case"], summary, t,
                                       run["request_time_s"],
                                       run["second_inference_start_s"])
        trial.runner.v35.two.set_distractor(model, person)
        model.body_pos[prop_body, :2] = trial.v50.prop_at(t, t0, end_xy)
        model.body_pos[model.body("audit_ape").id, :2] = pose
        mujoco.mj_forward(model, data)
        observed = scanner(
            model, data, pose, item["case"], 1000+index,
            person_xy=person,
            hidden_return=item["case"] in trial.runner.BLIND_RETURN)
        v46_audit.same_packet(observed, row["actor_packet"])
        valid = [(float(v), i) for i, v in enumerate(observed)
                 if v is not None]
        minimum, ray = min(valid)
        score = row["near_body_observation"]
        assert score["valid_rays"] == len(valid)
        assert score["healthy"] == (len(valid) >= 36)
        assert abs(score["nearest_range_m"]-minimum) < .000011
        assert score["nearest_bearing_deg"] == -180.+5.*ray
        risk_streak = risk_streak+1 if minimum < trial.ALARM_RANGE_M else 0
        assert score["risk_streak"] == risk_streak
        if risk_streak >= trial.ALARM_STREAK and alarm_index is None:
            alarm_index = index
            bearing = math.radians(-180.+5.*ray)
            expected_dir = -np.array([math.cos(bearing), math.sin(bearing)])
            assert np.linalg.norm(expected_dir-escape_direction) < 1e-8
            assert abs(t-item["near_body_alarm_s"]) < 1e-5
        assert score["alarm"] == (alarm_index is not None)
        command = row["near_body_command"]
        assert command is not None
        step = command["commanded_step_m"]
        if step > 0.:
            assert alarm_index is not None and index > alarm_index
            assert step <= trial.MAX_STEP_M+1e-9
            assert step <= trial.ESCAPE_SPEED_MPS*(t-old_time)+1e-9
            assert t-old_time <= trial.MAX_PACKET_AGE_S+1e-9
            assert np.linalg.norm(pose-(old_pose+escape_direction*step)) < 1e-8
            endpoints = trial.v41.JointHazardResponse.endpoints(
                old_packet, old_pose)
            nearest = float(np.min(np.linalg.norm(endpoints-pose, axis=1)))
            assert abs(command["minimum_endpoint_gap_m"]-nearest) < 1e-8
            assert nearest >= trial.STEP_ENDPOINT_GAP_M
            commanded_m += step
        else:
            assert np.linalg.norm(pose-old_pose) < 1e-8
        # Check the represented ape cylinder against collidable fixtures at
        # each sensor tick; person/prop distances are scored separately.
        ape_geom = model.geom("audit_ape_geom").id
        for geom_id in range(model.ngeom):
            if geom_id == ape_geom or not model.geom_contype[geom_id] or \
                    not model.geom_conaffinity[geom_id]:
                continue
            name = model.geom(geom_id).name
            if name.startswith(("floor_room", "floor_backing_room")):
                continue
            if np.linalg.norm(data.geom_xpos[geom_id, :2]-pose) > 1.5:
                continue
            fixture_gap_min = min(fixture_gap_min, float(
                mujoco.mj_geomDistance(model, data, ape_geom, geom_id,
                                       2., None)))
        old_pose, old_packet, old_time = pose, observed, t
    assert alarm_index is not None
    assert abs(commanded_m-item["near_body_commanded_m"]) < 1e-8
    assert len([row for row in watch if row["near_body_command"][
        "commanded_step_m"] > 0]) == item["near_body_commands"]
    assert commanded_m <= trial.MAX_TRAVEL_M+1e-9
    timeline = [(t0, np.asarray(start)),
                (run["resume_time_s"], np.asarray(start))]
    timeline.extend((row["time_s"], np.asarray(row["ape_xy_m"]))
                    for row in steps+watch)
    times = np.asarray([row[0] for row in timeline])
    poses = np.asarray([row[1] for row in timeline])
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
        pg = trial.runner.geometry.gap(model, data, ape, person)
        og = trial.runner.prop_gap(model, data, ape)
        assert abs(pg-sample["gap_m"]) < .00011
        assert abs(og-sample["prop_gap_m"]) < .00011
        person_min, prop_min = min(person_min, pg), min(prop_min, og)
    assert abs(person_min-run["minimum_proxy_gap_m"]) < .00011
    assert abs(prop_min-run["minimum_prop_gap_m"]) < .00011
    assert fixture_gap_min > 0.
    assert run["outcome"] == "near_body_escape_owner_turn_unresolved"
    return {"path": item["path"], "case": item["case"],
            "alarm_s": round(item["near_body_alarm_s"], 5),
            "commanded_m": round(commanded_m, 5),
            "minimum_person_gap_m": round(person_min, 5),
            "minimum_prop_gap_m": round(prop_min, 5),
            "minimum_proxy_fixture_gap_m": round(fixture_gap_min, 5),
            "full_video_frames": run["video"]["frames"]}


def main():
    report = json.loads((trial.OUT / "report.json").read_text())
    assert report["schema"] == "robot-learning-lab.ape-owner-full-angle-escape.v1"
    for key, path in (("source_sha256", trial.__file__),
                      ("runner_source_sha256", trial.runner.__file__),
                      ("v50_report_sha256", trial.v50.OUT / "report.json"),
                      ("v50_source_sha256", trial.v50.__file__),
                      ("scene_sha256", trial.runner.v35.moving.SCENE_XML),
                      ("v36_report_sha256", trial.runner.SOURCE),
                      ("current_design_pointer_sha256",
                       trial.ROOT / "robot/ape-current-design.json")):
        assert report[key] == prior.sha(path), key
    baseline_audit = json.loads((trial.v50.OUT / "audit.json").read_text())
    assert baseline_audit["passed"]
    assert all(row["path_fixture_clear"] for row in baseline_audit[
        "prop_static_fixture_clearance"].values())
    source = trial.runner.original()
    policy = json.loads(trial.runner.joined.CONFIG.read_text())
    scenario = next(x for x in policy["cases"] if x["id"] == "late_crossing")
    checked = [check(item, report, source, scenario)
               for item in report["results"]]
    audit = {"passed": True, "rows": checked,
             "scope": "Re-rendered all low/mid hold, restage and inference packets; independently replayed full-angle alarm and each prior-packet escape command; recomputed 20 ms person/prop geometry; checked proxy-vs-fixture at sensor ticks and decoded full scene/flank videos. Historical kinematic proxy and scripted prop; no selected-body safety qualification."}
    path = trial.OUT / "audit.json"
    path.write_text(json.dumps(audit, indent=2)+"\n")
    print(path)


if __name__ == "__main__":
    main()
