"""Independent packet, surface-track, command, geometry and video audit for V53."""

from __future__ import annotations

import json
import math
from pathlib import Path

import mujoco
import numpy as np

import audit_owner_joint_hazard_response as prior
import audit_owner_low_prop_scan_planes as v46_audit
import owner_surface_track_faster_turn as trial


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
    model.geom_size[prop_geom, 1] = trial.v51.v46.PROP_HALF_HEIGHT_M
    model.body_pos[prop_body, 2] = trial.v51.v46.PROP_CENTER_Z_M
    summary = source["summary"]
    start, _, t0, source_hash, endpoint = trial.runner.geometry_at(
        summary, source["result"]["result"]["episode"], scenario,
        model, data)
    assert t0 == report["checkpoint_s"] == run["checkpoint_time_s"]
    assert source_hash == run["source_v36_report_sha256"]
    v46_audit.same_packet(rows[0]["checkpoint_range_packet"], endpoint)
    scanner = trial.v51.v46.plane_packet(trial.runner.observed_packet, (.05, .30))
    end_xy = trial.v50.PATHS[item["path"]]
    hold = [row for row in rows if row["stage"] == "post_alarm_range_watch"]
    assert hold[-1]["actor_action"] == \
        "resume_after_three_fresh_clear_packets"
    for index, row in enumerate(hold):
        t = row["time_s"]
        person = trial.runner.human_xy(item["case"], summary, t,
                                       run["request_time_s"], None)
        trial.runner.v35.two.set_distractor(model, person)
        model.body_pos[prop_body, :2] = trial.prop_at(t, t0, end_xy)
        model.body_pos[model.body("audit_ape").id, :2] = start
        mujoco.mj_forward(model, data)
        v46_audit.same_packet(scanner(model, data, start, item["case"],
                                       index), row["actor_packet"])
    steps = [row for row in rows if row["stage"] == "pre_show_restage_step"]
    assert steps
    old_pose, old_row = np.asarray(start), hold[-1]
    direction = trial.v51.v47.v44.CANDIDATES["southwest"]
    for index, row in enumerate(steps):
        t = row["time_s"]
        pose = np.asarray(row["ape_xy_m"])
        assert np.linalg.norm(pose-(old_pose+direction*
                                  row["commanded_step_m"])) < 1e-8
        person = trial.runner.human_xy(item["case"], summary, t,
                                       run["request_time_s"], None)
        trial.runner.v35.two.set_distractor(model, person)
        model.body_pos[prop_body, :2] = trial.prop_at(t, t0, end_xy)
        model.body_pos[model.body("audit_ape").id, :2] = pose
        mujoco.mj_forward(model, data)
        v46_audit.same_packet(scanner(model, data, pose, item["case"],
                                       500+index), row["actor_packet"])
        old_pose, old_row = pose, row
    watch = [row for row in rows if row["stage"] ==
             "second_inference_range_watch"]
    assert watch
    alarm_index = None
    alarm = next((row for row in rows if row["stage"] ==
                  "near_body_escape_plan"), None)
    escape_direction = None
    old_center, old_center_s, track_count = None, None, 0
    old_pose = np.asarray(steps[-1]["ape_xy_m"])
    old_packet, old_time = None, None
    commanded_m = 0.
    fixture_gap_min = float("inf")
    response_old_sector, response_switched = None, False
    for index, row in enumerate(watch):
        t = row["sensor_time_s_raw"]
        assert abs(t-row["time_s"]) <= .0000051
        pose = np.asarray(row["ape_xy_m"])
        person = trial.runner.human_xy(item["case"], summary, t,
                                       run["request_time_s"],
                                       run["second_inference_start_s"])
        trial.runner.v35.two.set_distractor(model, person)
        model.body_pos[prop_body, :2] = trial.prop_at(t, t0, end_xy)
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
        healthy = len(valid) >= 36
        assert score["healthy"] == healthy
        assert abs(score["nearest_range_m"]-minimum) < .000011
        assert score["nearest_bearing_deg"] == -180.+5.*ray
        width = report["cluster_half_width_rays"]
        indexes = [j for j in range(max(0, ray-width),
                                    min(len(observed), ray+width+1))
                   if observed[j] is not None and
                   observed[j] <= minimum+report["cluster_range_band_m"]]
        endpoints = np.asarray([
            pose+observed[j]*np.array([math.cos(math.radians(-180.+5*j)),
                                       math.sin(math.radians(-180.+5*j))])
            for j in indexes])
        center = np.mean(endpoints, axis=0) if healthy else None
        assert np.linalg.norm(center-score["cluster_center_xy_m"]) < 1e-8
        predicted, closest_at_s, velocity = None, None, None
        associated = False
        if healthy and old_center is not None:
            dt = t-old_center_s
            delta = center-old_center
            associated = (0. < dt <= report["maximum_track_age_s"] and
                          np.linalg.norm(delta) <= report[
                              "maximum_association_step_m"])
            if associated:
                track_count += 1
                velocity = delta/dt
                speed2 = float(np.dot(velocity, velocity))
                if speed2 >= report["minimum_track_speed_mps"]**2:
                    closest_at_s = float(np.clip(
                        -np.dot(center-pose, velocity)/speed2,
                        0., report["prediction_horizon_s"]))
                    predicted = float(np.linalg.norm(
                        center+closest_at_s*velocity-pose))
            else:
                track_count = 0
        if not healthy:
            track_count = 0
        assert score["associated"] == associated
        assert score["track_count"] == track_count
        if velocity is None:
            assert score["cluster_velocity_mps"] is None
        else:
            assert np.linalg.norm(velocity-score["cluster_velocity_mps"]) < 1e-8
        if predicted is None:
            assert score["predicted_center_to_surface_m"] is None
            assert score["closest_at_s"] is None
        else:
            assert abs(predicted-score["predicted_center_to_surface_m"]) < .000011
            assert abs(closest_at_s-score["closest_at_s"]) < 1e-8
        if (predicted is not None and
                predicted < report["held_center_clearance_m"] and
                alarm_index is None):
            alarm_index = index
            assert alarm is not None
            future = center+closest_at_s*velocity
            expected_dir = (pose-future)/np.linalg.norm(pose-future)
            escape_direction = np.asarray(alarm["actor_decision"][
                "escape_direction"])
            assert np.linalg.norm(expected_dir-escape_direction) < 1e-8
            assert abs(t-item["near_body_alarm_s"]) < 1e-5
        assert score["alarm"] == (alarm_index is not None)
        command = row["near_body_command"]
        response = row["response_step_info"]
        if response is not None:
            assert command is None and old_packet is not None
            assert abs(response["range_packet_time_s"]-old_time) < .00001
            sector = prior.packet_sector_x(old_packet, old_pose)
            recorded_sector = response["east_south_sector_x_m"]
            assert (sector is None and recorded_sector is None or
                    sector is not None and
                    abs(sector-recorded_sector) < 1e-8)
            drop = None if sector is None or response_old_sector is None \
                else response_old_sector-sector
            if drop is None:
                assert response["east_south_x_drop_m"] is None
            else:
                assert abs(drop-response["east_south_x_drop_m"]) < 1e-8
                if drop >= trial.v51.v41.CLOSING_X_M:
                    response_switched = True
            if sector is not None:
                response_old_sector = sector
            requested = (trial.v51.v41.SOUTHWEST if response_switched
                         else trial.v51.v41.SOUTH)
            clear = prior.packet_clear(old_packet, old_pose, old_pose,
                                       requested)
            assert response["corridor_clear_from_range"] == clear
            direction = requested if clear else np.zeros(2)
            assert np.linalg.norm(direction-response["direction_xy"]) < 1e-8
            step = response["commanded_step_m"]
            assert np.linalg.norm(pose-(old_pose+direction*step)) < 1e-8
        elif command is not None:
            step = command["commanded_step_m"]
        else:
            assert np.linalg.norm(pose-old_pose) < 1e-8
            step = 0.
        if command is not None and step > 0.:
            assert alarm_index is not None and index > alarm_index
            assert step <= trial.v51.MAX_STEP_M+1e-9
            assert step <= trial.v51.ESCAPE_SPEED_MPS*(t-old_time)+1e-9
            assert t-old_time <= trial.v51.MAX_PACKET_AGE_S+1e-9
            assert np.linalg.norm(pose-(old_pose+escape_direction*step)) < 1e-8
            endpoints = trial.v51.v41.JointHazardResponse.endpoints(
                old_packet, old_pose)
            nearest = float(np.min(np.linalg.norm(endpoints-pose, axis=1)))
            assert abs(command["minimum_endpoint_gap_m"]-nearest) < 1e-8
            assert nearest >= trial.v51.STEP_ENDPOINT_GAP_M
            commanded_m += step
        elif response is None:
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
        old_center, old_center_s = center, t if healthy else None
    assert (alarm_index is not None) == (alarm is not None)
    assert abs(commanded_m-item["near_body_commanded_m"]) < 1e-8
    assert len([row for row in watch if row["near_body_command"] is not None
                and row["near_body_command"]["commanded_step_m"] > 0]) == \
        item["near_body_commands"]
    assert commanded_m <= trial.v51.MAX_TRAVEL_M+1e-9
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
        model.body_pos[prop_body, :2] = trial.prop_at(t, t0, end_xy)
        pg = trial.runner.geometry.gap(model, data, ape, person)
        og = trial.runner.prop_gap(model, data, ape)
        assert abs(pg-sample["gap_m"]) < .00011
        assert abs(og-sample["prop_gap_m"]) < .00011
        person_min, prop_min = min(person_min, pg), min(prop_min, og)
    assert abs(person_min-run["minimum_proxy_gap_m"]) < .00011
    assert abs(prop_min-run["minimum_prop_gap_m"]) < .00011
    assert fixture_gap_min > 0.
    if alarm_index is not None:
        assert run["outcome"] == "near_body_escape_owner_turn_unresolved"
    return {"path": item["path"], "case": item["case"],
            "outcome": run["outcome"],
            "alarm_s": (round(item["near_body_alarm_s"], 5)
                        if item["near_body_alarm_s"] is not None else None),
            "commanded_m": round(commanded_m, 5),
            "minimum_person_gap_m": round(person_min, 5),
            "minimum_prop_gap_m": round(prop_min, 5),
            "minimum_proxy_fixture_gap_m": round(fixture_gap_min, 5),
            "full_video_frames": run["video"]["frames"]}


def room_gap(end_xy, report, source, scenario):
    model = mujoco.MjModel.from_xml_path(str(trial.runner.v35.moving.SCENE_XML))
    data = mujoco.MjData(model)
    prop_geom = model.geom("route_obstacle_geom").id
    prop_body = model.body("route_obstacle").id
    model.geom_size[prop_geom, 1] = trial.v51.v46.PROP_HALF_HEIGHT_M
    model.body_pos[prop_body, 2] = trial.v51.v46.PROP_CENTER_Z_M
    trial.runner.geometry_at(source["summary"],
                             source["result"]["result"]["episode"],
                             scenario, model, data)
    t0 = report["checkpoint_s"]
    minimum, nearest = float("inf"), None
    for t in np.arange(t0, t0+trial.END_S+.2+.0001, .02):
        point = trial.prop_at(t, t0, end_xy)
        model.body_pos[prop_body, :2] = point
        mujoco.mj_forward(model, data)
        for geom_id in range(model.ngeom):
            if geom_id == prop_geom or not model.geom_contype[geom_id] or \
                    not model.geom_conaffinity[geom_id]:
                continue
            name = model.geom(geom_id).name
            if name.startswith(("floor_room", "floor_backing_room")):
                continue
            if np.linalg.norm(data.geom_xpos[geom_id, :2]-point) > 1.5:
                continue
            gap = float(mujoco.mj_geomDistance(
                model, data, prop_geom, geom_id, 2., None))
            if gap < minimum:
                minimum, nearest = gap, name
    return {"minimum_static_fixture_gap_m": round(minimum, 5),
            "path_fixture_clear": minimum >= 0.,
            "nearest_fixture": nearest}


def main():
    report = json.loads((trial.OUT / "report.json").read_text())
    assert report["schema"] == "robot-learning-lab.ape-owner-surface-track-faster-turn.v1"
    for key, path in (("source_sha256", trial.__file__),
                      ("runner_source_sha256", trial.runner.__file__),
                      ("v52_report_sha256", trial.v52.OUT / "report.json"),
                      ("v52_source_sha256", trial.v52.__file__),
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
    fixture = {name: room_gap(end, report, source, scenario)
               for name, end in trial.v50.PATHS.items()}
    assert all(row["minimum_static_fixture_gap_m"] >= .05
               for row in fixture.values())
    parameters = json.loads((trial.v52.OUT / "report.json").read_text())
    replay_report = {**parameters, **report}
    checked = [check(item, replay_report, source, scenario)
               for item in report["results"]]
    audit = {"passed": True, "rows": checked,
             "prop_static_fixture_clearance": fixture,
             "scope": "Re-rendered low/mid packets; independently rebuilt the anonymous surface cluster, association, velocity, short-horizon closest approach and every escape command; recomputed 20 ms person/prop geometry, checked proxy-vs-fixture at sensor ticks and decoded full scene/flank videos. Historical kinematic proxy and scripted prop; no selected-body safety qualification."}
    path = trial.OUT / "audit.json"
    path.write_text(json.dumps(audit, indent=2)+"\n")
    print(path)


if __name__ == "__main__":
    main()
