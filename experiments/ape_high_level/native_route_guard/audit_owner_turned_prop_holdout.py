"""Audit V49's matched actor history and changed future prop path."""

from __future__ import annotations

import json
import math
from pathlib import Path

import mujoco
import numpy as np

import audit_owner_joint_hazard_response as prior
import audit_owner_low_prop_scan_planes as v46_audit
import owner_turned_prop_holdout as trial


def check(item, report, source, scenario):
    run = item["run"]
    rows = json.loads(Path(run["trace"]["path"]).read_text())
    assert prior.sha(run["trace"]["path"]) == run["trace"]["sha256"]
    assert prior.sha(run["video"]["path"]) == run["video"]["sha256"]
    assert prior.video_frames(run["video"]["path"], (528, 1040, 3)) == \
        run["video"]["frames"]
    model = mujoco.MjModel.from_xml_path(str(trial.v38.v35.moving.SCENE_XML))
    data = mujoco.MjData(model)
    prop_geom = model.geom("route_obstacle_geom").id
    prop_body = model.body("route_obstacle").id
    model.geom_size[prop_geom, 1] = trial.v46.PROP_HALF_HEIGHT_M
    model.body_pos[prop_body, 2] = trial.v46.PROP_CENTER_Z_M
    summary = source["summary"]
    start, _, t0, source_hash, endpoint = trial.v38.geometry_at(
        summary, source["result"]["result"]["episode"], scenario,
        model, data)
    assert t0 == report["checkpoint_s"] == run["checkpoint_time_s"]
    assert source_hash == run["source_v36_report_sha256"]
    v46_audit.same_packet(rows[0]["checkpoint_range_packet"], endpoint)
    scanner = trial.v46.plane_packet(trial.v38.observed_packet, (.05, .30))
    end_xy = trial.PATHS[item["path"]]
    hold = [row for row in rows if row["stage"] == "post_alarm_range_watch"]
    assert hold[-1]["actor_action"] == \
        "resume_after_three_fresh_clear_packets"
    for index, row in enumerate(hold):
        t = row["time_s"]
        person = trial.v38.human_xy(item["case"], summary, t,
                                    run["request_time_s"], None)
        trial.v38.v35.two.set_distractor(model, person)
        model.body_pos[prop_body, :2] = trial.prop_at(t, t0, end_xy)
        model.body_pos[model.body("audit_ape").id, :2] = start
        mujoco.mj_forward(model, data)
        v46_audit.same_packet(scanner(model, data, start, item["case"],
                                       index), row["actor_packet"])
    expected_novel = trial.v47.novel_forward_returns(
        hold[-2]["actor_packet"], hold[-1]["actor_packet"])
    assert expected_novel == item["novel_forward_returns"] and expected_novel
    assert item["choice"] == "northeast"
    steps = [row for row in rows if row["stage"] == "pre_show_restage_step"]
    assert len(steps) == item["restage_steps"]
    old_pose, old_row = np.asarray(start), hold[-1]
    for index, row in enumerate(steps):
        t = row["time_s"]
        pose = np.asarray(row["ape_xy_m"])
        distance = row["commanded_step_m"]
        assert abs(t-old_row["time_s"]-trial.v38.HOLD_PERIOD_S) < 1e-5
        assert np.linalg.norm(pose-(old_pose+
                        trial.v47.NORTHEAST*distance)) < 1e-8
        points = trial.v41.JointHazardResponse.endpoints(
            old_row["actor_packet"], old_pose)
        minimum = float(np.min(np.linalg.norm(points-pose, axis=1)))
        assert abs(minimum-row["actor_decision"][
            "minimum_endpoint_gap_m"]) < 1e-8
        assert minimum >= trial.v47.v44.MIN_ENDPOINT_GAP_M
        person = trial.v38.human_xy(item["case"], summary, t,
                                    run["request_time_s"], None)
        trial.v38.v35.two.set_distractor(model, person)
        model.body_pos[prop_body, :2] = trial.prop_at(t, t0, end_xy)
        model.body_pos[model.body("audit_ape").id, :2] = pose
        mujoco.mj_forward(model, data)
        v46_audit.same_packet(scanner(model, data, pose, item["case"],
                                       500+index), row["actor_packet"])
        old_pose, old_row = pose, row
    watch = [row for row in rows if row["stage"] ==
             "second_inference_range_watch"]
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
        person = trial.v38.human_xy(item["case"], summary, t,
                                    run["request_time_s"],
                                    run["second_inference_start_s"])
        model.body_pos[prop_body, :2] = trial.prop_at(t, t0, end_xy)
        person_gap = trial.v38.geometry.gap(model, data, ape, person)
        prop_gap = trial.v38.prop_gap(model, data, ape)
        assert abs(person_gap-sample["gap_m"]) < .00011
        assert abs(prop_gap-sample["prop_gap_m"]) < .00011
        person_min = min(person_min, person_gap)
        prop_min = min(prop_min, prop_gap)
    assert abs(person_min-run["minimum_proxy_gap_m"]) < .00011
    assert abs(prop_min-run["minimum_prop_gap_m"]) < .00011
    if watch:
        sensor = next(row for row in rows if row["stage"] ==
                      "flank_sensor_video")
        assert prior.sha(sensor["path"]) == sensor["sha256"]
        assert prior.video_frames(sensor["path"], (480, 640, 3)) == \
            sensor["frames"]
    # Counterfactual hold after the terminal decision: a stop command is not
    # proof that the approaching prop stops. This is evaluator-only.
    prop_finish_s = t0+trial.MOTION["move_begin_after_checkpoint_s"]+\
        trial.MOTION["move_duration_s"]
    post_stop_min = None
    if run["last_time_s"] < prop_finish_s:
        post_stop_min = float("inf")
        for t in np.arange(run["last_time_s"], prop_finish_s+.0001, .02):
            model.body_pos[prop_body, :2] = trial.prop_at(t, t0, end_xy)
            post_stop_min = min(post_stop_min, trial.v38.prop_gap(
                model, data, poses[-1]))
    return {"path": item["path"], "case": item["case"],
            "outcome": run["outcome"],
            "minimum_person_gap_m": round(person_min, 5),
            "minimum_prop_gap_m": round(prop_min, 5),
            "post_stop_prop_gap_m": (round(post_stop_min, 5)
                                     if post_stop_min is not None else None),
            "full_video_frames": run["video"]["frames"]}


def room_gap(end_xy, report, source, scenario):
    model = mujoco.MjModel.from_xml_path(str(trial.v38.v35.moving.SCENE_XML))
    data = mujoco.MjData(model)
    prop_geom = model.geom("route_obstacle_geom").id
    prop_body = model.body("route_obstacle").id
    model.geom_size[prop_geom, 1] = trial.v46.PROP_HALF_HEIGHT_M
    model.body_pos[prop_body, 2] = trial.v46.PROP_CENTER_Z_M
    trial.v38.geometry_at(source["summary"],
                          source["result"]["result"]["episode"],
                          scenario, model, data)
    t0 = report["checkpoint_s"]
    min_gap, nearest = float("inf"), None
    for t in np.arange(t0, t0+3.2+.0001, .02):
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
            if gap < min_gap:
                min_gap, nearest = gap, name
    return {"minimum_static_fixture_gap_m": round(min_gap, 5),
            "path_fixture_clear": min_gap >= 0.,
            "nearest_fixture": nearest}


def main():
    report = json.loads((trial.OUT / "report.json").read_text())
    assert report["schema"] == "robot-learning-lab.ape-owner-turned-prop-holdout.v1"
    for key, path in (("source_sha256", trial.__file__),
                      ("v38_source_sha256", trial.v38.__file__),
                      ("v46_source_sha256", trial.v46.__file__),
                      ("v47_source_sha256", trial.v47.__file__),
                      ("v48_report_sha256", trial.v48.OUT / "report.json"),
                      ("scene_sha256", trial.v38.v35.moving.SCENE_XML),
                      ("v36_report_sha256", trial.v38.SOURCE),
                      ("current_design_pointer_sha256",
                       trial.ROOT / "robot/ape-current-design.json")):
        assert report[key] == prior.sha(path), key
    assert report["path_end_xy_m"] == {
        name: xy.tolist() for name, xy in trial.PATHS.items()}
    source = trial.v38.original()
    policy = json.loads(trial.v38.joined.CONFIG.read_text())
    scenario = next(x for x in policy["cases"] if x["id"] == "late_crossing")
    checked = [check(item, report, source, scenario)
               for item in report["results"]]
    prefix_checks = []
    for case in trial.v47.CASES:
        pair = [next(item for item in report["results"] if item["path"] ==
                     path and item["case"] == case) for path in trial.PATHS]
        rows = [json.loads(Path(item["run"]["trace"]["path"]).read_text())
                for item in pair]
        keep = lambda rs: [(row["stage"], row["time_s"],
                            row.get("actor_packet"), row.get("ape_xy_m"),
                            row.get("actor_decision")) for row in rs
                           if row.get("time_s", 0) <=
                           report["identical_path_through_s"]+1e-5 and
                           row["stage"] in ("post_alarm_range_watch",
                                            "pre_show_restage_step")]
        assert keep(rows[0]) == keep(rows[1])
        prefix_checks.append({"case": case,
                              "matching_actor_rows": len(keep(rows[0]))})
    fixture = {name: room_gap(end, report, source, scenario)
               for name, end in trial.PATHS.items()}
    start = np.asarray(source["summary"]["ape_after_episode_xy_m"])
    final = trial.TURNED_END_XY
    t0 = report["checkpoint_s"]
    resume_s = next(item["run"]["resume_time_s"] for item in report["results"]
                    if item["path"] == "north_turn_holdout")
    finish_s = t0+trial.MOTION["move_begin_after_checkpoint_s"]+\
        trial.MOTION["move_duration_s"]
    # No .25 m/s command from the resume pose can have 5 cm proxy clearance
    # from this final center by the time the prop arrives, even along an
    # unobstructed straight line pointing directly away from it.
    best_center_separation = (float(np.linalg.norm(start-final))+
                              trial.v47.v44.SPEED_MPS*(finish_s-resume_s))
    required = .17+trial.v46.PROP_RADIUS_M+.05
    assert best_center_separation < required
    audit = {"passed": True, "rows": checked,
             "matched_pre_turn_actor_history": prefix_checks,
             "prop_static_fixture_clearance": fixture,
             "resume_to_prop_finish_reachability_upper_bound": {
                 "initial_center_distance_m": round(float(np.linalg.norm(
                     start-final)), 5),
                 "duration_s": round(finish_s-resume_s, 5),
                 "maximum_speed_mps": trial.v47.v44.SPEED_MPS,
                 "best_possible_center_separation_m": round(
                     best_center_separation, 5),
                 "center_separation_needed_for_5cm_gap_m": required},
             "scope": "Exact V36 checkpoint, re-rendered low/mid packets, identical pre-turn actor rows, independent 20 ms person/prop gaps, static fixture screen, full video decode and evaluator-only post-stop continuation. The inherited staging path penetrates the island and is not a fixture-feasible physical scenario."}
    path = trial.OUT / "audit.json"
    path.write_text(json.dumps(audit, indent=2)+"\n")
    print(path)


if __name__ == "__main__":
    main()
