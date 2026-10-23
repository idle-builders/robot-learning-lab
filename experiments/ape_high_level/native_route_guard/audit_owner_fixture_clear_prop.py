"""Audit V50's fixture-clear prop path and complete owner continuation."""

from __future__ import annotations

import json
import math
from pathlib import Path

import mujoco
import numpy as np

import audit_owner_joint_hazard_response as prior
import audit_owner_low_prop_scan_planes as v46_audit
import audit_owner_scanned_meeting_pose as v44_audit
import owner_fixture_clear_prop as trial
import render_owner_fixture_clear_prop as renderer


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
    assert expected_novel == item["novel_forward_returns"]
    scores = v44_audit.independent_candidates(hold[-1]["actor_packet"], start)
    eligible = [name for name, score in scores.items() if score["eligible"]]
    expected_choice = max(eligible, key=lambda name: scores[name][
        "robust_score_m"]) if eligible else None
    if expected_novel:
        northeast = trial.v47.v44.endpoint_score(
            hold[-1]["actor_packet"], start, trial.v47.NORTHEAST)
        if northeast["eligible"]:
            expected_choice = "northeast"
    assert item["choice"] == expected_choice
    direction = (trial.v47.NORTHEAST if expected_choice == "northeast"
                 else trial.v47.v44.CANDIDATES[expected_choice])
    steps = [row for row in rows if row["stage"] == "pre_show_restage_step"]
    assert len(steps) == item["restage_steps"]
    old_pose, old_row = np.asarray(start), hold[-1]
    for index, row in enumerate(steps):
        t = row["time_s"]
        pose = np.asarray(row["ape_xy_m"])
        distance = row["commanded_step_m"]
        assert abs(t-old_row["time_s"]-trial.v38.HOLD_PERIOD_S) < 1e-5
        assert np.linalg.norm(pose-(old_pose+direction*distance)) < 1e-8
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
    prop_finish_s = t0+trial.END_S
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
            if gap < min_gap:
                min_gap, nearest = gap, name
    return {"minimum_static_fixture_gap_m": round(min_gap, 5),
            "path_fixture_clear": min_gap >= 0.,
            "nearest_fixture": nearest}


def main():
    report = json.loads((trial.OUT / "report.json").read_text())
    assert report["schema"] == "robot-learning-lab.ape-owner-fixture-clear-prop.v1"
    for key, path in (("source_sha256", trial.__file__),
                      ("v38_source_sha256", trial.v38.__file__),
                      ("v46_source_sha256", trial.v46.__file__),
                      ("v47_source_sha256", trial.v47.__file__),
                      ("v49_source_sha256", trial.v49.__file__),
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
    assert all(row["path_fixture_clear"] and
               row["minimum_static_fixture_gap_m"] >= .05
               for row in fixture.values())
    assert report["waypoints"] == [
        {"after_checkpoint_s": t, "xy_m": xy.tolist()}
        for t, xy in trial.WAYPOINTS]
    assert report["end_after_checkpoint_s"] == trial.END_S
    comparison = json.loads((trial.OUT / "comparison.json").read_text())
    assert comparison["schema"] == "robot-learning-lab.ape-fixture-clear-video.v1"
    assert comparison["report_sha256"] == prior.sha(trial.OUT / "report.json")
    assert comparison["source_sha256"] == prior.sha(renderer.__file__)
    video = comparison["video"]
    assert video["sha256"] == prior.sha(video["path"])
    assert prior.video_frames(video["path"], tuple(video["shape"])) == \
        video["frames"]
    audit = {"passed": True, "rows": checked,
             "matched_pre_turn_actor_history": prefix_checks,
             "prop_static_fixture_clearance": fixture,
             "comparison_video": video,
             "scope": "Exact V36 checkpoint, re-rendered low/mid packets, identical pre-turn actor rows, independent 20 ms person/prop gaps, full prop-vs-fixture path screen, full video decode and evaluator-only post-stop continuation. The prop path is kinematically scripted, not pushed by simulated contact."}
    path = trial.OUT / "audit.json"
    path.write_text(json.dumps(audit, indent=2)+"\n")
    print(path)


if __name__ == "__main__":
    main()
