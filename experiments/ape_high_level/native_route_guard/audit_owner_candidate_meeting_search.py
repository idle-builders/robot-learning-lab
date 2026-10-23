"""Independent range, meeting-path, scene-gap and media audit for V56."""

from __future__ import annotations

import json
from pathlib import Path

import mujoco
import numpy as np

import audit_owner_fixed_clock_pair as matched
import audit_owner_low_prop_scan_planes as v46_audit
import owner_candidate_meeting_search as trial


def check(item, report, source, scenario):
    run = item["run"]
    rows = json.loads(Path(run["trace"]["path"]).read_text())
    model = mujoco.MjModel.from_xml_path(str(trial.runner.v35.moving.SCENE_XML))
    data = mujoco.MjData(model)
    prop_geom = model.geom("route_obstacle_geom").id
    prop_body = model.body("route_obstacle").id
    ape_geom = model.geom("audit_ape_geom").id
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
    end_xy = trial.PATHS[item["path"]]
    hold = [row for row in rows if row["stage"] == "post_alarm_range_watch"]
    assert hold[-1]["actor_action"] == \
        "resume_after_three_fresh_clear_packets"
    for index, row in enumerate(hold):
        t = row["time_s"]
        person = trial.runner.human_xy(trial.v54.CASE, summary, t,
                                       run["request_time_s"], None)
        trial.runner.v35.two.set_distractor(model, person)
        model.body_pos[prop_body, :2] = trial.v50.prop_at(t, t0, end_xy)
        model.body_pos[model.body("audit_ape").id, :2] = start
        mujoco.mj_forward(model, data)
        v46_audit.same_packet(scanner(model, data, start, trial.v54.CASE,
                                       index), row["actor_packet"])
    initial = hold[-1]["actor_packet"]
    endpoints = trial.v51.v41.JointHazardResponse.endpoints(initial, start)
    healthy = len([value for value in initial if value is not None]) >= 36
    candidate_scores = []
    for name, direction in trial.DIRECTIONS.items():
        for distance in trial.DISTANCES_M:
            target = start+direction*distance
            gaps = [float(np.min(np.linalg.norm(
                endpoints-(start+u*direction), axis=1)))
                for u in np.linspace(0., distance, 15)]
            owner_range = float(np.linalg.norm(
                trial.OWNER_SHOW_STATION_XY_M-target))
            candidate_scores.append({
                "name": name, "distance_m": distance,
                "target_xy_m": target.tolist(),
                "minimum_path_endpoint_gap_m": min(gaps),
                "held_endpoint_reserve_m": gaps[-1],
                "owner_station_range_m": owner_range,
                "eligible": bool(healthy and
                    min(gaps) >= trial.MIN_PATH_ENDPOINT_GAP_M and
                    gaps[-1] >= trial.MIN_HELD_ENDPOINT_RESERVE_M and
                    owner_range <= trial.MAX_OWNER_STATION_RANGE_M)})
    eligible = [row for row in candidate_scores if row["eligible"]]
    chosen = max(eligible, key=lambda row: (
        row["distance_m"], row["held_endpoint_reserve_m"],
        -row["owner_station_range_m"])) if eligible else None
    expected_choice = (f"{chosen['name']}_{int(chosen['distance_m']*100)}cm"
                       if chosen else None)
    assert expected_choice == item["choice"] == "southwest_60cm"
    direction = trial.DIRECTIONS[chosen["name"]]
    distance = chosen["distance_m"]
    steps = [row for row in rows if row["stage"] == "pre_show_restage_step"]
    assert len(steps) == item["restage_steps"] == 24
    old_pose, old_packet = np.asarray(start), initial
    for index, row in enumerate(steps):
        t = row["time_s"]
        pose = np.asarray(row["ape_xy_m"])
        step = row["commanded_step_m"]
        assert 0 < step <= trial.v55.SPEED_MPS*trial.runner.HOLD_PERIOD_S+1e-9
        assert np.linalg.norm(pose-(old_pose+direction*step)) < 1e-8
        decision = row["actor_decision"]
        assert decision["choice"] == expected_choice
        if decision["candidate_scores"] is not None:
            assert len(decision["candidate_scores"]) == len(candidate_scores)
            for recorded, independently_scored in zip(
                    decision["candidate_scores"], candidate_scores):
                assert recorded["name"] == independently_scored["name"]
                assert recorded["distance_m"] == independently_scored["distance_m"]
                for key in ("minimum_path_endpoint_gap_m",
                            "held_endpoint_reserve_m",
                            "owner_station_range_m"):
                    assert abs(recorded[key]-independently_scored[key]) < 1e-8
                assert recorded["eligible"] == independently_scored["eligible"]
        old_endpoints = trial.v51.v41.JointHazardResponse.endpoints(
            old_packet, old_pose)
        nearest = float(np.min(np.linalg.norm(old_endpoints-pose, axis=1)))
        assert abs(decision["minimum_endpoint_gap_m"]-nearest) < 1e-8
        assert nearest >= trial.MIN_PATH_ENDPOINT_GAP_M
        person = trial.runner.human_xy(trial.v54.CASE, summary, t,
                                       run["request_time_s"], None)
        trial.runner.v35.two.set_distractor(model, person)
        model.body_pos[prop_body, :2] = trial.v50.prop_at(t, t0, end_xy)
        model.body_pos[model.body("audit_ape").id, :2] = pose
        mujoco.mj_forward(model, data)
        v46_audit.same_packet(scanner(model, data, pose, trial.v54.CASE,
                                       500+index), row["actor_packet"])
        old_pose, old_packet = pose, row["actor_packet"]
    assert abs(np.linalg.norm(old_pose-start)-distance) < 1e-8
    assert np.linalg.norm(old_pose-item["meeting_xy_m"]) < 1e-8
    if item["policy"] == "side_only":
        commanded = 0.
    else:
        commanded = sum(row["near_body_command"]["commanded_step_m"]
                        for row in rows if row["stage"] ==
                        "second_inference_range_watch")
    derived = {**item, "case": trial.v54.CASE,
               "near_body_commanded_m": commanded}
    matched_score = matched.score_item(derived, report, source, scenario)
    minimum_scene_gap = float("inf")
    for sample in run["evaluator_gap_samples"]:
        t = sample["time_s"]
        pose = np.asarray(sample["ape_xy_m"])
        person = trial.runner.human_xy(trial.v54.CASE, summary, t,
                                       run["request_time_s"],
                                       run["second_inference_start_s"])
        trial.runner.v35.two.set_distractor(model, person)
        model.body_pos[prop_body, :2] = trial.v50.prop_at(t, t0, end_xy)
        model.body_pos[model.body("audit_ape").id, :2] = pose
        mujoco.mj_forward(model, data)
        for geom_id in range(model.ngeom):
            if geom_id == ape_geom or not model.geom_contype[geom_id] or \
                    not model.geom_conaffinity[geom_id]:
                continue
            name = model.geom(geom_id).name
            if name.startswith(("floor_room", "floor_backing_room")):
                continue
            if np.linalg.norm(data.geom_xpos[geom_id, :2]-pose) > 1.5:
                continue
            minimum_scene_gap = min(minimum_scene_gap, float(
                mujoco.mj_geomDistance(model, data, ape_geom, geom_id,
                                       2., None)))
    if item["path"] == "new_west_south_turn":
        assert minimum_scene_gap < 0.
    else:
        assert minimum_scene_gap > 0.
    capture = next(row for row in rows if row["stage"] ==
                   "fresh_new_show_capture")
    assert matched.prior.sha(capture["raw_rgbd"]["path"]) == \
        capture["raw_rgbd"]["sha256"]
    candidate = next((row for row in rows if row["stage"] ==
                      "fresh_item_candidate"), None)
    item_result = next((row for row in rows if row["stage"] ==
                        "new_show_metric_item_seen"), None)
    if item["path"] != "new_west_south_turn" and \
            item["policy"] != "nearest_always":
        assert run["outcome"] == "new_show_metric_item_seen_unattributed"
        assert candidate is not None and candidate["depth_valid_pixels"] >= 50
        assert item_result["teacher_identity"] == "unattributed_new_session"
        assert item_result["item_position_error_evaluator_m"] < .05
    else:
        assert item_result is None
    return {**matched_score, "restage_steps": len(steps),
            "meeting_xy_m": old_pose.tolist(),
            "minimum_scene_geom_gap_m": round(minimum_scene_gap, 5),
            "choice": expected_choice,
            "eligible_candidates": len(eligible),
            "item_valid_depth_pixels": (candidate["depth_valid_pixels"]
                                        if candidate else None),
            "item_center_error_m": (round(
                item_result["item_position_error_evaluator_m"], 5)
                if item_result else None)}


def main():
    report_path = trial.OUT / "report.json"
    report = json.loads(report_path.read_text())
    assert report["schema"] == "robot-learning-lab.ape-owner-candidate-meeting-search.v1"
    for key, target in (("source_sha256", trial.__file__),
                        ("runner_source_sha256", trial.runner.__file__),
                        ("v55_report_sha256", trial.v55.OUT / "report.json"),
                        ("scene_sha256", trial.runner.v35.moving.SCENE_XML),
                        ("v36_report_sha256", trial.runner.SOURCE),
                        ("current_design_pointer_sha256",
                         trial.ROOT / "robot/ape-current-design.json")):
        assert report[key] == matched.prior.sha(target), key
    v55_audit = json.loads((trial.v55.OUT / "audit.json").read_text())
    assert v55_audit["passed"]
    source = trial.runner.original()
    policy = json.loads(trial.runner.joined.CONFIG.read_text())
    scenario = next(x for x in policy["cases"] if x["id"] == "late_crossing")
    import audit_owner_fixture_clear_prop as fixture_audit
    v50_report = json.loads((trial.v50.OUT / "report.json").read_text())
    fixture = {name: fixture_audit.room_gap(end, v50_report,
                                            source, scenario)
               for name, end in trial.PATHS.items()}
    assert all(row["minimum_static_fixture_gap_m"] >= .05
               for row in fixture.values())
    old_paths = dict(trial.v50.PATHS)
    try:
        trial.v50.PATHS.update(trial.PATHS)
        results = [check(item, report, source, scenario)
                   for item in report["results"]]
    finally:
        trial.v50.PATHS.clear()
        trial.v50.PATHS.update(old_paths)
    paired = []
    preturn_end = report["checkpoint_s"]+trial.v50.WAYPOINTS[-1][0]
    for mode in ("side_only", "nearest_always", "surface_track"):
        group = [next(item for item in report["results"]
                      if item["policy"] == mode and item["path"] == name)
                 for name in trial.PATHS]
        traces = [json.loads(Path(item["run"]["trace"]["path"]).read_text())
                  for item in group]
        stages = []
        for rows in traces:
            selected = [row for row in rows if row["stage"] in (
                "post_alarm_range_watch", "pre_show_restage_step",
                "second_inference_range_watch") and
                row["time_s"] <= preturn_end+.000005]
            stages.append(selected)
        assert len(set(map(len, stages))) == 1
        for aligned in zip(*stages):
            assert len(set(row["stage"] for row in aligned)) == 1
            assert max(row["time_s"] for row in aligned)-min(
                row["time_s"] for row in aligned) < 1e-8
            for row in aligned[1:]:
                v46_audit.same_packet(aligned[0]["actor_packet"],
                                      row["actor_packet"])
                if "ape_xy_m" in aligned[0]:
                    assert np.linalg.norm(np.asarray(aligned[0]["ape_xy_m"])-
                                          row["ape_xy_m"]) < 1e-8
                assert aligned[0].get("near_body_observation") == row.get(
                    "near_body_observation")
                assert aligned[0].get("near_body_command") == row.get(
                    "near_body_command")
        paired.append({"policy": mode,
                       "matching_pre_turn_actor_rows_per_path": len(stages[0])})
    audit = {"passed": True, "rows": results,
             "prop_static_fixture_clearance": fixture,
             "paired_pre_turn_history": paired,
             "scope": "Exact V36 checkpoint; independently scored all 20 meeting candidates from the actor's resume packet; regenerated hold, 24 restage and fixed-clock inference low/mid packets; checked movement from prior packets, 20 ms ape/scene geometry, fresh-show evidence and videos. New fixture-clear path is an intentionally failed held-pose counterexample. Kinematic body/prop and provisional view range do not qualify physical safety."}
    path = trial.OUT / "audit.json"
    path.write_text(json.dumps(audit, indent=2)+"\n")
    print(path)


if __name__ == "__main__":
    main()
