"""Independent range, meeting-path, scene-gap and media audit for V55."""

from __future__ import annotations

import json
from pathlib import Path

import mujoco
import numpy as np

import audit_owner_fixed_clock_pair as matched
import audit_owner_low_prop_scan_planes as v46_audit
import owner_longer_meeting_pose as trial


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
    end_xy = trial.v50.PATHS[item["path"]]
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
    initial_gap = min(float(np.min(np.linalg.norm(
        endpoints-(start+trial.DIRECTION*distance), axis=1)))
        for distance in np.linspace(0., trial.TARGET_M, 7))
    assert initial_gap >= report["minimum_endpoint_gap_m"]
    steps = [row for row in rows if row["stage"] == "pre_show_restage_step"]
    assert len(steps) == item["restage_steps"] == 24
    old_pose, old_packet = np.asarray(start), initial
    for index, row in enumerate(steps):
        t = row["time_s"]
        pose = np.asarray(row["ape_xy_m"])
        step = row["commanded_step_m"]
        assert 0 < step <= trial.SPEED_MPS*trial.runner.HOLD_PERIOD_S+1e-9
        assert np.linalg.norm(pose-(old_pose+trial.DIRECTION*step)) < 1e-8
        decision = row["actor_decision"]
        assert decision["choice"] == "southwest_60cm"
        assert abs(decision["initial_corridor_gap_m"]-initial_gap) < 1e-8
        old_endpoints = trial.v51.v41.JointHazardResponse.endpoints(
            old_packet, old_pose)
        nearest = float(np.min(np.linalg.norm(old_endpoints-pose, axis=1)))
        assert abs(decision["minimum_endpoint_gap_m"]-nearest) < 1e-8
        assert nearest >= trial.MIN_ENDPOINT_GAP_M
        person = trial.runner.human_xy(trial.v54.CASE, summary, t,
                                       run["request_time_s"], None)
        trial.runner.v35.two.set_distractor(model, person)
        model.body_pos[prop_body, :2] = trial.v50.prop_at(t, t0, end_xy)
        model.body_pos[model.body("audit_ape").id, :2] = pose
        mujoco.mj_forward(model, data)
        v46_audit.same_packet(scanner(model, data, pose, trial.v54.CASE,
                                       500+index), row["actor_packet"])
        old_pose, old_packet = pose, row["actor_packet"]
    assert abs(np.linalg.norm(old_pose-start)-trial.TARGET_M) < 1e-8
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
    assert minimum_scene_gap > 0.
    assert run["outcome"] == "new_show_metric_item_seen_unattributed"
    capture = next(row for row in rows if row["stage"] ==
                   "fresh_new_show_capture")
    assert matched.prior.sha(capture["raw_rgbd"]["path"]) == \
        capture["raw_rgbd"]["sha256"]
    candidate = next(row for row in rows if row["stage"] ==
                     "fresh_item_candidate")
    assert candidate["depth_valid_pixels"] >= 50
    item_result = next(row for row in rows if row["stage"] ==
                       "new_show_metric_item_seen")
    assert item_result["teacher_identity"] == "unattributed_new_session"
    assert item_result["item_position_error_evaluator_m"] < .05
    return {**matched_score, "restage_steps": len(steps),
            "meeting_xy_m": old_pose.tolist(),
            "minimum_scene_geom_gap_m": round(minimum_scene_gap, 5),
            "item_valid_depth_pixels": candidate["depth_valid_pixels"],
            "item_center_error_m": round(
                item_result["item_position_error_evaluator_m"], 5)}


def main():
    report_path = trial.OUT / "report.json"
    report = json.loads(report_path.read_text())
    assert report["schema"] == "robot-learning-lab.ape-owner-longer-meeting-pose.v1"
    for key, target in (("source_sha256", trial.__file__),
                        ("runner_source_sha256", trial.runner.__file__),
                        ("v54_report_sha256", trial.v54.OUT / "report.json"),
                        ("scene_sha256", trial.runner.v35.moving.SCENE_XML),
                        ("v36_report_sha256", trial.runner.SOURCE),
                        ("current_design_pointer_sha256",
                         trial.ROOT / "robot/ape-current-design.json")):
        assert report[key] == matched.prior.sha(target), key
    v54_audit = json.loads((trial.v54.OUT / "audit.json").read_text())
    assert v54_audit["passed"]
    source = trial.runner.original()
    policy = json.loads(trial.runner.joined.CONFIG.read_text())
    scenario = next(x for x in policy["cases"] if x["id"] == "late_crossing")
    results = [check(item, report, source, scenario)
               for item in report["results"]]
    audit = {"passed": True, "rows": results,
             "scope": "Exact V36 checkpoint; independently regenerated hold, all 24 longer-restage and fixed-clock inference low/mid packets; checked each restage and escape command from the prior packet; 20 ms person/prop and ape-scene geometry; fresh RGB-D capture hash, depth evidence, item error and full scene/flank videos. One fixed candidate and kinematic body/prop, not selected-body qualification."}
    path = trial.OUT / "audit.json"
    path.write_text(json.dumps(audit, indent=2)+"\n")
    print(path)


if __name__ == "__main__":
    main()
