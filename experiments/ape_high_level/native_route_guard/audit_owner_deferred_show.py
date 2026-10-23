"""Independent packet, clearance, no-claim and video audit for V60."""

from __future__ import annotations

import json
from pathlib import Path

import mujoco
import numpy as np

import audit_owner_approach_aware_meeting as v59_audit
import audit_owner_low_prop_scan_planes as sensor_audit
import audit_owner_joint_hazard_response as media_audit
import owner_deferred_show as trial


def check(item, report, source, scenario, *, require_margin=True):
    run = item["run"]
    rows = json.loads(Path(run["trace"]["path"]).read_text())
    assert media_audit.sha(run["trace"]["path"]) == run["trace"]["sha256"]
    assert media_audit.sha(run["video"]["path"]) == run["video"]["sha256"]
    assert media_audit.video_frames(run["video"]["path"], (528, 1040, 3)) == \
        run["video"]["frames"]
    assert run["outcome"] == "deferred_owner_show_no_item_claim"
    assert run["second_inference_start_s"] is None
    assert run["fresh_item_capture_count"] == 0
    assert not any(row["stage"].startswith(("fresh_", "second_inference"))
                   for row in rows)
    model = mujoco.MjModel.from_xml_path(str(trial.runner.v35.moving.SCENE_XML))
    data = mujoco.MjData(model)
    prop_body = model.body("route_obstacle").id
    prop_geom = model.geom("route_obstacle_geom").id
    model.geom_size[prop_geom, 1] = trial.v51.v46.PROP_HALF_HEIGHT_M
    model.body_pos[prop_body, 2] = trial.v51.v46.PROP_CENTER_Z_M
    summary = source["summary"]
    start, _, t0, source_hash, endpoint = trial.runner.geometry_at(
        summary, source["result"]["result"]["episode"], scenario,
        model, data)
    assert t0 == report["checkpoint_s"] == run["checkpoint_time_s"]
    assert source_hash == run["source_v36_report_sha256"]
    sensor_audit.same_packet(rows[0]["checkpoint_range_packet"], endpoint)
    scanner = trial.v51.v46.plane_packet(trial.runner.observed_packet,
                                          (.05, .30))
    end_xy = trial.v59.v56.PATHS[item["path"]]
    hold = [row for row in rows if row["stage"] == "post_alarm_range_watch"]
    steps = [row for row in rows if row["stage"] == "pre_show_restage_step"]
    deferred = [row for row in rows if row["stage"] == "deferred_range_watch"]
    assert hold[-1]["actor_action"] == \
        "resume_after_three_fresh_clear_packets"
    assert len(steps) == 24 and len(deferred) >= 35
    assert v59_audit.choice_from_rows(rows)["choice"] == "west_60cm"
    decision = next(row for row in rows if row["stage"] == "defer_owner_show")
    assert decision["no_fresh_item_capture"]
    assert decision["actor_meeting_choice"] == "west_60cm"
    assert abs(run["scripted_unprompted_return_clock_s"]-
               (decision["time_s"]+.6)) < 1e-8
    for index, row in enumerate(hold):
        t = row["time_s"]
        person = trial.runner.human_xy(trial.v54.CASE, summary, t,
                                       run["request_time_s"], None)
        trial.runner.v35.two.set_distractor(model, person)
        model.body_pos[prop_body, :2] = trial.v50.prop_at(t, t0, end_xy)
        model.body_pos[model.body("audit_ape").id, :2] = start
        mujoco.mj_forward(model, data)
        sensor_audit.same_packet(scanner(model, data, start,
                                        trial.v54.CASE, index),
                                 row["actor_packet"])
    for index, row in enumerate(steps):
        t = row["time_s"]
        pose = np.asarray(row["ape_xy_m"])
        person = trial.runner.human_xy(trial.v54.CASE, summary, t,
                                       run["request_time_s"], None)
        trial.runner.v35.two.set_distractor(model, person)
        model.body_pos[prop_body, :2] = trial.v50.prop_at(t, t0, end_xy)
        model.body_pos[model.body("audit_ape").id, :2] = pose
        mujoco.mj_forward(model, data)
        sensor_audit.same_packet(scanner(model, data, pose,
                                        trial.v54.CASE, 500+index),
                                 row["actor_packet"])
    final_pose = np.asarray(steps[-1]["ape_xy_m"])
    for index, row in enumerate(deferred):
        t = row["time_s"]
        person = trial.runner.human_xy(
            trial.v54.CASE, summary, t, run["request_time_s"],
            run["scripted_unprompted_return_clock_s"])
        trial.runner.v35.two.set_distractor(model, person)
        model.body_pos[prop_body, :2] = trial.v50.prop_at(t, t0, end_xy)
        model.body_pos[model.body("audit_ape").id, :2] = final_pose
        mujoco.mj_forward(model, data)
        sensor_audit.same_packet(scanner(model, data, final_pose,
                                        trial.v54.CASE, 1000+index,
                                        person_xy=person),
                                 row["actor_packet"])
        assert np.linalg.norm(np.asarray(row["ape_xy_m"])-final_pose) < 1e-8
    person_min = prop_min = float("inf")
    for sample in run["evaluator_gap_samples"]:
        t = sample["time_s"]
        pose = trial.runner.actor_ape_at(summary, rows, t)
        assert np.linalg.norm(pose-sample["ape_xy_m"]) < 1e-8
        person = trial.runner.human_xy(
            trial.v54.CASE, summary, t, run["request_time_s"],
            run["scripted_unprompted_return_clock_s"])
        model.body_pos[prop_body, :2] = trial.v50.prop_at(t, t0, end_xy)
        person_gap = trial.runner.geometry.gap(model, data, pose, person)
        prop_gap = trial.runner.prop_gap(model, data, pose)
        assert abs(person_gap-sample["gap_m"]) < .00011
        assert abs(prop_gap-sample["prop_gap_m"]) < .00011
        person_min, prop_min = min(person_min, person_gap), min(prop_min, prop_gap)
    assert abs(person_min-run["minimum_proxy_gap_m"]) < .00011
    assert abs(prop_min-run["minimum_prop_gap_m"]) < .00011
    if require_margin:
        assert person_min >= .05 and prop_min >= .05
    return {"path": item["path"], "choice": "west_60cm",
            "minimum_person_gap_m": round(person_min, 5),
            "minimum_prop_gap_m": round(prop_min, 5),
            "deferred_watch_packets": len(deferred),
            "video_frames": run["video"]["frames"],
            "fresh_item_captures": 0}


def main():
    report = json.loads((trial.OUT / "report.json").read_text())
    assert report["schema"] == "robot-learning-lab.ape-deferred-owner-show.v1"
    for key, path in (("source_sha256", trial.__file__),
                      ("v59_report_sha256", trial.v59.OUT / "report.json"),
                      ("runner_source_sha256", trial.runner.__file__),
                      ("scene_sha256", trial.runner.v35.moving.SCENE_XML),
                      ("current_design_pointer_sha256",
                       trial.ROOT / "robot/ape-current-design.json")):
        assert report[key] == trial.v52.sha(path)
    assert json.loads((trial.v59.OUT / "audit.json").read_text())["passed"]
    source = trial.runner.original()
    policy = json.loads(trial.runner.joined.CONFIG.read_text())
    scenario = next(x for x in policy["cases"] if x["id"] == "late_crossing")
    checked = [check(item, report, source, scenario)
               for item in report["results"]]
    comparison = json.loads((trial.OUT / "comparison.json").read_text())
    assert comparison["v56_report_sha256"] == trial.v52.sha(
        trial.v59.v56.OUT / "report.json")
    assert comparison["v60_report_sha256"] == trial.v52.sha(
        trial.OUT / "report.json")
    for name, video in comparison["comparisons"].items():
        assert name in ("new_west_south_turn", "outbound_control")
        assert media_audit.sha(video["path"]) == video["sha256"]
        assert media_audit.video_frames(video["path"], (620, 1080, 3)) == \
            video["frames"]
    result = {"passed": True, "rows": checked,
              "comparison_videos": list(comparison["comparisons"]),
              "scope": "Independently regenerated every range packet in the hold, 24 actor-selected west steps and deferred 10 Hz watch; recomputed 20 ms MuJoCo person/prop gaps; verified no image capture or item claim and decoded each full episode video. Unprompted person return is scripted evaluator disturbance; body/prop are historical kinematic proxies."}
    (trial.OUT / "audit.json").write_text(json.dumps(result, indent=2)+"\n")
    print(trial.OUT / "audit.json")


if __name__ == "__main__":
    main()
