"""Independent V64 deferred sensor/action, no-claim, geometry and media audit."""

from __future__ import annotations

import json
from pathlib import Path

import mujoco
import numpy as np

import audit_owner_approach_aware_meeting as choice_audit
import audit_owner_joint_hazard_response as media_audit
import audit_owner_low_prop_scan_planes as sensor_audit
import audit_owner_multi_surface_escape as cluster_audit
import owner_deferred_multi_surface as trial


def checked_pose(rows, summary, time_s):
    start = np.asarray(summary["ape_after_episode_xy_m"])
    resume = next(row["time_s"] for row in rows
                  if row["stage"] == "post_alarm_range_watch" and
                  row["actor_action"] ==
                  "resume_after_three_fresh_clear_packets")
    stages = [(resume, start)]
    stages.extend((row["time_s"], np.asarray(row["ape_xy_m"]))
                  for row in rows if row["stage"] in (
                      "pre_show_restage_step", "deferred_range_watch"))
    if time_s <= resume:
        return start
    for (left_t, left), (right_t, right) in zip(stages, stages[1:]):
        if time_s <= right_t:
            if right_t == left_t:
                return right
            u = np.clip((time_s-left_t)/(right_t-left_t), 0., 1.)
            return left+u*(right-left)
    return stages[-1][1]


def check(item, report, source, scenario, *, route_fn=None,
          monitor_scanner=None, require_margin=True):
    run = item["run"]
    rows = json.loads(Path(run["trace"]["path"]).read_text())
    assert media_audit.sha(run["trace"]["path"]) == run["trace"]["sha256"]
    assert media_audit.sha(run["video"]["path"]) == run["video"]["sha256"]
    assert media_audit.video_frames(run["video"]["path"], (528, 1040, 3)) == \
        run["video"]["frames"]
    assert run["outcome"] == "deferred_owner_show_no_item_claim"
    assert run["fresh_item_capture_count"] == 0
    assert run["second_inference_start_s"] is None
    assert not any(row["stage"].startswith(("fresh_", "second_inference"))
                   for row in rows)
    choice = choice_audit.choice_from_rows(rows)
    assert choice["choice"] == "west_60cm"
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
    monitor_scanner = monitor_scanner if monitor_scanner is not None else scanner
    end_xy = (trial.PATHS[item["path"]] if route_fn is None else None)
    obstacle_at = (route_fn if route_fn is not None else
                   lambda t: trial.v60.v50.prop_at(t, t0, end_xy))
    hold = [row for row in rows if row["stage"] == "post_alarm_range_watch"]
    steps = [row for row in rows if row["stage"] == "pre_show_restage_step"]
    watch = [row for row in rows if row["stage"] == "deferred_range_watch"]
    assert hold[-1]["actor_action"] == \
        "resume_after_three_fresh_clear_packets"
    assert len(steps) == 24 and len(watch) == 42
    decision = next(row for row in rows if row["stage"] == "defer_owner_show")
    assert decision["no_fresh_item_capture"]
    assert abs(run["scripted_unprompted_return_clock_s"]-
               (decision["time_s"]+.6)) < 1e-8
    for stage_rows, offset in ((hold, 0), (steps, 500)):
        for index, row in enumerate(stage_rows):
            t = row["time_s"]
            pose = (start if offset == 0 else np.asarray(row["ape_xy_m"]))
            person = trial.runner.human_xy(trial.v54.CASE, summary, t,
                                           run["request_time_s"], None)
            trial.runner.v35.two.set_distractor(model, person)
            model.body_pos[prop_body, :2] = obstacle_at(t)
            model.body_pos[model.body("audit_ape").id, :2] = pose
            mujoco.mj_forward(model, data)
            sensor_audit.same_packet(scanner(model, data, pose,
                                            trial.v54.CASE, offset+index),
                                     row["actor_packet"])
    previous_pose = np.asarray(steps[-1]["ape_xy_m"])
    previous_packet = None
    previous_groups = None
    previous_t = None
    direction = None
    alarm_s = None
    commanded = 0.
    for index, row in enumerate(watch):
        t = row["time_s"]
        pose = np.asarray(row["ape_xy_m"])
        person = trial.runner.human_xy(
            trial.v54.CASE, summary, t, run["request_time_s"],
            run["scripted_unprompted_return_clock_s"])
        trial.runner.v35.two.set_distractor(model, person)
        model.body_pos[prop_body, :2] = obstacle_at(t)
        model.body_pos[model.body("audit_ape").id, :2] = pose
        mujoco.mj_forward(model, data)
        packet = monitor_scanner(model, data, pose, trial.v54.CASE,
                                 1000+index, person_xy=person)
        sensor_audit.same_packet(packet, row["actor_packet"])
        groups = cluster_audit.independent_groups(packet, pose)
        claimed = row["near_body_observation"]["clusters"]
        assert len(groups) == len(claimed)
        for (first, last, center), cluster in zip(groups, claimed):
            assert [first, last] == cluster["rays"]
            assert np.linalg.norm(center-cluster["center_xy_m"]) < 1e-8
        if alarm_s is None and row["near_body_observation"]["alarm"]:
            alarm_s = t
            focus = row["near_body_observation"]["focus"]
            assert focus is not None and previous_groups is not None
            center = next(center for first, last, center in groups
                          if [first, last] == focus["rays"])
            old = min(previous_groups, key=lambda previous:
                      np.linalg.norm(center-previous[2]))
            velocity = (center-old[2])/(t-previous_t)
            assert np.linalg.norm(velocity-focus["velocity_mps"]) < 1e-8
            tau = float(np.clip(-np.dot(center-pose, velocity)/
                                np.dot(velocity, velocity), 0., .8))
            forecast = float(np.linalg.norm(center+velocity*tau-pose))
            assert abs(forecast-focus["predicted_surface_to_center_m"]) < 1e-8
            assert forecast < .40
            endpoints = trial.v51.v41.JointHazardResponse.endpoints(
                packet, pose)
            scores = {}
            for name, candidate_direction in trial.v63.DIRECTIONS.items():
                horizon = np.arange(0., .80001, .1)
                future = [pose+candidate_direction*.25*u for u in horizon]
                clear = all(np.min(np.linalg.norm(
                    endpoints-position, axis=1)) >= .27 for position in future)
                margin = min(float(np.linalg.norm(
                    center+velocity*u-position))-.17
                    for u, position in zip(horizon, future))
                scores[name] = (clear, margin)
            eligible = [name for name, (clear, _) in scores.items() if clear]
            expected = max(eligible, key=lambda name: (
                scores[name][1], name != "hold")) if eligible else None
            assert expected == row["near_body_observation"]["chosen_response"]
            direction = trial.v63.DIRECTIONS[expected]
        command = row["near_body_command"]
        if command["commanded_step_m"] > 0.:
            step = command["commanded_step_m"]
            assert alarm_s is not None and t > alarm_s
            assert previous_packet is not None and t-previous_t <= .35
            assert step <= .25*(t-previous_t)+1e-8
            assert np.linalg.norm(pose-(previous_pose+direction*step)) < 1e-8
            endpoints = trial.v51.v41.JointHazardResponse.endpoints(
                previous_packet, previous_pose)
            nearest = float(np.min(np.linalg.norm(endpoints-pose, axis=1)))
            assert nearest >= .27
            assert abs(nearest-command["minimum_endpoint_gap_m"]) < 1e-8
            commanded += step
        else:
            assert np.linalg.norm(pose-previous_pose) < 1e-8
        previous_pose, previous_packet = pose, packet
        previous_groups, previous_t = groups, t
    assert (None if alarm_s is None else round(alarm_s, 5)) == (
        None if run["near_body_alarm_s"] is None else
        round(run["near_body_alarm_s"], 5))
    assert abs(commanded-run["watch_commanded_m"]) < 1e-8
    person_min = prop_min = float("inf")
    for sample in run["evaluator_gap_samples"]:
        t = sample["time_s"]
        pose = checked_pose(rows, summary, t)
        assert np.linalg.norm(pose-sample["ape_xy_m"]) < 1e-8
        person = trial.runner.human_xy(
            trial.v54.CASE, summary, t, run["request_time_s"],
            run["scripted_unprompted_return_clock_s"])
        model.body_pos[prop_body, :2] = obstacle_at(t)
        gap = trial.runner.geometry.gap(model, data, pose, person)
        prop = trial.runner.prop_gap(model, data, pose)
        assert abs(gap-sample["gap_m"]) < .00011
        assert abs(prop-sample["prop_gap_m"]) < .00011
        person_min, prop_min = min(person_min, gap), min(prop_min, prop)
    assert abs(person_min-run["minimum_proxy_gap_m"]) < .00011
    assert abs(prop_min-run["minimum_prop_gap_m"]) < .00011
    assert person_min >= .05
    if require_margin:
        assert prop_min >= .05
    return {"path": item["path"], "choice": choice["choice"],
            "alarm_s": None if alarm_s is None else round(alarm_s, 5),
            "commanded_m": round(commanded, 5),
            "minimum_person_gap_m": round(person_min, 5),
            "minimum_prop_gap_m": round(prop_min, 5),
            "watch_packets": len(watch),
            "fresh_item_captures": 0,
            "video_frames": run["video"]["frames"]}


def main():
    report = json.loads((trial.OUT / "report.json").read_text())
    assert report["schema"] == "robot-learning-lab.ape-deferred-multi-surface.v1"
    for key, path in (("source_sha256", trial.__file__),
                      ("v60_source_sha256", trial.v60.__file__),
                      ("v63_source_sha256", trial.v63.__file__),
                      ("scene_sha256", trial.runner.v35.moving.SCENE_XML),
                      ("current_design_pointer_sha256",
                       trial.ROOT / "robot/ape-current-design.json")):
        assert report[key] == trial.v52.sha(path)
    assert json.loads((trial.v60.OUT / "audit.json").read_text())["passed"]
    assert json.loads((trial.v63.OUT / "audit.json").read_text())["passed"]
    source = trial.runner.original()
    policy = json.loads(trial.runner.joined.CONFIG.read_text())
    scenario = next(x for x in policy["cases"] if x["id"] == "late_crossing")
    checked = [check(item, report, source, scenario)
               for item in report["results"]]
    assert checked[0]["alarm_s"] is None and checked[0]["commanded_m"] == 0.
    assert checked[-1]["alarm_s"] == 24.61078
    traces = [json.loads(Path(item["run"]["trace"]["path"]).read_text())
              for item in report["results"]]
    until = report["checkpoint_s"]+trial.v60.v50.WAYPOINTS[-1][0]
    prefixes = [[row for row in rows if row["stage"] in (
        "post_alarm_range_watch", "pre_show_restage_step",
        "deferred_range_watch") and row["time_s"] <= until+.000005]
        for rows in traces]
    assert len(set(map(len, prefixes))) == 1
    for aligned in zip(*prefixes):
        for row in aligned[1:]:
            assert row["stage"] == aligned[0]["stage"]
            assert row["time_s"] == aligned[0]["time_s"]
            sensor_audit.same_packet(row["actor_packet"],
                                     aligned[0]["actor_packet"])
            assert row.get("ape_xy_m") == aligned[0].get("ape_xy_m")
    comparison = json.loads((trial.OUT / "comparison.json").read_text())
    video = comparison["video"]
    assert comparison["v64_report_sha256"] == trial.v52.sha(
        trial.OUT / "report.json")
    assert media_audit.sha(video["path"]) == video["sha256"]
    assert media_audit.video_frames(video["path"], (620, 1080, 3)) == \
        video["frames"]
    result = {"passed": True, "rows": checked,
              "matching_pre_turn_actor_rows_per_path": len(prefixes[0]),
              "comparison_video_frames": video["frames"],
              "scope": "Regenerated all hold/restage/deferred low+mid packets, actor meeting choice, simultaneous cluster and first alarm forecast, prior-packet escape steps, 20 ms person/prop gaps, no-image/no-claim branch and decoded full/comparison videos. Provisional exact-pose kinematic proxy; no physical safety or sensor qualification."}
    (trial.OUT / "audit.json").write_text(json.dumps(result, indent=2)+"\n")
    print(trial.OUT / "audit.json")


if __name__ == "__main__":
    main()
