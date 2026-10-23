"""Independent packet, multi-surface response, geometry and video audit V63."""

from __future__ import annotations

import json
import math
from pathlib import Path

import mujoco
import numpy as np

import audit_owner_approach_aware_meeting as choice_audit
import audit_owner_joint_hazard_response as media_audit
import audit_owner_low_prop_scan_planes as sensor_audit
import owner_multi_surface_escape as trial


def independent_groups(packet, pose):
    groups, current = [], []
    for ray, distance in enumerate(packet):
        if distance is None or distance > 1.5:
            continue
        theta = math.radians(-180.+5.*ray)
        point = np.asarray(pose)+distance*np.array([
            math.cos(theta), math.sin(theta)])
        if current and (ray-current[-1][0] != 1 or
                        np.linalg.norm(point-current[-1][1]) > .25):
            groups.append(current)
            current = []
        current.append((ray, point))
    if current:
        groups.append(current)
    return [(group[0][0], group[-1][0],
             np.mean([point for _, point in group], axis=0))
            for group in groups if len(group) >= 3]


def check(item, report, source, scenario):
    run = item["run"]
    rows = json.loads(Path(run["trace"]["path"]).read_text())
    assert media_audit.sha(run["trace"]["path"]) == run["trace"]["sha256"]
    assert media_audit.sha(run["video"]["path"]) == run["video"]["sha256"]
    assert media_audit.video_frames(run["video"]["path"], (528, 1040, 3)) == \
        run["video"]["frames"]
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
    end_xy = trial.PATHS[item["path"]]
    steps = [row for row in rows if row["stage"] == "pre_show_restage_step"]
    watch = [row for row in rows if row["stage"] ==
             "second_inference_range_watch"]
    assert len(steps) == 24 and len(watch) == trial.runner.FIXED_MONITOR_TICKS
    previous_pose = np.asarray(steps[-1]["ape_xy_m"])
    previous_packet = None
    previous_time = None
    previous_groups = None
    alarm_s = None
    direction = None
    commanded = 0.
    first_cluster_focus = None
    for index, row in enumerate(watch):
        t = row["time_s"]
        pose = np.asarray(row["ape_xy_m"])
        person = trial.runner.human_xy(
            trial.v54.CASE, summary, t, run["request_time_s"],
            run["second_inference_start_s"])
        trial.runner.v35.two.set_distractor(model, person)
        model.body_pos[prop_body, :2] = trial.v59.v50.prop_at(t, t0, end_xy)
        model.body_pos[model.body("audit_ape").id, :2] = pose
        mujoco.mj_forward(model, data)
        observed = scanner(model, data, pose, trial.v54.CASE, 1000+index,
                           person_xy=person, hidden_return=False)
        sensor_audit.same_packet(observed, row["actor_packet"])
        groups = independent_groups(observed, pose)
        claimed_groups = row["near_body_observation"]["clusters"]
        assert len(groups) == len(claimed_groups)
        for (first, last, center), claimed in zip(groups, claimed_groups):
            assert [first, last] == claimed["rays"]
            assert np.linalg.norm(center-claimed["center_xy_m"]) < 1e-8
        if alarm_s is None and row["near_body_observation"]["alarm"]:
            alarm_s = t
            focus = row["near_body_observation"]["focus"]
            assert focus is not None and previous_groups is not None
            center = next(center for first, last, center in groups
                          if [first, last] == focus["rays"])
            old = min(previous_groups, key=lambda previous:
                      np.linalg.norm(center-previous[2]))
            velocity = (center-old[2])/(t-previous_time)
            assert np.linalg.norm(center-focus["center_xy_m"]) < 1e-8
            assert np.linalg.norm(velocity-focus["velocity_mps"]) < 1e-8
            assert np.linalg.norm(velocity) >= .20
            tau = float(np.clip(-np.dot(center-pose, velocity)/
                                np.dot(velocity, velocity), 0., .8))
            forecast = float(np.linalg.norm(center+velocity*tau-pose))
            assert abs(forecast-focus["predicted_surface_to_center_m"]) < 1e-8
            assert forecast < .40
            scores = {}
            endpoints = trial.v51.v41.JointHazardResponse.endpoints(
                observed, pose)
            for name, candidate_direction in trial.DIRECTIONS.items():
                future = [pose+candidate_direction*.25*u
                          for u in np.arange(0., .80001, .1)]
                path_clear = all(np.min(np.linalg.norm(
                    endpoints-position, axis=1)) >= .27 for position in future)
                score = min(float(np.linalg.norm(
                    center+velocity*u-position))-.17
                    for u, position in zip(np.arange(0., .80001, .1), future))
                scores[name] = (path_clear, score)
            eligible = [name for name, (clear, _) in scores.items() if clear]
            expected = max(eligible, key=lambda name: (
                scores[name][1], name != "hold")) if eligible else None
            assert expected == item["response_direction"] == "north"
            direction = trial.DIRECTIONS[expected]
            first_cluster_focus = focus["rays"]
        command = row["near_body_command"]
        if command is not None and command["commanded_step_m"] > 0.:
            step = command["commanded_step_m"]
            assert alarm_s is not None and t > alarm_s
            assert previous_packet is not None and t-previous_time <= .35
            assert step <= .25*(t-previous_time)+1e-8
            assert np.linalg.norm(pose-(previous_pose+direction*step)) < 1e-8
            endpoints = trial.v51.v41.JointHazardResponse.endpoints(
                previous_packet, previous_pose)
            nearest = float(np.min(np.linalg.norm(endpoints-pose, axis=1)))
            assert nearest >= .27
            assert abs(nearest-command["minimum_endpoint_gap_m"]) < 1e-8
            commanded += step
        else:
            assert np.linalg.norm(pose-previous_pose) < 1e-8
        previous_pose, previous_packet = pose, observed
        previous_groups, previous_time = groups, t
    assert alarm_s == item["alarm_s"]
    assert abs(commanded-item["commanded_m"]) < 1e-8
    person_min = prop_min = float("inf")
    for sample in run["evaluator_gap_samples"]:
        t = sample["time_s"]
        pose = trial.runner.actor_ape_at(summary, rows, t)
        assert np.linalg.norm(pose-sample["ape_xy_m"]) < 1e-8
        person = trial.runner.human_xy(
            trial.v54.CASE, summary, t, run["request_time_s"],
            run["second_inference_start_s"])
        model.body_pos[prop_body, :2] = trial.v59.v50.prop_at(t, t0, end_xy)
        gap = trial.runner.geometry.gap(model, data, pose, person)
        prop = trial.runner.prop_gap(model, data, pose)
        assert abs(gap-sample["gap_m"]) < .00011
        assert abs(prop-sample["prop_gap_m"]) < .00011
        person_min, prop_min = min(person_min, gap), min(prop_min, prop)
    assert abs(person_min-run["minimum_proxy_gap_m"]) < .00011
    assert abs(prop_min-run["minimum_prop_gap_m"]) < .00011
    assert person_min >= .05 and prop_min >= .05
    return {"path": item["path"], "choice": choice["choice"],
            "alarm_s": alarm_s, "focus_rays": first_cluster_focus,
            "response_direction": item["response_direction"],
            "commanded_m": round(commanded, 5),
            "minimum_person_gap_m": round(person_min, 5),
            "minimum_prop_gap_m": round(prop_min, 5),
            "video_frames": run["video"]["frames"],
            "outcome": run["outcome"]}


def main():
    report = json.loads((trial.OUT / "report.json").read_text())
    assert report["schema"] == "robot-learning-lab.ape-multi-surface-escape.v1"
    for key, path in (("source_sha256", trial.__file__),
                      ("v61_report_sha256", trial.v61.OUT / "report.json"),
                      ("scene_sha256", trial.runner.v35.moving.SCENE_XML),
                      ("current_design_pointer_sha256",
                       trial.ROOT / "robot/ape-current-design.json")):
        assert report[key] == trial.v52.sha(path)
    assert json.loads((trial.v61.OUT / "audit.json").read_text())["passed"]
    source = trial.runner.original()
    policy = json.loads(trial.runner.joined.CONFIG.read_text())
    scenario = next(x for x in policy["cases"] if x["id"] == "late_crossing")
    rows = [check(item, report, source, scenario) for item in report["results"]]
    assert rows[0]["alarm_s"] is None and rows[0]["commanded_m"] == 0.
    assert rows[1]["alarm_s"] == rows[2]["alarm_s"] == 24.61078
    assert rows[2]["minimum_prop_gap_m"] >= .05
    comparison = json.loads((trial.OUT / "comparison.json").read_text())
    video = comparison["video"]
    assert comparison["v63_report_sha256"] == trial.v52.sha(
        trial.OUT / "report.json")
    assert media_audit.sha(video["path"]) == video["sha256"]
    assert media_audit.video_frames(video["path"], (620, 1080, 3)) == \
        video["frames"]
    result = {"passed": True, "rows": rows,
              "comparison_video_frames": video["frames"],
              "scope": "Rebuilt all 36 fixed-clock low/mid packets, simultaneous endpoint clusters, moving focus and directional forecast at first alarm, prior-packet command geometry, 20 ms person/prop gaps and full-video decodes. V63 is development-tuned on the V61 failure, not a held-out or physical-body safety result."}
    (trial.OUT / "audit.json").write_text(json.dumps(result, indent=2)+"\n")
    print(trial.OUT / "audit.json")


if __name__ == "__main__":
    main()
