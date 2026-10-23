"""Audit V46's actual low/mid ray packets and full owner-loop gaps."""

from __future__ import annotations

import json
from pathlib import Path

import mujoco
import numpy as np

import audit_owner_joint_hazard_response as prior
import owner_low_prop_scan_planes as trial
import render_owner_low_prop_scan_planes as renderer


def same_packet(left, right):
    assert len(left) == len(right) == 72
    for a, b in zip(left, right):
        assert (a is None and b is None) or (
            a is not None and b is not None and abs(a-b) < 1e-10)


def check(item, report, source, scenario):
    run = item["run"]
    rows = json.loads(Path(run["trace"]["path"]).read_text())
    assert prior.sha(run["trace"]["path"]) == run["trace"]["sha256"]
    assert prior.sha(run["video"]["path"]) == run["video"]["sha256"]
    assert prior.video_frames(run["video"]["path"], (528, 1040, 3)) == \
        run["video"]["frames"]
    model = mujoco.MjModel.from_xml_path(str(trial.v38.v35.moving.SCENE_XML))
    data = mujoco.MjData(model)
    geom_id = model.geom("route_obstacle_geom").id
    body_id = model.body("route_obstacle").id
    model.geom_size[geom_id, 1] = trial.PROP_HALF_HEIGHT_M
    model.body_pos[body_id, 2] = trial.PROP_CENTER_Z_M
    summary = source["summary"]
    start, _, t0, source_hash, endpoint = trial.v38.geometry_at(
        summary, source["result"]["result"]["episode"], scenario,
        model, data)
    assert t0 == run["checkpoint_time_s"] == report["checkpoint_s"]
    assert source_hash == run["source_v36_report_sha256"]
    same_packet(rows[0]["checkpoint_range_packet"], endpoint)
    heights = trial.PROFILES[item["profile"]]
    scanner = trial.plane_packet(trial.v38.observed_packet, heights)
    watch_before = [row for row in rows if row["stage"] ==
                    "post_alarm_range_watch"]
    assert watch_before
    for index, row in enumerate(watch_before):
        t = row["time_s"]
        person = trial.v38.human_xy(item["case"], summary, t,
                                    run["request_time_s"], None)
        trial.v38.v35.two.set_distractor(model, person)
        model.body_pos[body_id, :2] = trial.v41.v40.prop_at(
            t, t0, report["prop_motion"])
        model.body_pos[model.body("audit_ape").id, :2] = start
        mujoco.mj_forward(model, data)
        observed = scanner(model, data, start, item["case"], index)
        same_packet(observed, row["actor_packet"])
    steps = [row for row in rows if row["stage"] ==
             "pre_show_restage_step"]
    assert len(steps) == item["restage_steps"] and steps
    for index, row in enumerate(steps):
        t = row["time_s"]
        ape = np.asarray(row["ape_xy_m"])
        person = trial.v38.human_xy(item["case"], summary, t,
                                    run["request_time_s"], None)
        trial.v38.v35.two.set_distractor(model, person)
        model.body_pos[body_id, :2] = trial.v41.v40.prop_at(
            t, t0, report["prop_motion"])
        model.body_pos[model.body("audit_ape").id, :2] = ape
        mujoco.mj_forward(model, data)
        observed = scanner(model, data, ape, item["case"], 500+index)
        same_packet(observed, row["actor_packet"])
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
        model.body_pos[body_id, :2] = trial.v41.v40.prop_at(
            t, t0, report["prop_motion"])
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
    if run["outcome"] == "no_supported_meeting_pose":
        assert run["second_inference_start_s"] is None
        assert not watch
    return {"profile": item["profile"], "case": item["case"],
            "outcome": run["outcome"], "choice": item["choice"],
            "minimum_person_gap_m": round(person_min, 5),
            "minimum_prop_gap_m": round(prop_min, 5),
            "above_provisional_5cm_both": person_min >= .05 and prop_min >= .05,
            "restage_steps": len(steps),
            "full_video_frames": run["video"]["frames"]}


def main():
    report = json.loads((trial.OUT / "report.json").read_text())
    assert report["schema"] == "robot-learning-lab.ape-owner-low-prop-planes.v1"
    assert report["source_sha256"] == prior.sha(trial.__file__)
    assert report["v38_source_sha256"] == prior.sha(trial.v38.__file__)
    assert report["v44_source_sha256"] == prior.sha(trial.v44.__file__)
    assert report["scene_sha256"] == prior.sha(trial.v38.v35.moving.SCENE_XML)
    assert report["v36_report_sha256"] == prior.sha(trial.v38.SOURCE)
    assert report["current_design_pointer_sha256"] == prior.sha(
        trial.ROOT / "robot/ape-current-design.json")
    assert report["prop_geometry"] == {
        "radius_m": trial.PROP_RADIUS_M,
        "half_height_m": trial.PROP_HALF_HEIGHT_M,
        "center_z_m": trial.PROP_CENTER_Z_M}
    source = trial.v38.original()
    policy = json.loads(trial.v38.joined.CONFIG.read_text())
    scenario = next(row for row in policy["cases"] if row["id"] ==
                    "late_crossing")
    checked = [check(item, report, source, scenario)
               for item in report["results"]]
    comparison = json.loads((trial.OUT / "comparison.json").read_text())
    assert comparison["schema"] == \
        "robot-learning-lab.ape-owner-low-prop-comparison.v1"
    assert comparison["report_sha256"] == prior.sha(trial.OUT / "report.json")
    assert comparison["source_sha256"] == prior.sha(renderer.__file__)
    video = comparison["video"]
    assert video["sha256"] == prior.sha(video["path"])
    assert prior.video_frames(video["path"], tuple(video["shape"])) == \
        video["frames"]
    audit = {"passed": True, "rows": checked,
             "comparison_video": video,
             "scope": "Independently regenerate all hold/restage low/mid packets from the changed MuJoCo geometry, remeasure 20 ms person/prop gaps and decode complete owner/flank/comparison videos. Sensor mounting and dynamics remain provisional."}
    path = trial.OUT / "audit.json"
    path.write_text(json.dumps(audit, indent=2)+"\n")
    print(path)


if __name__ == "__main__":
    main()
