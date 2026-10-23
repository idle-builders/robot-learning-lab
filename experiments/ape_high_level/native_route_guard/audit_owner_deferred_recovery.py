"""Independent packet, decision, geometry, evidence and media audit for V67."""

from __future__ import annotations

import json
from pathlib import Path

import imageio
import mujoco
import numpy as np

import owner_deferred_recovery as v67


runner, v64, v60, v52 = v67.runner, v67.v64, v67.v60, v67.v52
OUT = v67.OUT


def same_packet(actual, expected):
    assert len(actual) == len(expected) == 72
    for index, (a, b) in enumerate(zip(actual, expected)):
        assert (a is None) == (b is None)
        if a is not None:
            assert abs(a-b) < 2e-5, (index, a, b)


def person_fn(summary, run, cooperative):
    request_s = run["request_time_s"]
    script_return_s = run["scripted_unprompted_return_clock_s"]
    second_s = run["second_request_s"]
    start = runner.human_xy(v64.v54.CASE, summary, second_s, request_s,
                            script_return_s)

    def at(t):
        if t <= second_s:
            return runner.human_xy(v64.v54.CASE, summary, t, request_s,
                                   script_return_s)
        if not cooperative or t <= second_s+runner.HUMAN_DELAY_S:
            return start.copy()
        distance = float(np.linalg.norm(v67.STATION_XY_M-start))
        u = min(1., runner.HUMAN_AWAY_MPS*(t-second_s-
                                           runner.HUMAN_DELAY_S)/distance)
        return start+(v67.STATION_XY_M-start)*u

    return at


def pose_at(rows, t):
    poses = [(row["time_s"], np.asarray(row["ape_xy_m"]))
             for row in rows if row["stage"] in
             ("deferred_range_watch", "second_space_range_watch",
              "recovered_inference_range_watch")]
    if t <= poses[0][0]:
        return poses[0][1]
    for (a, pa), (b, pb) in zip(poses, poses[1:]):
        if t <= b:
            u = np.clip((t-a)/(b-a), 0., 1.)
            return pa+u*(pb-pa)
    return poses[-1][1]


def main():
    report = json.loads((OUT / "report.json").read_text())
    assert report["source_sha256"] == v52.sha(v67.__file__)
    assert report["v64_report_sha256"] == v52.sha(v64.OUT / "report.json")
    assert report["scene_sha256"] == v52.sha(runner.v35.moving.SCENE_XML)
    assert report["current_design_pointer_sha256"] == v52.sha(
        v67.ROOT / "robot/ape-current-design.json")
    source = runner.original()
    summary = source["summary"]
    policy = json.loads(runner.joined.CONFIG.read_text())
    scenario = next(x for x in policy["cases"] if x["id"] ==
                    "late_crossing")
    base = json.loads((v64.OUT / "report.json").read_text())
    base_by_path = {x["path"]: x["run"] for x in base["results"]}
    scanner = v64.v51.v46.plane_packet(runner.observed_packet,
                                       (.05, .30))
    checked = []
    for entry in report["results"]:
        path, cooperative, run = (entry["path"], entry["cooperative"],
                                  entry["run"])
        trace_path = Path(run["trace"]["path"])
        assert v52.sha(trace_path) == run["trace"]["sha256"]
        rows = json.loads(trace_path.read_text())
        base_trace = json.loads(Path(base_by_path[path]["trace"]["path"])
                                .read_text())
        assert rows[:len(base_trace)] == base_trace
        post = rows[len(base_trace):]
        assert post[0]["stage"] == "second_space_request"
        second = [x for x in post if x["stage"] ==
                  "second_space_range_watch"]
        inference = [x for x in post if x["stage"] ==
                     "recovered_inference_range_watch"]
        assert second and (len(inference) >= 36 if cooperative else
                           len(inference) == 0)
        if cooperative:
            assert run["fresh_item_capture_count"] == 1
            assert len([x for x in post if x["stage"] ==
                        "fresh_new_show_capture"]) == 1
            assert run["second_clear_s"] is not None
            assert run["fresh_show_s"] > run["second_clear_s"]
            assert run["recovery_alarm_s"] is None
            assert run["outcome"] in (
                "new_show_metric_item_seen_unattributed",
                "new_show_unconfirmed_unknown")
            evidence = next(x for x in post if x["stage"] ==
                            "fresh_new_show_capture")
            raw = Path(evidence["raw_rgbd"]["path"])
            assert v52.sha(raw) == evidence["raw_rgbd"]["sha256"]
            with np.load(raw) as data:
                assert data["before_rgb"].shape == (480, 640, 3)
                assert data["shown_depth"].shape == (480, 640)
                assert np.any(data["before_rgb"] != data["shown_rgb"])
            item = next(x for x in post if x["stage"] ==
                        "fresh_item_candidate")
            assert item["rgb_change_in_box"] >= .05
            assert item["depth_valid_fraction"] >= .7
            assert item["depth_valid_pixels"] >= 50
            checks = [x for x in post if x["stage"] ==
                      "fresh_new_show_check"]
            assert len(checks) == 2
            assert all(x["accepted"] ==
                       (x["age_s"] <= runner.owner.MAX_FRESH_AGE_S and
                        x["actor_box_xyxy"] is not None and
                        x["point_xyz_est_m"] is not None and
                        x["depth_valid_fraction"] >= .7)
                       for x in checks)
            assert (all(x["accepted"] for x in checks)) == (
                run["outcome"] == "new_show_metric_item_seen_unattributed")
        else:
            assert run["fresh_item_capture_count"] == 0
            assert not inference
            assert run["second_clear_s"] is None
            assert run["outcome"] == \
                "deferred_after_ignored_second_space_request"
        assert all(x["stage"] != "fresh_new_show_capture"
                   for x in base_trace)
        assert all(x.get("old_owner_identity", "unknown") == "unknown"
                   for x in rows if "old_owner_identity" in x)

        person = person_fn(summary, run, cooperative)
        model = mujoco.MjModel.from_xml_path(str(runner.v35.moving.SCENE_XML))
        model.geom_size[model.geom("route_obstacle_geom").id, 1] = \
            v64.v51.v46.PROP_HALF_HEIGHT_M
        model.body_pos[model.body("route_obstacle").id, 2] = \
            v64.v51.v46.PROP_CENTER_Z_M
        data = mujoco.MjData(model)
        runner.geometry_at(summary, source["result"]["result"]["episode"],
                           scenario, model, data)
        baseline = base_trace[-1]["actor_packet"]
        streak = 0
        for stage, group, tick_base in (("second_space_range_watch", second,
                                        2000),
                                        ("recovered_inference_range_watch",
                                         inference, 3000)):
            for index, row in enumerate(group):
                t = row["time_s"]
                xy = np.asarray(row["ape_xy_m"])
                runner.v35.two.set_distractor(model, person(t))
                model.body_pos[model.body("audit_ape").id, :2] = xy
                model.body_pos[model.body("route_obstacle").id, :2] = \
                    v60.v50.prop_at(t, report["checkpoint_s"],
                                    v64.PATHS[path])
                mujoco.mj_forward(model, data)
                expected = scanner(model, data, xy, v64.v54.CASE,
                                   tick_base+index+(1 if stage ==
                                                    "second_space_range_watch"
                                                    else 0),
                                   person_xy=person(t))
                same_packet(row["actor_packet"], expected)
                assert abs(runner.geometry.gap(model, data, xy, person(t))-
                           row["evaluator_person_gap_m"]) < 2e-5
                assert abs(runner.prop_gap(model, data, xy)-
                           row["evaluator_prop_gap_m"]) < 2e-5
                if stage == "second_space_range_watch":
                    receding = sum(a is not None and b is not None and a < .8
                                   and b-a >= .10
                                   for a, b in zip(baseline, expected))
                    bearing = np.degrees(np.arctan2(
                        v67.STATION_XY_M[1]-xy[1],
                        v67.STATION_XY_M[0]-xy[0]))
                    ray = round((bearing+180.)/5.) % 72
                    station_new = sum(
                        baseline[j] is not None and expected[j] is not None
                        and baseline[j]-expected[j] >= 1. and
                        1. <= expected[j] <= 2.4
                        for j in ((ray+d) % 72 for d in range(-2, 3)))
                    assert row["receding_near_rays"] == receding
                    assert row["new_station_rays"] == station_new
                    command = row["near_body_command"]
                    step = command["commanded_step_m"] if command else 0.
                    clear = (runner.side_healthy(expected) and
                             receding >= 2 and station_new >= 1 and
                             row["near_body_observation"]["risk_tracks"] == 0
                             and step == 0.)
                    streak = streak+1 if clear else 0
                    assert row["clear_streak"] == streak
        if cooperative:
            assert second[-1]["clear_streak"] == runner.CLEAR_STREAK
        else:
            assert max(x["clear_streak"] for x in second) == 0

        samples = run["evaluator_gap_samples"]
        for sample in samples:
            if sample["time_s"] <= run["second_request_s"]:
                continue  # exact V64 prefix was independently audited there
            t = sample["time_s"]
            xy = pose_at(rows, t)
            runner.v35.two.set_distractor(model, person(t))
            model.body_pos[model.body("route_obstacle").id, :2] = \
                v60.v50.prop_at(t, report["checkpoint_s"], v64.PATHS[path])
            assert abs(runner.geometry.gap(model, data, xy, person(t))-
                       sample["gap_m"]) < 3e-4
            assert abs(runner.prop_gap(model, data, xy)-
                       sample["prop_gap_m"]) < 3e-4
        assert run["first_below_5cm_s"] is None
        assert min(x["gap_m"] for x in samples) == run["minimum_proxy_gap_m"]
        assert min(x["prop_gap_m"] for x in samples) == \
            run["minimum_prop_gap_m"]
        video = Path(run["video"]["path"])
        assert v52.sha(video) == run["video"]["sha256"]
        reader = imageio.get_reader(video)
        try:
            assert reader.count_frames() == run["video"]["frames"]
            assert reader.get_data(reader.count_frames()-1).shape == \
                (528, 1040, 3)
        finally:
            reader.close()
        checked.append({"variant": f"{path}-{'cooperative' if cooperative else 'ignored'}",
                        "matched_v64_prefix_rows": len(base_trace),
                        "second_space_packets": len(second),
                        "inference_packets": len(inference),
                        "video_frames": run["video"]["frames"]})
    comparison = OUT / "comparisons/second-request-recovery.mp4"
    reader = imageio.get_reader(comparison)
    try:
        comparison_frames = reader.count_frames()
        assert comparison_frames == 43
        assert reader.get_data(28).shape == (264, 1560, 3)
    finally:
        reader.close()
    audit = {"passed": True, "rows": checked,
             "comparison_video_sha256": v52.sha(comparison),
             "comparison_video_frames": comparison_frames,
             "scope": "V64 prefix equality, fresh low/mid packets, second-space gate, 20ms geometry, raw RGB-D, item checks and decoded full/comparison videos"}
    (OUT / "audit.json").write_text(json.dumps(audit, indent=2)+"\n")
    print(OUT / "audit.json")


if __name__ == "__main__":
    main()
