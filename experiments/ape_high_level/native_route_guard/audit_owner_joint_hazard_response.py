"""Independently audit V41 camera evidence, range decisions and swept gaps."""

from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path

import cv2
import mujoco
import numpy as np

import owner_joint_hazard_response as trial
import render_owner_joint_hazard_response as renderer


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def video_frames(path, shape):
    capture = cv2.VideoCapture(str(path))
    assert capture.isOpened(), path
    count = 0
    while True:
        ok, image = capture.read()
        if not ok:
            break
        assert image.shape == shape
        count += 1
    capture.release()
    return count


def image_observation(evidence, index):
    rgb = evidence["rgb"][index]
    depth = evidence["depth"][index]
    old_rgb = evidence["baseline_rgb"][index]
    old_depth = evidence["baseline_depth"][index]
    color = np.max(cv2.absdiff(rgb, old_rgb), axis=2) > \
        trial.v40.v38.v36.RGB_CHANGE
    foreground = (color & np.isfinite(depth) & np.isfinite(old_depth) &
                  (old_depth-depth > trial.v40.v38.v36.DEPTH_CHANGE_M) &
                  (depth >= .15) & (depth <= 2.8))
    ys, xs = np.where(foreground)
    if len(xs) < 250:
        return len(xs), None
    focus = 480. / (2*math.tan(math.radians(float(
        evidence["fovy_deg"][index]))/2))
    z = depth[ys, xs]
    local = np.vstack(((xs-320.)*z/focus,
                       -(ys-240.)*z/focus, -z))
    point = (evidence["camera_rotation"][index] @ local).T + \
        evidence["camera_position"][index]
    return len(xs), np.median(point[:, :2], axis=0)


def packet_sector_x(packet, pose):
    angles = np.linspace(-math.pi, math.pi, len(packet), endpoint=False)
    values = [pose[0]+reading*math.cos(angle)
              for reading, angle in zip(packet, angles)
              if reading is not None and .2 < reading < 1.3 and
              math.radians(-55) <= angle <= math.radians(-10)]
    return float(np.percentile(values, 20)) if len(values) >= 3 else None


def packet_clear(packet, pose, xy, direction):
    angles = np.linspace(-math.pi, math.pi, len(packet), endpoint=False)
    hits = np.asarray([np.asarray(pose)+reading*np.array([
        math.cos(angle), math.sin(angle)])
        for reading, angle in zip(packet, angles) if reading is not None])
    if not len(hits):
        return False
    delta = hits-np.asarray(xy)
    along = np.clip(delta @ direction, 0., .30)
    distance = np.linalg.norm(delta-along[:, None]*direction, axis=1)
    return bool(np.all(distance >= .22))


def audit_result(item, report, summary, source, scenario):
    case = item["prop_case"]
    run = item["run"]
    assert sha(run["trace"]["path"]) == run["trace"]["sha256"]
    assert sha(run["video"]["path"]) == run["video"]["sha256"]
    assert video_frames(run["video"]["path"], (528, 1040, 3)) == run["video"]["frames"]
    rows = json.loads(Path(run["trace"]["path"]).read_text())
    watch = [row for row in rows if row["stage"] == "second_inference_range_watch"]
    plan = next(row for row in rows if row["stage"] == "second_response_plan")
    inference = next(row for row in rows if row["stage"] == "second_inference_result")
    sensor_video = next(row for row in rows if row["stage"] == "flank_sensor_video")
    assert sha(sensor_video["path"]) == sensor_video["sha256"]
    assert video_frames(sensor_video["path"], (480, 640, 3)) == sensor_video["frames"]
    assert sensor_video["frames"] == len(watch)+1
    assert plan["adaptive_controller"] and plan["south_range_clear"]
    assert inference["camera_alarm_s"] is not None
    assert inference["alarm_s"] is None
    assert inference["pose_loss_s"] is None
    assert run["checkpoint_range_packet_exact"]
    model = mujoco.MjModel.from_xml_path(str(trial.v40.v38.v35.moving.SCENE_XML))
    data = mujoco.MjData(model)
    start, _, t0, source_hash, endpoint = trial.v40.v38.geometry_at(
        summary, source["result"]["result"]["episode"], scenario,
        model, data)
    assert t0 == run["checkpoint_time_s"]
    assert source_hash == run["source_v36_report_sha256"]
    assert rows[0]["checkpoint_range_packet"] == endpoint

    raw = item["camera_rgbd_evidence"]
    assert sha(raw["path"]) == raw["sha256"]
    evidence = np.load(raw["path"])
    camera_track = []
    for index, t in enumerate(evidence["time_s"]):
        matches = [row for row in watch if abs(row["time_s"]-t) < .0001]
        assert len(matches) == 1
        sensor = matches[0]["flank_rgbd_packet"]
        assert sensor["actor_cue_usable"]
        count, estimate = image_observation(evidence, index)
        assert count == sensor["foreground_pixels"]
        if estimate is None:
            assert sensor["foreground_xy_est_m"] is None
        else:
            assert np.linalg.norm(estimate-sensor[
                "foreground_xy_est_m"]) < 1e-6
            camera_track.append((t, estimate))
            if len(camera_track) >= 2:
                old_t, old_xy = camera_track[-2]
                if t-old_t > .04:
                    expected = (estimate-old_xy)/(t-old_t)
                    assert np.linalg.norm(expected-sensor[
                        "foreground_velocity_est_mps"]) < 1e-6
    evidence.close()
    assert len(camera_track) >= 2

    camera_streak = 0
    camera_alarm = None
    switched = False
    old_sector = None
    prior = None
    movement = 0.
    decisions = []
    for row in watch:
        sensor = row["flank_rgbd_packet"]
        assert sensor is not None
        cue = (sensor["actor_cue_usable"] and
               sensor["rgb_changed_pixels"] >= trial.v39.RGB_MOTION_PIXELS and
               sensor["depth_changed_pixels"] >= trial.v39.DEPTH_MOTION_PIXELS)
        assert sensor["actor_motion_cue"] == cue
        camera_streak = camera_streak+1 if cue else 0
        assert row["camera_motion_streak"] == camera_streak
        if camera_streak >= 2 and camera_alarm is None:
            camera_alarm = row["time_s"]
        assert len(row["actor_packet"]) == 72
        assert np.linalg.norm(np.asarray(row[
            "evaluator_route_obstacle_xy_m"])-trial.v40.prop_at(
                row["time_s"], t0, trial.PROP_CASES[case])) < .0001
        step = row["response_step_info"]
        if step is not None:
            assert prior is not None
            assert abs(step["range_packet_time_s"]-prior["time_s"]) < .0001
            sector = packet_sector_x(prior["actor_packet"],
                                     prior["ape_xy_m"])
            assert (sector is None and step["east_south_sector_x_m"] is None or
                    sector is not None and
                    abs(sector-step["east_south_sector_x_m"]) < 1e-8)
            delta_x = None if sector is None or old_sector is None else old_sector-sector
            if delta_x is None:
                assert step["east_south_x_drop_m"] is None
            else:
                assert abs(delta_x-step["east_south_x_drop_m"]) < 1e-8
                if delta_x >= trial.CLOSING_X_M:
                    switched = True
            if sector is not None:
                old_sector = sector
            direction = trial.SOUTHWEST if switched else trial.SOUTH
            clear = packet_clear(prior["actor_packet"], prior["ape_xy_m"],
                                 prior["ape_xy_m"], direction)
            assert step["corridor_clear_from_range"] == clear
            expected_dir = direction if clear else np.zeros(2)
            assert np.linalg.norm(expected_dir-step["direction_xy"]) < 1e-8
            expected_xy = np.asarray(prior["ape_xy_m"]) + \
                expected_dir*step["commanded_step_m"]
            assert np.linalg.norm(expected_xy-row["ape_xy_m"]) < .0001
            movement += step["commanded_step_m"]
            decisions.append(step["action"])
        prior = row
    assert camera_alarm is not None
    assert abs(camera_alarm-inference["camera_alarm_s"]) < .0001
    assert switched and "swerve_southwest" in decisions
    assert movement <= trial.v40.v38.SECOND_RESPONSE_MAX_M+.0001

    # Score from independently interpolated packet poses, not the runner's
    # actor_ape_at helper. Person/prop scripted geometry is evaluator-only.
    times = np.asarray([row["time_s"] for row in watch])
    positions = np.asarray([row["ape_xy_m"] for row in watch])
    person_gaps, prop_gaps = [], []
    for row in run["evaluator_gap_samples"]:
        t = row["time_s"]
        ape = np.array([np.interp(t, times, positions[:, coordinate],
                                  left=start[coordinate],
                                  right=positions[-1, coordinate])
                        for coordinate in (0, 1)])
        assert np.linalg.norm(ape-row["ape_xy_m"]) < .00011
        person = trial.v40.v38.human_xy(
            "blind_return_flank_fast_response", summary, t,
            run["request_time_s"], run["second_inference_start_s"])
        model.body_pos[model.body("route_obstacle").id, :2] = \
            trial.v40.prop_at(t, t0, trial.PROP_CASES[case])
        person_gap = trial.v40.v38.geometry.gap(model, data, ape, person)
        prop_gap = trial.v40.v38.prop_gap(model, data, ape)
        assert abs(person_gap-row["gap_m"]) < .00011
        assert abs(prop_gap-row["prop_gap_m"]) < .00011
        person_gaps.append(row["gap_m"])
        prop_gaps.append(row["prop_gap_m"])
    assert run["minimum_proxy_gap_m"] == min(person_gaps)
    assert run["minimum_prop_gap_m"] == min(prop_gaps)
    assert run["outcome"] == "new_show_flank_motion_unresolved"
    return {"prop_case": case, "camera_alarm_s": camera_alarm,
            "first_response_s": inference["first_response_s"],
            "person_gap_m": min(person_gaps), "prop_gap_m": min(prop_gaps),
            "switched": switched, "decision_count": len(decisions),
            "camera_evidence_frames": len(camera_track),
            "video_frames": run["video"]["frames"]}


def main():
    report = json.loads((trial.OUT / "report.json").read_text())
    assert report["schema"] == "robot-learning-lab.ape-owner-joint-hazard-response.v1"
    assert report["source_sha256"] == sha(trial.__file__)
    assert report["v38_source_sha256"] == sha(trial.v40.v38.__file__)
    assert report["v39_source_sha256"] == sha(trial.v39.__file__)
    assert report["v40_source_sha256"] == sha(trial.v40.__file__)
    assert report["scene_sha256"] == sha(trial.v40.v38.v35.moving.SCENE_XML)
    assert report["v36_report_sha256"] == sha(trial.v40.v38.SOURCE)
    assert report["current_design_pointer_sha256"] == sha(
        trial.ROOT / "robot/ape-current-design.json")
    assert report["prop_cases"] == trial.PROP_CASES
    source = trial.v40.v38.original()
    summary = source["summary"]
    policy = json.loads(trial.v40.v38.joined.CONFIG.read_text())
    scenario = next(x for x in policy["cases"] if x["id"] == "late_crossing")
    results = [audit_result(item, report, summary, source, scenario)
               for item in report["results"]]
    assert [row["prop_case"] for row in results] == list(report["prop_cases"])
    comparison = json.loads((trial.OUT / "comparison.json").read_text())
    assert comparison["schema"] == \
        "robot-learning-lab.ape-owner-joint-hazard-comparisons.v1"
    assert comparison["report_sha256"] == sha(trial.OUT / "report.json")
    assert comparison["source_sha256"] == sha(renderer.__file__)
    assert set(comparison["overhead_videos"]) == set(report["prop_cases"])
    lookup = {item["prop_case"]: item for item in report["results"]}
    for case, video in comparison["overhead_videos"].items():
        assert video["kind"] == "evaluator_replay_from_20ms_gap_samples"
        assert video["source_trace_sha256"] == lookup[case]["run"]["trace"]["sha256"]
        assert sha(video["path"]) == video["sha256"]
        assert video_frames(video["path"], (680, 800, 3)) == video["frames"]
    audit = {"passed": True, "rows": results,
             "overhead_videos": comparison["overhead_videos"],
             "scope": "V41 independently reprojects retained RGB-D foreground to world coordinates, checks packet-causal range closure and corridor choice, interpolates commanded adaptive poses, remeasures MuJoCo person/prop gaps at 20 ms, and decodes full videos. Historical kinematic proxy and scripted hazards remain unqualified."}
    (trial.OUT / "audit.json").write_text(json.dumps(audit, indent=2)+"\n")
    print(json.dumps(audit, indent=2))


if __name__ == "__main__":
    main()
