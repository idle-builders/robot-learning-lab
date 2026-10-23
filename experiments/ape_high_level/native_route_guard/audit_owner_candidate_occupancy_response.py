"""Audit V42 RGB-D lower track, candidate choice and executed geometry."""

from __future__ import annotations

import json
import math
from pathlib import Path

import cv2
import mujoco
import numpy as np

import audit_owner_joint_hazard_response as prior_audit
import owner_candidate_occupancy_response as trial
import render_owner_candidate_occupancy_response as renderer


def lower_observation(raw, index):
    rgb, depth = raw["rgb"][index], raw["depth"][index]
    old_rgb = raw["baseline_rgb"][index]
    old_depth = raw["baseline_depth"][index]
    mask = ((np.max(cv2.absdiff(rgb, old_rgb), axis=2) >
             trial.v41.v40.v38.v36.RGB_CHANGE) &
            np.isfinite(depth) & np.isfinite(old_depth) &
            (old_depth-depth > trial.v41.v40.v38.v36.DEPTH_CHANGE_M) &
            (depth >= .15) & (depth <= 2.8))
    rows, columns = np.where(mask)
    if not len(columns):
        return 0, None
    focus = 480. / (2*math.tan(math.radians(float(raw["fovy_deg"][index]))/2))
    z = depth[rows, columns]
    camera_points = np.vstack(((columns-320.)*z/focus,
                               -(rows-240.)*z/focus, -z))
    world = (raw["camera_rotation"][index] @ camera_points).T + \
        raw["camera_position"][index]
    lower = world[(world[:, 2] >= 0.) & (world[:, 2] <= .30)]
    return len(lower), (np.median(lower[:, :2], axis=0)
                        if len(lower) >= 250 else None)


def low_surface(packet, pose):
    angles = np.linspace(-math.pi, math.pi, len(packet), endpoint=False)
    points = np.asarray([
        np.asarray(pose)+reading*np.array([math.cos(angle), math.sin(angle)])
        for reading, angle in zip(packet, angles)
        if reading is not None and .2 < reading < 1.3 and
        math.radians(-55) <= angle <= math.radians(-10)])
    if len(points) < 3:
        return None
    return np.array([np.percentile(points[:, 0], 20),
                     np.median(points[:, 1])])


def check_result(item, source, scenario, checkpoint_s):
    case, run = item["case"], item["run"]
    assert prior_audit.sha(run["trace"]["path"]) == run["trace"]["sha256"]
    assert prior_audit.sha(run["video"]["path"]) == run["video"]["sha256"]
    assert prior_audit.video_frames(run["video"]["path"], (528, 1040, 3)) == \
        run["video"]["frames"]
    rows = json.loads(Path(run["trace"]["path"]).read_text())
    watch = [row for row in rows if row["stage"] == "second_inference_range_watch"]
    inference = next(row for row in rows if row["stage"] ==
                     "second_inference_result")
    sensor_video = next(row for row in rows if row["stage"] ==
                        "flank_sensor_video")
    assert prior_audit.sha(sensor_video["path"]) == sensor_video["sha256"]
    assert prior_audit.video_frames(sensor_video["path"], (480, 640, 3)) == \
        sensor_video["frames"]
    assert sensor_video["frames"] == len(watch)+1
    assert run["checkpoint_range_packet_exact"]
    model = mujoco.MjModel.from_xml_path(str(
        trial.v41.v40.v38.v35.moving.SCENE_XML))
    data = mujoco.MjData(model)
    summary = source["summary"]
    start, _, t0, source_hash, endpoint = trial.v41.v40.v38.geometry_at(
        summary, source["result"]["result"]["episode"], scenario,
        model, data)
    assert t0 == checkpoint_s == run["checkpoint_time_s"]
    assert run["source_v36_report_sha256"] == source_hash
    assert rows[0]["checkpoint_range_packet"] == endpoint

    evidence = item["camera_rgbd_evidence"]
    assert prior_audit.sha(evidence["path"]) == evidence["sha256"]
    raw = np.load(evidence["path"])
    track = []
    for index, time_s in enumerate(raw["time_s"]):
        matches = [row for row in watch if abs(row["time_s"]-time_s) < .0001]
        assert len(matches) == 1
        sensor = matches[0]["flank_rgbd_packet"]
        assert sensor["actor_cue_usable"]
        count, point = lower_observation(raw, index)
        assert count == sensor["lower_foreground_pixels"]
        if point is None:
            assert sensor["lower_surface_xy_est_m"] is None
        else:
            assert np.linalg.norm(point-sensor["lower_surface_xy_est_m"]) < 1e-6
            track.append((float(time_s), point))
            if len(track) >= 2 and track[-1][0]-track[-2][0] > .04:
                velocity = (track[-1][1]-track[-2][1]) / \
                    (track[-1][0]-track[-2][0])
                assert np.linalg.norm(velocity-sensor[
                    "lower_surface_velocity_est_mps"]) < 1e-6
    raw.close()

    plans = [row for row in rows if row["stage"] == "second_response_plan"]
    first_choice = next((row["response_step_info"]["choice"] for row in watch
                         if row["response_step_info"] is not None and
                         row["response_step_info"]["choice"] is not None), None)
    if case == "parked_after_clear":
        assert not plans and first_choice is None
        assert inference["camera_alarm_s"] is None
        assert run["outcome"] == "new_show_metric_item_seen_unattributed"
    else:
        assert len(plans) == 1 and plans[0]["adaptive_controller"]
        assert first_choice == item["controller_choice"]
        assert inference["camera_alarm_s"] is not None
        assert run["outcome"] == "new_show_flank_motion_unresolved"
        chosen_row = next((index for index, row in enumerate(watch)
                           if row["response_step_info"] is not None and
                           row["response_step_info"]["choice"] is not None))
        assert chosen_row > 0
        prior = watch[chosen_row-1]
        choice = first_choice
        observed_low = low_surface(prior["actor_packet"], prior["ape_xy_m"])
        assert observed_low is not None
        assert np.linalg.norm(observed_low-choice["low_surface_xy_m"]) < 1e-8
        candidates = [row for row in watch[:chosen_row]
                      if row["flank_rgbd_packet"] and
                      row["flank_rgbd_packet"]["actor_cue_usable"] and
                      row["flank_rgbd_packet"]["lower_surface_velocity_est_mps"]
                      is not None]
        assert candidates
        camera = candidates[-1]["flank_rgbd_packet"]
        assert np.linalg.norm(np.asarray(choice["person_surface_xy_m"])-
                              camera["lower_surface_xy_est_m"]) < 1e-8
        assert np.linalg.norm(np.asarray(choice["person_velocity_mps"])-
                              camera["lower_surface_velocity_est_mps"]) < 1e-8
        chosen_time = watch[chosen_row]["time_s"]
        for name, direction in trial.CANDIDATES.items():
            person_min = low_min = float("inf")
            for future in trial.FUTURE_S:
                ape = np.asarray(prior["ape_xy_m"]) + \
                    direction*min(.25, .25*future)
                person = np.asarray(choice["person_surface_xy_m"]) + \
                    np.asarray(choice["person_velocity_mps"])*(
                        chosen_time-camera["time_s"]+future)
                person_min = min(person_min, np.linalg.norm(ape-person)-
                                 .17-.08-.05-.02*future)
                for speed in trial.PROP_SPEED_HYPOTHESES_MPS:
                    low = observed_low+np.array([-speed*future, 0.])
                    low_min = min(low_min, np.linalg.norm(ape-low)-
                                  .17-.08-.05-.03*future)
            recorded = choice["scores"][name]
            assert abs(recorded["person_min_m"]-person_min) < .0001
            assert abs(recorded["low_min_m"]-low_min) < .0001
            assert recorded["eligible_person"] == (person_min >= 0.)
        eligible = [name for name in trial.CANDIDATES if name != "hold" and
                    choice["scores"][name]["eligible_person"]]
        selected = (max(eligible, key=lambda name: (
            choice["scores"][name]["low_min_m"],
            choice["scores"][name]["person_min_m"])) if eligible else "hold")
        assert choice["chosen"] == selected
        for previous, row in zip(watch, watch[1:]):
            decision = row["response_step_info"]
            if decision is None:
                continue
            if decision["choice"] is None:
                assert decision["commanded_step_m"] == 0.
                continue
            direction = trial.CANDIDATES[selected]
            clear = (prior_audit.packet_clear(
                previous["actor_packet"], previous["ape_xy_m"],
                previous["ape_xy_m"], direction)
                     if selected != "hold" else False)
            assert decision["corridor_clear_from_range"] == clear
            applied = direction if clear else np.zeros(2)
            assert np.linalg.norm(applied-decision["direction_xy"]) < 1e-8
            expected = np.asarray(previous["ape_xy_m"]) + \
                applied*decision["commanded_step_m"]
            assert np.linalg.norm(expected-row["ape_xy_m"]) < .0001

    times = np.asarray([row["time_s"] for row in watch])
    poses = np.asarray([row["ape_xy_m"] for row in watch])
    person_gaps, prop_gaps = [], []
    for sample in run["evaluator_gap_samples"]:
        t = sample["time_s"]
        ape = np.array([np.interp(t, times, poses[:, axis],
                                  left=start[axis], right=poses[-1, axis])
                        for axis in (0, 1)])
        assert np.linalg.norm(ape-sample["ape_xy_m"]) < .00011
        person = trial.v41.v40.v38.human_xy(
            item["condition"], summary, t, run["request_time_s"],
            run["second_inference_start_s"])
        model.body_pos[model.body("route_obstacle").id, :2] = \
            trial.v41.v40.prop_at(t, checkpoint_s, item["prop_config"])
        gap = trial.v41.v40.v38.geometry.gap(model, data, ape, person)
        prop = trial.v41.v40.v38.prop_gap(model, data, ape)
        assert abs(gap-sample["gap_m"]) < .00011
        assert abs(prop-sample["prop_gap_m"]) < .00011
        person_gaps.append(sample["gap_m"])
        prop_gaps.append(sample["prop_gap_m"])
    assert run["minimum_proxy_gap_m"] == min(person_gaps)
    assert run["minimum_prop_gap_m"] == min(prop_gaps)
    return {"case": case, "choice": first_choice["chosen"] if first_choice else None,
            "camera_alarm_s": inference["camera_alarm_s"],
            "first_response_s": inference["first_response_s"],
            "minimum_person_gap_m": min(person_gaps),
            "minimum_prop_gap_m": min(prop_gaps),
            "raw_frames": len(track), "video_frames": run["video"]["frames"]}


def main():
    report = json.loads((trial.OUT / "report.json").read_text())
    assert report["schema"] == "robot-learning-lab.ape-owner-candidate-occupancy.v1"
    assert report["source_sha256"] == prior_audit.sha(trial.__file__)
    assert report["v41_source_sha256"] == prior_audit.sha(trial.v41.__file__)
    assert report["v38_source_sha256"] == prior_audit.sha(
        trial.v41.v40.v38.__file__)
    assert report["scene_sha256"] == prior_audit.sha(
        trial.v41.v40.v38.v35.moving.SCENE_XML)
    assert report["v36_report_sha256"] == prior_audit.sha(
        trial.v41.v40.v38.SOURCE)
    assert report["current_design_pointer_sha256"] == prior_audit.sha(
        trial.ROOT / "robot/ape-current-design.json")
    assert report["candidate_directions"] == {name: value.tolist()
                                               for name, value in
                                               trial.CANDIDATES.items()}
    assert report["cases"] == [item["case"] for item in report["results"]]
    source = trial.v41.v40.v38.original()
    policy = json.loads(trial.v41.v40.v38.joined.CONFIG.read_text())
    scenario = next(x for x in policy["cases"] if x["id"] == "late_crossing")
    results = [check_result(item, source, scenario, report["checkpoint_s"])
               for item in report["results"]]
    comparison = json.loads((trial.OUT / "comparison.json").read_text())
    assert comparison["schema"] == \
        "robot-learning-lab.ape-owner-candidate-occupancy-videos.v1"
    assert comparison["report_sha256"] == prior_audit.sha(
        trial.OUT / "report.json")
    assert comparison["source_sha256"] == prior_audit.sha(renderer.__file__)
    assert comparison["base_renderer_sha256"] == prior_audit.sha(
        renderer.view.__file__)
    assert set(comparison["overhead_videos"]) == set(trial.v41.PROP_CASES)
    for video in comparison["overhead_videos"].values():
        assert prior_audit.sha(video["path"]) == video["sha256"]
        assert prior_audit.video_frames(video["path"], (680, 800, 3)) == \
            video["frames"]
    audit = {"passed": True, "rows": results,
             "overhead_videos": comparison["overhead_videos"],
             "scope": "V42 independently reconstructs lower RGB-D surface observations, checks candidate future scores and packet-causal choices, remeasures 20ms person/prop gaps, and decodes full videos. Forecast speed and buffer assumptions remain development hypotheses, not calibrated physical guarantees."}
    (trial.OUT / "audit.json").write_text(json.dumps(audit, indent=2)+"\n")
    print(json.dumps(audit, indent=2))


if __name__ == "__main__":
    main()
