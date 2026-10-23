"""Audit V44's range-selected pose, owner evidence and swept geometry."""

from __future__ import annotations

import json
import math
from pathlib import Path

import mujoco
import numpy as np

import audit_owner_joint_hazard_response as prior
import owner_scanned_meeting_pose as trial
import render_owner_scanned_meeting_pose as renderer


def independent_candidates(packet, xy):
    angles = np.linspace(-math.pi, math.pi, len(packet), endpoint=False)
    hits = [(reading, angle) for reading, angle in zip(packet, angles)
            if reading is not None]
    points = np.asarray([np.asarray(xy)+reading*np.array([
        math.cos(angle), math.sin(angle)]) for reading, angle in hits])
    east_south = np.asarray([math.radians(-55) <= angle <= math.radians(-10)
                             for _, angle in hits])
    results = {}
    for name, direction in trial.CANDIDATES.items():
        target = np.asarray(xy)+direction*trial.TARGET_M
        vectors = points-target
        distances = np.linalg.norm(vectors, axis=1)
        bearing = np.arctan2(vectors[:, 1], vectors[:, 0])
        side = np.abs((bearing+math.pi)%(2*math.pi)-math.pi) >= \
            math.radians(120)
        side_min = float(np.min(distances[side])) if side.any() else float("inf")
        sweep = (min(float(np.min(np.linalg.norm(
            points[east_south]-np.array([shift, 0.])-target,
            axis=1))) for shift in trial.EAST_SOUTH_SWEEP_M)
            if east_south.any() else float("inf"))
        robust = min(float(np.min(distances)),
                     sweep-trial.SWEEP_UNCERTAINTY_M)
        results[name] = {"minimum_endpoint_distance_m": float(np.min(distances)),
                         "predicted_side_endpoint_distance_m": side_min,
                         "east_south_westward_sweep_min_m": sweep,
                         "robust_score_m": robust,
                         "eligible": bool(side_min >= trial.SIDE_CLEAR_M and
                                          robust >= trial.MIN_ENDPOINT_GAP_M)}
    return results


def check(item, source, scenario, checkpoint_s):
    run = item["run"]
    assert prior.sha(run["trace"]["path"]) == run["trace"]["sha256"]
    assert prior.sha(run["video"]["path"]) == run["video"]["sha256"]
    assert prior.video_frames(run["video"]["path"], (528, 1040, 3)) == \
        run["video"]["frames"]
    rows = json.loads(Path(run["trace"]["path"]).read_text())
    steps = [row for row in rows if row["stage"] == "pre_show_restage_step"]
    assert steps and len(steps) == item["restage_steps"]
    model = mujoco.MjModel.from_xml_path(str(trial.v38.v35.moving.SCENE_XML))
    data = mujoco.MjData(model)
    summary = source["summary"]
    start, _, t0, source_hash, endpoint = trial.v38.geometry_at(
        summary, source["result"]["result"]["episode"], scenario,
        model, data)
    assert t0 == checkpoint_s == run["checkpoint_time_s"]
    assert source_hash == run["source_v36_report_sha256"]
    assert rows[0]["checkpoint_range_packet"] == endpoint
    resume = next(row for row in rows if row["stage"] ==
                  "post_alarm_range_watch" and row["actor_action"] ==
                  "resume_after_three_fresh_clear_packets")
    scores = independent_candidates(resume["actor_packet"], start)
    chosen = max((name for name, score in scores.items() if score["eligible"]),
                 key=lambda name: scores[name]["robust_score_m"])
    assert chosen == item["choice"]
    assert steps[0]["actor_decision"]["candidate_scores"] == scores
    previous = resume
    previous_pose = np.asarray(start)
    for row in steps:
        assert abs(row["time_s"]-previous["time_s"]-
                   trial.v38.HOLD_PERIOD_S) < 1e-5
        step = row["commanded_step_m"]
        assert 0. < step <= trial.SPEED_MPS*trial.v38.HOLD_PERIOD_S+1e-9
        pose = np.asarray(row["ape_xy_m"])
        assert np.linalg.norm(pose-(previous_pose+
                                  trial.CANDIDATES[chosen]*step)) < 1e-8
        endpoints = trial.v41.JointHazardResponse.endpoints(
            previous["actor_packet"], previous_pose)
        nearest = float(np.min(np.linalg.norm(endpoints-pose, axis=1)))
        assert abs(nearest-row["actor_decision"][
            "minimum_endpoint_gap_m"]) < 1e-8
        assert nearest >= trial.MIN_ENDPOINT_GAP_M
        previous, previous_pose = row, pose
    assert abs(np.linalg.norm(previous_pose-start)-trial.TARGET_M) < 1e-8
    assert abs(run["second_inference_start_s"]-
               (steps[-1]["time_s"]+trial.v38.SHOW_SETTLE_S)) < 1e-5
    watch = [row for row in rows if row["stage"] ==
             "second_inference_range_watch"]
    assert watch
    timeline = [(t0, np.asarray(start)),
                (resume["time_s"], np.asarray(start))]
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
        person = trial.v38.human_xy(
            item["case"], summary, t, run["request_time_s"],
            run["second_inference_start_s"])
        model.body_pos[model.body("route_obstacle").id, :2] = \
            trial.v41.v40.prop_at(t, checkpoint_s,
                                  trial.v41.PROP_CASES[item["prop_case"]])
        person_gap = trial.v38.geometry.gap(model, data, ape, person)
        prop_gap = trial.v38.prop_gap(model, data, ape)
        assert abs(person_gap-sample["gap_m"]) < .00011
        assert abs(prop_gap-sample["prop_gap_m"]) < .00011
        person_min = min(person_min, person_gap)
        prop_min = min(prop_min, prop_gap)
    assert abs(person_min-run["minimum_proxy_gap_m"]) < .00011
    assert abs(prop_min-run["minimum_prop_gap_m"]) < .00011
    sensor = next(row for row in rows if row["stage"] ==
                  "flank_sensor_video")
    assert prior.sha(sensor["path"]) == sensor["sha256"]
    assert prior.video_frames(sensor["path"], (480, 640, 3)) == \
        sensor["frames"]
    if item["case"] == "parked_after_clear":
        candidate = next(row for row in rows if row["stage"] ==
                         "fresh_item_candidate")
        assert candidate["depth_valid_pixels"] >= 50
        assert run["outcome"] == "new_show_metric_item_seen_unattributed"
    return {"prop_case": item["prop_case"], "case": item["case"],
            "choice": chosen, "outcome": run["outcome"],
            "minimum_person_gap_m": round(person_min, 5),
            "minimum_prop_gap_m": round(prop_min, 5),
            "above_provisional_5cm_both": person_min >= .05 and prop_min >= .05,
            "last_side_range_m": watch[-1]["nearest_side_range_m"],
            "full_video_frames": run["video"]["frames"]}


def main():
    report = json.loads((trial.OUT / "report.json").read_text())
    assert report["schema"] == \
        "robot-learning-lab.ape-owner-scanned-meeting-pose.v1"
    assert report["source_sha256"] == prior.sha(trial.__file__)
    assert report["v38_source_sha256"] == prior.sha(trial.v38.__file__)
    assert report["v41_source_sha256"] == prior.sha(trial.v41.__file__)
    assert report["v43_report_sha256"] == prior.sha(trial.v43.OUT / "report.json")
    assert report["scene_sha256"] == prior.sha(trial.v38.v35.moving.SCENE_XML)
    assert report["v36_report_sha256"] == prior.sha(trial.v38.SOURCE)
    assert report["current_design_pointer_sha256"] == prior.sha(
        trial.ROOT / "robot/ape-current-design.json")
    assert report["owner_fovy_deg"] == trial.OWNER_FOVY_DEG
    source = trial.v38.original()
    policy = json.loads(trial.v38.joined.CONFIG.read_text())
    scenario = next(x for x in policy["cases"] if x["id"] == "late_crossing")
    checked = [check(item, source, scenario, report["checkpoint_s"])
               for item in report["results"]]
    comparison = json.loads((trial.OUT / "comparison.json").read_text())
    assert comparison["schema"] == \
        "robot-learning-lab.ape-owner-scanned-pose-comparisons.v1"
    assert comparison["v43_report_sha256"] == prior.sha(
        trial.v43.OUT / "report.json")
    assert comparison["v44_report_sha256"] == prior.sha(
        trial.OUT / "report.json")
    assert comparison["source_sha256"] == prior.sha(renderer.__file__)
    for video in comparison["videos"].values():
        assert video["sha256"] == prior.sha(video["path"])
        assert prior.video_frames(video["path"], tuple(video["shape"])) == \
            video["frames"]
    audit = {"passed": True, "rows": checked,
             "comparison_videos": comparison["videos"],
             "scope": "Recompute candidate range scores, replay each prior-packet pose command, remeasure 20 ms person/prop gaps and decode scene, sensor and overhead comparison videos. The sweep and 5 cm thresholds are exploratory, not safety guarantees."}
    path = trial.OUT / "audit.json"
    path.write_text(json.dumps(audit, indent=2)+"\n")
    print(path)


if __name__ == "__main__":
    main()
