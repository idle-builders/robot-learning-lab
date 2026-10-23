"""Replay the RGB-D packet provenance, local veto and owner geometry."""

from __future__ import annotations

import json
from pathlib import Path

import cv2
import numpy as np
from scipy.optimize import least_squares

import audit_robocasa_dynamic_suite_v440 as route_audit
from audit_robocasa_layout4_wait_and_remap_v462 import evaluator_prop_at
from audit_lateral_clutter_depth_v285 import project
from audit_low_prop_sensor_join_v207p import (
    low_evaluator_scene, owner_boundary, packets,
)
from audit_low_prop_source_track_v209p import low_packets
from audit_low_prop_coverage_recover_v217p import recovery_replay
from audit_low_prop_fast_crossing_v221p import fault_schedule
from audit_low_prop_track_validity_v223p import validity_replay
from audit_low_prop_capture_age_v228p import delivery_replay, age_gate_replay
from audit_low_prop_aligned_capture_v233p import actor_capture_replay
from audit_low_prop_capture_pose_bias_v235p import projection_replay
from run import ROOT, sha


SOURCE = Path(__file__)
OUT = ROOT / "artifacts/ape-robocasa-low-prop-depth-crosscheck-v236p"
REPORT = OUT / "report.json"


def fitted_depth(packet, estimated_xy, low_center, radius):
    depth = packet["measured_depth_m"]
    origin = np.r_[estimated_xy, .20]
    points = project(depth, (origin, packet["rotation"]), 70.)
    radial = np.linalg.norm(points[:, :2] - origin[:2], axis=1)
    low_center = np.asarray(low_center)
    near = (np.linalg.norm(points[:, :2] - low_center, axis=1) < .45) & (
        points[:, 2] > .04) & (points[:, 2] < .18) & (
        radial < np.linalg.norm(low_center - origin[:2]) - radius + .04)
    selected = points[near, :2]
    if len(selected) < 12:
        return {"status": "too_few_depth_points", "points": len(selected)}
    fit = least_squares(
        lambda center: np.linalg.norm(selected - center, axis=1) - radius,
        low_center, bounds=(low_center - .08, low_center + .08),
        loss="soft_l1", f_scale=.02)
    median = float(np.median(np.abs(np.linalg.norm(selected - fit.x,
                                                 axis=1) - radius)))
    if not fit.success or median > .012:
        return {"status": "ambiguous_depth_shape", "points": len(selected),
                "median_residual_m": median}
    return {"status": "depth_confirmed_circle", "points": len(selected),
            "median_residual_m": median, "center_xy_m": fit.x.tolist()}


def depth_replay(case):
    rows = case["low_depth_decisions"]
    captures = case["low_depth_captures"]
    route = case["route_episode"]["trace"]
    tracks = case["track_history"]
    checks = []
    for index, (row, capture, route_row, track) in enumerate(
            zip(rows, captures, route, tracks)):
        candidate_source = index - case["depth_delivery_delay_steps"]
        expected_source = (candidate_source if candidate_source >= 0 and
                           captures[candidate_source]["packet"] is not None
                           else None)
        expected_age = (case["depth_delivery_delay_steps"] * .4
                        if expected_source is not None else None)
        expected_health = (
            "missing" if expected_source is None else
            "stale" if case["depth_age_limit_s"] is not None and
            expected_age > case["depth_age_limit_s"] else "fresh")
        checked = {"index": index,
                   "alignment": (row["scan_index"] == index and
                                 capture["decision_index"] == index and
                                 abs(capture["time_s"] - index * .4) < 1e-5 and
                                 row["low_track_status"] == track["status"] and
                                 row["depth_source_capture_index"] ==
                                 expected_source and
                                 row["depth_age_s"] == expected_age and
                                 row["depth_age_status"] == expected_health),
                   "camera_packet": True, "projection": True,
                   "veto_rule": True}
        if capture["packet"] is not None:
            path = ROOT / capture["packet"]["path"]
            checked["camera_packet"] = path.is_file() and (
                capture["packet"]["sha256"] == sha(path))
            if checked["camera_packet"]:
                with np.load(path, allow_pickle=False) as packet:
                    previous = tracks[index - 1] if index else None
                    checked["camera_packet"] = (
                        packet["rgb"].shape == (240, 320, 3) and
                        packet["measured_depth_m"].shape == (240, 320) and
                        int(packet["decision_index"]) == index and
                        abs(float(packet["capture_time_s"]) - index * .4) < 1e-5 and
                        index > 0 and previous["center_xy_m"] is not None and
                        np.allclose(capture["actor_capture_pose_xy_m"],
                                    route_row["estimated_xy_m"], atol=1e-7) and
                        np.allclose(packet["previous_actor_estimated_xy_m"],
                                    route[index - 1]["estimated_xy_m"], atol=1e-7) and
                        np.allclose(packet["previous_actor_track_center_xy_m"],
                                    previous["center_xy_m"], atol=1e-7) and
                        np.allclose(packet["rotation"].T @ packet["rotation"],
                                    np.eye(3), atol=1e-6))
        if row["depth_fit"] is not None:
            if expected_source is None:
                checked["projection"] = False
            else:
                source_capture = captures[expected_source]
                with np.load(ROOT / source_capture["packet"]["path"],
                             allow_pickle=False) as packet:
                    fit = fitted_depth(
                        packet, source_capture["actor_capture_pose_xy_m"],
                        track["center_xy_m"], float(track["radius_m"]))
                saved = row["depth_fit"]
                checked["projection"] = (
                    row["depth_age_status"] == "fresh" and
                    fit["status"] == saved["status"] and
                    fit["points"] == saved["points"] and
                    ("median_residual_m" not in saved or abs(
                        fit["median_residual_m"] -
                        saved["median_residual_m"]) < 1e-6) and
                    ("center_xy_m" not in saved or np.allclose(
                        fit["center_xy_m"], saved["center_xy_m"],
                        atol=1e-6)))
        if row["low_minimum_m"] is not None:
            cells = np.asarray(row["lookahead_cells_xy_m"])
            low_center = np.asarray(track["center_xy_m"])
            radius = float(track["radius_m"])
            low_min = float(np.min(np.linalg.norm(cells - low_center,
                                                  axis=1)))
            required = .17 + radius + .05 + .115
            checked["veto_rule"] = (
                abs(low_min - row["low_minimum_m"]) < 1e-7 and
                abs(required - row["required_m"]) < 1e-7 and
                all(np.linalg.norm(cell - route_row["estimated_xy_m"]) <=
                    .6 + 1e-7 for cell in cells))
            if low_min < required:
                fit = row["depth_fit"]
                if fit is not None and fit["status"] == "depth_confirmed_circle":
                    depth_min = float(np.min(np.linalg.norm(
                        cells - np.asarray(fit["center_xy_m"]), axis=1)))
                    expected = ("depth_clears_low_veto" if
                                depth_min >= required + .01 else
                                "ask_owner_clear_corridor")
                    checked["veto_rule"] &= (
                        abs(depth_min - row["depth_minimum_m"]) < 1e-7)
                else:
                    expected = "ask_owner_clear_corridor"
                checked["veto_rule"] &= row["decision"] == expected
        checks.append(checked)
    video = case["low_depth_timeline_video"]
    camera = cv2.VideoCapture(video["path"])
    frame_count = int(camera.get(cv2.CAP_PROP_FRAME_COUNT))
    camera.release()
    return {"count": len(rows),
            "count_match": len(rows) == len(captures) == len(route) == len(tracks),
            "all_checks_pass": len(rows) == len(captures) == len(route) ==
                               len(tracks) and all(all(row[field] for field in
                               ("alignment", "camera_packet", "projection",
                                "veto_rule")) for row in checks),
            "first_failure": next((row for row in checks if not all(
                row[field] for field in ("alignment", "camera_packet",
                                          "projection", "veto_rule"))), None),
            "video_sha_pass": video["sha256"] == sha(Path(video["path"])),
            "video_frame_count_pass": frame_count == video["frames"] ==
                                      2 * len(rows),
            "depth_releases": sum(row["decision"] == "depth_clears_low_veto"
                                  for row in rows)}


def main():
    report = json.loads(REPORT.read_text())
    old_scene, old_path = route_audit.evaluator_scene, route_audit.prop_at
    route_audit.evaluator_scene = low_evaluator_scene
    route_audit.prop_at = evaluator_prop_at
    rows = []
    try:
        for case in report["episodes"]:
            geometry = route_audit.audit_case({
                "scenario": case["scenario"], "package": "depth_crosscheck",
                "episode": case["route_episode"]})
            rows.append({
                "scenario": case["scenario"]["id"],
                "route_state": case["route_state"],
                "outcome": case["outcome"],
                "geometry": geometry,
                "mid_packets": packets(case, 360),
                "low_packets": low_packets(case),
                "fault_schedule": fault_schedule(case, (999., 1000.)),
                "recovery": recovery_replay(case),
                "validity": validity_replay(case, {
                    "quiet_speed_m_s": .05, "path_clear_gap_m": .08,
                    "speed_bound_m_s": .45, "reach_horizon_s": 1.2,
                    "fresh_stable_packets": 3, "max_reobserve_s": 5.2,
                }),
                "delivery": delivery_replay(case, (10.4, 12.4)),
                "actor_capture": actor_capture_replay(case),
                "low_projection": projection_replay(case),
                "age_gate": age_gate_replay(case, .2, 3),
                "depth": depth_replay(case),
                "owner": owner_boundary(case),
            })
    finally:
        route_audit.evaluator_scene = old_scene
        route_audit.prop_at = old_path
    output = {
        "schema": "robot-learning-lab.ape-low-prop-depth-crosscheck-audit.v1",
        "source_sha256": sha(SOURCE),
        "report_sha256": sha(REPORT),
        "experiment_source_match": report["source_sha256"] == sha(
            SOURCE.with_name("run_low_prop_depth_crosscheck_v236p.py")),
        "v232_source_match": report["v232_source_sha256"] == sha(
            SOURCE.with_name("run_low_prop_aligned_capture_v232p.py")),
        "v230_source_match": report["v230_source_sha256"] == sha(
            SOURCE.with_name("run_low_prop_local_corridor_v230p.py")),
        "baseline_match": report["inward_bias_report_sha256"] == sha(
            ROOT / "artifacts/ape-robocasa-low-prop-capture-pose-bias-v234p/report.json"),
        "matched_scene_fingerprint": len({case["scene_fingerprint"]
                                          for case in report["episodes"]}) == 1,
        "episodes": rows,
    }
    output["evidence_integrity_pass"] = (
        output["experiment_source_match"] and output["v232_source_match"]
        and output["v230_source_match"] and output["baseline_match"] and
        output["matched_scene_fingerprint"] and
        all(row["geometry"]["video_sha_pass"] and
            row["geometry"]["video_frames_pass"] and
            row["geometry"]["prop_fixture_nonoverlap_pass"] and
            row["mid_packets"]["count_match"] and
            row["mid_packets"]["all_checks_pass"] and
            row["low_packets"]["count_match"] and
            row["low_packets"]["checks_pass"] and
            row["fault_schedule"]["all_pass"] and
            row["recovery"]["all_checks_pass"] and
            row["validity"]["all_checks_pass"] and
            row["delivery"]["count_match"] and
            row["delivery"]["all_checks_pass"] and
            row["actor_capture"]["count_match"] and
            row["actor_capture"]["all_checks_pass"] and
            row["low_projection"]["count_match"] and
            row["low_projection"]["all_checks_pass"] and
            row["age_gate"]["all_checks_pass"] and
            row["depth"]["all_checks_pass"] and
            row["depth"]["video_sha_pass"] and
            row["depth"]["video_frame_count_pass"] and
            row["owner"]["video_sha_pass"] and
            row["owner"]["video_frame_count_pass"] and
            row["owner"]["actor_outcome_matches_evidence"]
            for row in rows))
    output["all_5cm_margin_pass"] = all(
        row["geometry"]["ape_5cm_margin_pass"] for row in rows)
    (OUT / "audit.json").write_text(json.dumps(output, indent=2) + "\n")
    for row in rows:
        print(row["scenario"], row["route_state"], row["outcome"],
              "gap", round(row["geometry"]["nearest_ape_prop"]["signed_gap_m"], 3),
              "depth releases", row["depth"]["depth_releases"], flush=True)
    print("integrity", output["evidence_integrity_pass"],
          "margins", output["all_5cm_margin_pass"])
    if not output["evidence_integrity_pass"]:
        raise AssertionError("Depth-crosscheck evidence integrity failed")


if __name__ == "__main__":
    main()
