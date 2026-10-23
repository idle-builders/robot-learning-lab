"""Audit delayed RGB-D source pose/time, decisions and full owner episodes."""

from __future__ import annotations

import json
from pathlib import Path

import audit_robocasa_dynamic_suite_v440 as route_audit
from audit_robocasa_layout4_wait_and_remap_v462 import evaluator_prop_at
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
from audit_low_prop_depth_crosscheck_v237p import depth_replay
from run import ROOT, sha


SOURCE = Path(__file__)
OUT = ROOT / "artifacts/ape-robocasa-low-prop-depth-latency-v240p"
REPORT = OUT / "report.json"
CURRENT = ROOT / "artifacts/ape-robocasa-low-prop-depth-crosscheck-v236p/report.json"


def main():
    report = json.loads(REPORT.read_text())
    old_scene, old_path = route_audit.evaluator_scene, route_audit.prop_at
    route_audit.evaluator_scene = low_evaluator_scene
    route_audit.prop_at = evaluator_prop_at
    rows = []
    try:
        for case in report["episodes"]:
            geometry = route_audit.audit_case({
                "scenario": case["scenario"], "package": "depth_latency",
                "episode": case["route_episode"]})
            rows.append({
                "scenario": case["scenario"]["id"],
                "profile": case["depth_latency_profile"],
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
                "low_delivery": delivery_replay(case, (10.4, 12.4)),
                "actor_capture": actor_capture_replay(case),
                "low_projection": projection_replay(case),
                "low_age_gate": age_gate_replay(case, .2, 3),
                "depth": depth_replay(case),
                "owner": owner_boundary(case),
            })
    finally:
        route_audit.evaluator_scene = old_scene
        route_audit.prop_at = old_path
    output = {
        "schema": "robot-learning-lab.ape-low-prop-depth-latency-audit.v1",
        "source_sha256": sha(SOURCE),
        "report_sha256": sha(REPORT),
        "experiment_source_match": report["source_sha256"] == sha(
            SOURCE.with_name("run_low_prop_depth_latency_v240p.py")),
        "depth_source_match": report["depth_crosscheck_source_sha256"] == sha(
            SOURCE.with_name("run_low_prop_depth_crosscheck_v236p.py")),
        "current_report_match": report["current_depth_report_sha256"] == sha(CURRENT),
        "matched_scene_fingerprint": len({case["scene_fingerprint"]
                                          for case in report["episodes"]}) == 1,
        "episodes": rows,
    }
    output["evidence_integrity_pass"] = (
        output["experiment_source_match"] and output["depth_source_match"] and
        output["current_report_match"] and
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
            row["low_delivery"]["count_match"] and
            row["low_delivery"]["all_checks_pass"] and
            row["actor_capture"]["count_match"] and
            row["actor_capture"]["all_checks_pass"] and
            row["low_projection"]["count_match"] and
            row["low_projection"]["all_checks_pass"] and
            row["low_age_gate"]["all_checks_pass"] and
            row["depth"]["count_match"] and
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
        print(row["scenario"], row["profile"], row["route_state"],
              "gap", round(row["geometry"]["nearest_ape_prop"]["signed_gap_m"], 3),
              "depth releases", row["depth"]["depth_releases"], flush=True)
    print("integrity", output["evidence_integrity_pass"],
          "margins", output["all_5cm_margin_pass"])
    if not output["evidence_integrity_pass"]:
        raise AssertionError("Depth latency audit integrity failed")


if __name__ == "__main__":
    main()
