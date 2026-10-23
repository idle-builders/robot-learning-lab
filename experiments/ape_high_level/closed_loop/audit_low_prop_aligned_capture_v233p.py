"""Audit estimated capture-pose use, delayed rays and full owner outcomes."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

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
from audit_low_prop_local_corridor_v231p import veto_replay
from run import ROOT, sha


SOURCE = Path(__file__)
OUT = ROOT / "artifacts/ape-robocasa-low-prop-aligned-capture-v232p"
REPORT = OUT / "report.json"
RESERVE = {"lookahead_m": .6, "body_radius_m": .17,
           "required_gap_m": .05, "historical_stop_path_m": .115}


def actor_capture_replay(case):
    events = case["aligned_tracker_events"]
    route = case["route_episode"]["trace"]
    delivered = case["low_deliveries"]
    tracks = case["track_history"]
    checks = []
    last_source = -1
    for index, event in enumerate(events):
        source = event["source_capture_index"]
        duplicate = source == last_source
        result = tracks[index]
        checks.append({
            "decision_index": index,
            "source_match": source == delivered[index]["capture_index"],
            "capture_pose_match": np.allclose(
                event["actor_capture_pose_xy_m"],
                route[source]["estimated_xy_m"], atol=1e-7),
            "receipt_pose_match": np.allclose(
                event["actor_receipt_pose_xy_m"],
                route[index]["estimated_xy_m"], atol=1e-7),
            "time_match": (abs(event["source_capture_time_s"] - source*.4) < 1e-5
                           and abs(event["receipt_time_s"] - index*.4) < 1e-5),
            "duplicate_action_match": (event["action"] ==
                                       ("reused_without_tracker_update" if duplicate
                                        else "capture_pose_update")),
            "result_match": (result["status"] == event["result_status"] and
                             result["source_capture_index"] == source and
                             np.allclose(result["actor_capture_pose_xy_m"],
                                         event["actor_capture_pose_xy_m"], atol=1e-7)),
        })
        last_source = source
    fields = ("source_match", "capture_pose_match", "receipt_pose_match",
              "time_match", "duplicate_action_match", "result_match")
    return {"count": len(events),
            "count_match": len(events) == len(route) == len(delivered) == len(tracks),
            "duplicate_captures": sum(event["action"] ==
                                      "reused_without_tracker_update"
                                      for event in events),
            "all_checks_pass": all(all(row[field] for field in fields)
                                   for row in checks),
            "first_failure": next((row for row in checks if not all(
                row[field] for field in fields)), None)}


def main():
    report = json.loads(REPORT.read_text())
    old_scene, old_path = route_audit.evaluator_scene, route_audit.prop_at
    route_audit.evaluator_scene = low_evaluator_scene
    route_audit.prop_at = evaluator_prop_at
    rows = []
    try:
        for case in report["episodes"]:
            geometry = route_audit.audit_case({
                "scenario": case["scenario"], "package": "aligned_capture",
                "episode": case["route_episode"]})
            rows.append({
                "scenario": case["scenario"]["id"],
                "profile": case["aligned_capture_profile"],
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
                "delivery": delivery_replay(case, report["delay_window_s"]),
                "actor_capture": actor_capture_replay(case),
                "age_gate": age_gate_replay(case, .2, 3),
                "low_veto": veto_replay(case, RESERVE)
                if case["aligned_capture_profile"] == "aligned_age_local"
                else {"events": 0, "all_checks_pass": True},
                "owner": owner_boundary(case),
            })
    finally:
        route_audit.evaluator_scene = old_scene
        route_audit.prop_at = old_path
    output = {
        "schema": "robot-learning-lab.ape-low-prop-aligned-capture-audit.v1",
        "source_sha256": sha(SOURCE),
        "report_sha256": sha(REPORT),
        "experiment_source_match": report["source_sha256"] == sha(
            SOURCE.with_name("run_low_prop_aligned_capture_v232p.py")),
        "v227_source_match": report["v227_source_sha256"] == sha(
            SOURCE.with_name("run_low_prop_capture_age_v227p.py")),
        "v230_source_match": report["v230_source_sha256"] == sha(
            SOURCE.with_name("run_low_prop_local_corridor_v230p.py")),
        "tracker_source_match": report["tracker_source_sha256"] == sha(
            SOURCE.with_name("novel_range_object.py")),
        "matched_scene_fingerprint": len({case["scene_fingerprint"]
                                          for case in report["episodes"]}) == 1,
        "episodes": rows,
    }
    output["evidence_integrity_pass"] = (
        output["experiment_source_match"] and output["v227_source_match"]
        and output["v230_source_match"] and output["tracker_source_match"]
        and output["matched_scene_fingerprint"] and
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
            row["age_gate"]["all_checks_pass"] and
            row["low_veto"]["all_checks_pass"] and
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
              "duplicates", row["actor_capture"]["duplicate_captures"], flush=True)
    print("evidence integrity passed", output["evidence_integrity_pass"],
          "all margins", output["all_5cm_margin_pass"])
    if not output["evidence_integrity_pass"]:
        raise AssertionError("Aligned-capture audit integrity failed")


if __name__ == "__main__":
    main()
