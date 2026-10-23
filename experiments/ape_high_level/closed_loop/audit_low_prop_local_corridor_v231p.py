"""Audit delayed packets, local low veto, owner result and 20 ms geometry."""

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
from run import ROOT, sha


SOURCE = Path(__file__)
OUT = ROOT / "artifacts/ape-robocasa-low-prop-local-corridor-v230p"
REPORT = OUT / "report.json"


def veto_replay(case, reserve):
    events = case["low_corridor_vetoes"]
    tracks = case["track_history"]
    route = case["route_episode"]["trace"]
    checks = []
    for event in events:
        index = event["scan_index"]
        track = tracks[index]
        required = (reserve["body_radius_m"] + float(track["radius_m"]) +
                    reserve["required_gap_m"] +
                    reserve["historical_stop_path_m"])
        checks.append({
            "index": index,
            "source_match": (track["status"] == event["track_status"] and
                             np.allclose(track["center_xy_m"],
                                         event["track_center_xy_m"], atol=1e-8) and
                             abs(float(track["radius_m"]) -
                                 event["track_radius_m"]) < 1e-8),
            "threshold_match": abs(required - event["required_separation_m"]) < 1e-8,
            "rule_pass": (event["lookahead_m"] == reserve["lookahead_m"] and
                          event["minimum_lookahead_center_separation_m"] < required),
            "action_match": (route[index]["plan_status"] in
                             ("hold_reobserve_stale_low_scan",
                              "ask_owner_clear_corridor")),
        })
    return {"events": len(events),
            "all_checks_pass": all(all(x[key] for key in (
                "source_match", "threshold_match", "rule_pass", "action_match"))
                for x in checks),
            "first_failure": next((x for x in checks if not all(x[key]
                for key in ("source_match", "threshold_match", "rule_pass",
                            "action_match"))), None)}


def main():
    report = json.loads(REPORT.read_text())
    old_scene, old_path = route_audit.evaluator_scene, route_audit.prop_at
    route_audit.evaluator_scene = low_evaluator_scene
    route_audit.prop_at = evaluator_prop_at
    rows = []
    try:
        for case in report["episodes"]:
            geometry = route_audit.audit_case({
                "scenario": case["scenario"], "package": "local_corridor",
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
                "delivery": delivery_replay(case, report["delay_window_s"]),
                "age_gate": age_gate_replay(case, report["age_limit_s"],
                                            report["fresh_release_packets"]),
                "low_veto": veto_replay(case, report["low_route_reserve"]),
                "owner": owner_boundary(case),
            })
    finally:
        route_audit.evaluator_scene = old_scene
        route_audit.prop_at = old_path
    output = {
        "schema": "robot-learning-lab.ape-low-prop-local-corridor-audit.v1",
        "source_sha256": sha(SOURCE),
        "report_sha256": sha(REPORT),
        "experiment_source_match": report["source_sha256"] == sha(
            SOURCE.with_name("run_low_prop_local_corridor_v230p.py")),
        "v227_source_match": report["v227_source_sha256"] == sha(
            SOURCE.with_name("run_low_prop_capture_age_v227p.py")),
        "historical_stop_match": report["historical_stop_report_sha256"] == sha(
            ROOT / "artifacts/ape-skill-contract-cove50-20261012/historical-stop-trace.json"),
        "matched_scene_fingerprint": len({case["scene_fingerprint"]
                                          for case in report["episodes"]}) == 1,
        "episodes": rows,
    }
    output["evidence_integrity_pass"] = (
        output["experiment_source_match"] and output["v227_source_match"]
        and output["historical_stop_match"] and output["matched_scene_fingerprint"] and
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
        print(row["scenario"], row["route_state"],
              "gap", round(row["geometry"]["nearest_ape_prop"]["signed_gap_m"], 3),
              "low_vetoes", row["low_veto"]["events"], flush=True)
    print("evidence integrity passed", output["evidence_integrity_pass"],
          "all margins", output["all_5cm_margin_pass"])
    if not output["evidence_integrity_pass"]:
        raise AssertionError("Local corridor audit integrity failed")


if __name__ == "__main__":
    main()
