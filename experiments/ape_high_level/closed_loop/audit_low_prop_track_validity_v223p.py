"""Audit V222p track-loss holds against packets, actions and full geometry."""

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
from run import ROOT, sha


SOURCE = Path(__file__)
OUT = ROOT / "artifacts/ape-robocasa-low-prop-track-validity-v222p"
REPORT = OUT / "report.json"
HOLDS = {"track_validity_lost", "track_validity_wait"}


def validity_replay(case, rule):
    events = case["track_validity_decisions"]
    route = case["route_episode"]["trace"]
    tracks = case["track_history"]
    streak = 0
    checks = []
    for event in events:
        index = event["scan_index"]
        track, row = tracks[index], route[index]
        velocity = track["velocity_xy_m_s"]
        speed = float(np.linalg.norm(velocity)) if velocity is not None else None
        gap = event["static_route_gap_m"]
        stable = (track["status"] == "tracked" and speed is not None and
                  speed < rule["quiet_speed_m_s"] and gap is not None and
                  gap > rule["path_clear_gap_m"])
        streak = streak + 1 if stable else 0
        kind = event["event"]
        if kind == "track_validity_lost":
            preceding = tracks[index - 1] if index > 0 else None
            rule_valid = (preceding is not None and
                          preceding["status"] == "tracked" and
                          track["status"] in ("shape_uncertain", "shape_confirmed") and
                          velocity is None and
                          event["last_tracked_path_distance_m"] <
                          .17 + .24 + rule["path_clear_gap_m"] +
                          rule["speed_bound_m_s"] * rule["reach_horizon_s"])
        elif kind == "track_validity_release":
            rule_valid = stable and streak >= rule["fresh_stable_packets"]
        elif kind == "track_validity_timeout_help":
            rule_valid = (event["time_s"] - events[0]["time_s"] >
                          rule["max_reobserve_s"] and
                          streak < rule["fresh_stable_packets"])
        else:
            rule_valid = kind == "track_validity_wait" and streak < rule[
                "fresh_stable_packets"]
        hold = kind in HOLDS
        requested = row["requested_xy_m"]
        next_pose_same = (index + 1 < len(route) and np.allclose(
            route[index + 1]["actual_xy_m"], row["actual_xy_m"], atol=1e-5))
        action_valid = (row["plan_status"] == event["result"] and
                        (not hold or
                         (requested is not None and
                          np.allclose(requested, row["estimated_xy_m"], atol=1e-5)
                          and next_pose_same and
                          len(row["motion_substeps"]) == 20 and
                          not any(sub["stopped_before_move"]
                                  for sub in row["motion_substeps"]))))
        timing_valid = abs(row["time_s"] - event["time_s"]) < 1e-4
        source_valid = (track["status"] == event["track_status"] and
                        (speed is None and event["track_speed_m_s"] is None or
                         speed is not None and event["track_speed_m_s"] is not None and
                         abs(speed - event["track_speed_m_s"]) < 1e-8))
        checks.append({"index": index, "event": kind,
                       "rule_valid": bool(rule_valid),
                       "action_valid": bool(action_valid),
                       "timing_valid": bool(timing_valid),
                       "source_valid": bool(source_valid),
                       "stable_streak_before_reset": streak})
    return {"events": len(events), "holds": sum(x["event"] in HOLDS
                                                   for x in checks),
            "releases": sum(x["event"] == "track_validity_release"
                            for x in checks),
            "help_timeouts": sum(x["event"] == "track_validity_timeout_help"
                                 for x in checks),
            "all_checks_pass": all(all(x[key] for key in
                                   ("rule_valid", "action_valid", "timing_valid",
                                    "source_valid")) for x in checks),
            "first_failure": next((x for x in checks if not all(x[key] for key in
                                   ("rule_valid", "action_valid", "timing_valid",
                                    "source_valid"))), None)}


def main():
    report = json.loads(REPORT.read_text())
    old_scene, old_path = route_audit.evaluator_scene, route_audit.prop_at
    route_audit.evaluator_scene = low_evaluator_scene
    route_audit.prop_at = evaluator_prop_at
    rows = []
    try:
        for case in report["episodes"]:
            geometry = route_audit.audit_case({
                "scenario": case["scenario"], "package": "track_validity",
                "episode": case["route_episode"]})
            rows.append({
                "scenario": case["scenario"]["id"],
                "fault": case["fault"],
                "route_state": case["route_state"],
                "outcome": case["outcome"],
                "geometry": geometry,
                "mid_packets": packets(case, 360),
                "low_packets": low_packets(case),
                "fault_schedule": fault_schedule(case, report["fault_window_s"]),
                "recovery": recovery_replay(case),
                "validity": validity_replay(case, report["actor_rule"]),
                "owner": owner_boundary(case),
            })
    finally:
        route_audit.evaluator_scene = old_scene
        route_audit.prop_at = old_path
    output = {
        "schema": "robot-learning-lab.ape-low-prop-track-validity-audit.v1",
        "source_sha256": sha(SOURCE),
        "report_sha256": sha(REPORT),
        "experiment_source_match": report["source_sha256"] == sha(
            SOURCE.with_name("run_low_prop_track_validity_v222p.py")),
        "v216_source_match": report["v216_source_sha256"] == sha(
            SOURCE.with_name("run_low_prop_coverage_recover_v216p.py")),
        "v220_report_match": report["v220_report_sha256"] == sha(
            ROOT / "artifacts/ape-robocasa-low-prop-fast-crossing-v220p/report.json"),
        "matched_scene_fingerprint": len({case["scene_fingerprint"]
                                          for case in report["episodes"]}) == 1,
        "episodes": rows,
    }
    output["evidence_integrity_pass"] = (
        output["experiment_source_match"] and output["v216_source_match"]
        and output["v220_report_match"] and output["matched_scene_fingerprint"] and
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
              "holds", row["validity"]["holds"],
              "release", row["validity"]["releases"],
              "help", row["validity"]["help_timeouts"], flush=True)
    print("evidence integrity passed", output["evidence_integrity_pass"],
          "all margins", output["all_5cm_margin_pass"])
    if not output["evidence_integrity_pass"]:
        raise AssertionError("Track-validity audit integrity failed")


if __name__ == "__main__":
    main()
