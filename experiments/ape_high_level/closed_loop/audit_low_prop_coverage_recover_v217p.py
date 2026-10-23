"""Replay V216p hold/release evidence and full-horizon owner geometry."""

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
from audit_low_prop_role_fault_v213p import fault_schedule
from run import ROOT, sha


SOURCE = Path(__file__)
OUT = ROOT / "artifacts/ape-robocasa-low-prop-coverage-recover-v216p"
REPORT = OUT / "report.json"
HOLDS = {"first_occlusion", "occluded_wait", "fresh_wait"}


def recovery_replay(case):
    events = case["coverage_recovery_decisions"]
    route = case["route_episode"]["trace"]
    blocked = None
    clear_streak = 0
    checks = []
    for event in events:
        index = event["scan_index"]
        with np.load(ROOT / case["raw_low_source_packets"][index]["path"],
                     allow_pickle=False) as packet:
            status = packet["status"]
        if event["event"] == "first_occlusion":
            blocked = event["blocked_beams"]
            clear_streak = 0
            sensor_valid = (len(blocked) == 3 and
                            all(int(status[beam]) == 3 for beam in blocked))
        else:
            healthy = all(int(status[beam]) != 3 for beam in blocked)
            clear_streak = clear_streak + 1 if healthy else 0
            sensor_valid = (
                (event["event"] == "occluded_wait" and not healthy) or
                (event["event"] == "fresh_wait" and healthy and clear_streak < 3) or
                (event["event"] == "fresh_release" and healthy and clear_streak >= 3))
        row = route[index]
        hold = event["event"] in HOLDS
        requested = row["requested_xy_m"]
        next_pose_same = (index + 1 < len(route) and np.allclose(
            route[index + 1]["actual_xy_m"], row["actual_xy_m"],
            atol=1e-5))
        action_valid = (row["plan_status"] == event["result"] and
                        (not hold or
                         (requested is not None and
                          np.allclose(requested, row["estimated_xy_m"], atol=1e-5)
                          and next_pose_same and
                          len(row["motion_substeps"]) == 20 and
                          not any(sub["stopped_before_move"]
                                  for sub in row["motion_substeps"]))))
        checks.append({"index": index, "event": event["event"],
                       "sensor_valid": bool(sensor_valid),
                       "action_valid": bool(action_valid),
                       "fresh_streak": clear_streak})
    return {"events": len(events),
            "holds": sum(row["event"] in HOLDS for row in checks),
            "releases": sum(row["event"] == "fresh_release" for row in checks),
            "all_checks_pass": all(row["sensor_valid"] and row["action_valid"]
                                   for row in checks),
            "first_failure": next((row for row in checks
                                   if not (row["sensor_valid"] and
                                           row["action_valid"])), None)}


def main():
    report = json.loads(REPORT.read_text())
    old_scene, old_path = route_audit.evaluator_scene, route_audit.prop_at
    route_audit.evaluator_scene = low_evaluator_scene
    route_audit.prop_at = evaluator_prop_at
    rows = []
    try:
        for case in report["episodes"]:
            geometry = route_audit.audit_case({
                "scenario": case["scenario"], "package": "coverage_recovery",
                "episode": case["route_episode"]})
            rows.append({
                "scenario": case["scenario"]["id"],
                "fault": case["fault"],
                "route_state": case["route_state"],
                "outcome": case["outcome"],
                "geometry": geometry,
                "mid_packets": packets(case, 360),
                "low_packets": low_packets(case),
                "fault_schedule": fault_schedule(case),
                "recovery": recovery_replay(case),
                "owner": owner_boundary(case),
            })
    finally:
        route_audit.evaluator_scene = old_scene
        route_audit.prop_at = old_path
    output = {
        "schema": "robot-learning-lab.ape-low-prop-coverage-recover-audit.v1",
        "source_sha256": sha(SOURCE),
        "report_sha256": sha(REPORT),
        "experiment_source_match": report["source_sha256"] == sha(
            SOURCE.with_name("run_low_prop_coverage_recover_v216p.py")),
        "matched_scene_fingerprint": len({case["scene_fingerprint"]
                                          for case in report["episodes"]}) == 1,
        "episodes": rows,
    }
    output["evidence_integrity_pass"] = (
        output["experiment_source_match"] and
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
            row["owner"]["video_sha_pass"] and
            row["owner"]["video_frame_count_pass"] and
            row["owner"]["actor_outcome_matches_evidence"]
            for row in rows))
    output["all_5cm_margin_pass"] = all(
        row["geometry"]["ape_5cm_margin_pass"] for row in rows)
    (OUT / "audit.json").write_text(json.dumps(output, indent=2) + "\n")
    for row in rows:
        print(row["scenario"], row["fault"], row["route_state"],
              "gap", round(row["geometry"]["nearest_ape_prop"]["signed_gap_m"], 3),
              "holds", row["recovery"]["holds"],
              "release", row["recovery"]["releases"], flush=True)
    print("evidence integrity passed", output["evidence_integrity_pass"],
          "all margins", output["all_5cm_margin_pass"])
    if not output["evidence_integrity_pass"]:
        raise AssertionError("Coverage-recovery audit integrity failed")


if __name__ == "__main__":
    main()
