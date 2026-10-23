"""Audit true captured/delivered low packets and full owner-route geometry."""

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
from run import ROOT, sha


SOURCE = Path(__file__)
OUT = ROOT / "artifacts/ape-robocasa-low-prop-capture-age-v227p"
REPORT = OUT / "report.json"
HOLDS = {"first_stale", "stale_wait", "fresh_wait"}


def delivery_replay(case, window):
    captured = case["captured_low_packets"]
    delivered = case["low_deliveries"]
    raw = case["raw_low_source_packets"]
    route = case["route_episode"]["trace"]
    checks = []
    for index, delivery in enumerate(delivered):
        source = delivery["capture_index"]
        with np.load(ROOT / captured[source]["path"], allow_pickle=False) as origin:
            with np.load(ROOT / raw[index]["path"], allow_pickle=False) as receipt:
                same = (np.array_equal(origin["ranges_m"], receipt["ranges_m"],
                                       equal_nan=True) and
                        np.array_equal(origin["status"], receipt["status"]))
                source_pose = origin["actual_capture_xy_m"].tolist()
                source_index = int(origin["sample_index"])
                capture_s = float(origin["capture_time_s"])
        expected_source = index - 1 if window[0] <= index * .4 <= window[1] else index
        check = {
            "index": index, "source": source,
            "capture_hash_pass": sha(ROOT / captured[source]["path"]) ==
            captured[source]["sha256"],
            "expected_source_pass": source == expected_source,
            "array_match": same,
            "source_index_pass": source_index == source,
            "source_pose_pass": np.allclose(source_pose,
                                            route[source]["actual_xy_m"], atol=1e-6),
            "time_pass": (abs(capture_s - source * .4) < 1e-5 and
                          abs(delivery["decision_time_s"] - index * .4) < 1e-5 and
                          abs(delivery["age_s"] - (index-source)*.4) < 1e-5),
        }
        checks.append(check)
    return {"count": len(checks),
            "count_match": len(captured) == len(delivered) == len(raw) == len(route),
            "all_checks_pass": all(all(row[key] for key in (
                "capture_hash_pass", "expected_source_pass", "array_match",
                "source_index_pass", "source_pose_pass", "time_pass"))
                for row in checks),
            "first_failure": next((row for row in checks if not all(row[key]
                for key in ("capture_hash_pass", "expected_source_pass",
                            "array_match", "source_index_pass", "source_pose_pass",
                            "time_pass"))), None)}


def age_gate_replay(case, limit, required):
    events = case["capture_age_gate_events"]
    route = case["route_episode"]["trace"]
    deliveries = case["low_deliveries"]
    checks = []
    streak = 0
    for event in events:
        index = event["scan_index"]
        row = route[index]
        delivery = deliveries[index]
        stale = delivery["age_s"] > limit
        streak = 0 if stale else streak + 1
        kind = event["event"]
        if kind in ("first_stale", "stale_wait"):
            rule_valid = stale and streak == 0
        elif kind == "fresh_wait":
            rule_valid = not stale and streak < required
        elif kind == "fresh_release":
            rule_valid = not stale and streak >= required
        else:
            rule_valid = kind == "age_timeout_help"
        hold = kind in HOLDS
        next_pose_same = (index + 1 < len(route) and np.allclose(
            route[index+1]["actual_xy_m"], row["actual_xy_m"], atol=1e-5))
        action_valid = (row["plan_status"] == event["result"] and
                        (not hold or
                         (row["requested_xy_m"] is not None and
                          np.allclose(row["requested_xy_m"], row["estimated_xy_m"],
                                      atol=1e-5) and next_pose_same and
                          len(row["motion_substeps"]) == 20)))
        checks.append({"index": index, "event": kind,
                       "source_valid": (event["capture_index"] ==
                                        delivery["capture_index"] and
                                        abs(event["age_s"] - delivery["age_s"]) < 1e-5),
                       "rule_valid": rule_valid,
                       "action_valid": bool(action_valid)})
    return {"events": len(events), "holds": sum(x["event"] in HOLDS
                                                   for x in checks),
            "all_checks_pass": all(x["source_valid"] and x["rule_valid"]
                                   and x["action_valid"] for x in checks),
            "first_failure": next((x for x in checks if not
                                   (x["source_valid"] and x["rule_valid"]
                                    and x["action_valid"])), None)}


def main():
    report = json.loads(REPORT.read_text())
    old_scene, old_path = route_audit.evaluator_scene, route_audit.prop_at
    route_audit.evaluator_scene = low_evaluator_scene
    route_audit.prop_at = evaluator_prop_at
    rows = []
    try:
        for case in report["episodes"]:
            geometry = route_audit.audit_case({
                "scenario": case["scenario"], "package": "capture_age",
                "episode": case["route_episode"]})
            rows.append({
                "scenario": case["scenario"]["id"],
                "profile": case["capture_age_profile"],
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
                "owner": owner_boundary(case),
            })
    finally:
        route_audit.evaluator_scene = old_scene
        route_audit.prop_at = old_path
    output = {
        "schema": "robot-learning-lab.ape-low-prop-capture-age-audit.v1",
        "source_sha256": sha(SOURCE),
        "report_sha256": sha(REPORT),
        "experiment_source_match": report["source_sha256"] == sha(
            SOURCE.with_name("run_low_prop_capture_age_v227p.py")),
        "matched_scene_fingerprint": len({case["scene_fingerprint"]
                                          for case in report["episodes"]}) == 1,
        "episodes": rows,
    }
    output["evidence_integrity_pass"] = (
        output["experiment_source_match"] and output["matched_scene_fingerprint"] and
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
              "age_holds", row["age_gate"]["holds"], flush=True)
    print("evidence integrity passed", output["evidence_integrity_pass"],
          "all margins", output["all_5cm_margin_pass"])
    if not output["evidence_integrity_pass"]:
        raise AssertionError("Capture-age audit integrity failed")


if __name__ == "__main__":
    main()
