"""Audit V214p raw-beam refusal and stopped full-horizon geometry."""

from __future__ import annotations

import json
import math
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
OUT = ROOT / "artifacts/ape-robocasa-low-prop-coverage-gate-v214p"
REPORT = OUT / "report.json"


def gate_replay(case):
    checks = []
    for event in case["coverage_gate_log"]:
        index = event["scan_index"]
        with np.load(ROOT / case["raw_low_source_packets"][index]["path"],
                     allow_pickle=False) as packet:
            status = packet["status"]
        bearing = event["first_occluded_cell"]["bearing_rad"]
        center = int(round((bearing + math.pi) / (2 * math.pi) * 72)) % 72
        neighbors = [int(status[(center + offset) % 72])
                     for offset in (-1, 0, 1)]
        row = case["route_episode"]["trace"][index]
        checks.append({
            "scan_index": index,
            "status_replay_match": neighbors == [3, 3, 3] ==
                                   event["first_occluded_cell"]["statuses"],
            "decision_replay_match": row["plan_status"] ==
                                     "low_source_occluded_stop",
            "no_move_request": row["requested_xy_m"] is None,
            "time_match": abs(row["time_s"] - event["time_s"]) <= 1e-5,
        })
    return {"events": len(checks),
            "all_checks_pass": all(all(v for k, v in row.items()
                                   if k != "scan_index") for row in checks),
            "first_failure": next((row for row in checks
                                   if not all(v for k, v in row.items()
                                              if k != "scan_index")), None)}


def main():
    report = json.loads(REPORT.read_text())
    old_scene, old_path = route_audit.evaluator_scene, route_audit.prop_at
    route_audit.evaluator_scene = low_evaluator_scene
    route_audit.prop_at = evaluator_prop_at
    rows = []
    try:
        for case in report["episodes"]:
            geometry = route_audit.audit_case({
                "scenario": case["scenario"], "package": "coverage_gate",
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
                "gate_replay": gate_replay(case),
                "owner": owner_boundary(case),
            })
    finally:
        route_audit.evaluator_scene = old_scene
        route_audit.prop_at = old_path
    output = {
        "schema": "robot-learning-lab.ape-low-prop-coverage-gate-audit.v1",
        "source_sha256": sha(SOURCE),
        "report_sha256": sha(REPORT),
        "experiment_source_match": report["source_sha256"] == sha(
            SOURCE.with_name("run_low_prop_coverage_gate_v214p.py")),
        "v212_source_match": report["v212_source_sha256"] == sha(
            SOURCE.with_name("run_low_prop_role_fault_v212p.py")),
        "matched_scene_fingerprint": len({case["scene_fingerprint"]
                                          for case in report["episodes"]}) == 1,
        "episodes": rows,
    }
    output["evidence_integrity_pass"] = (
        output["experiment_source_match"] and output["v212_source_match"]
        and output["matched_scene_fingerprint"]
        and all(row["geometry"]["video_sha_pass"] and
                row["geometry"]["video_frames_pass"] and
                row["geometry"]["prop_fixture_nonoverlap_pass"] and
                row["mid_packets"]["count_match"] and
                row["mid_packets"]["all_checks_pass"] and
                row["low_packets"]["count_match"] and
                row["low_packets"]["checks_pass"] and
                row["fault_schedule"]["all_pass"] and
                row["gate_replay"]["all_checks_pass"] and
                row["owner"]["video_sha_pass"] and
                row["owner"]["video_frame_count_pass"] and
                row["owner"]["actor_outcome_matches_evidence"]
                for row in rows))
    (OUT / "audit.json").write_text(json.dumps(output, indent=2) + "\n")
    for row in rows:
        print(row["scenario"], row["fault"], row["route_state"],
              "gap", round(row["geometry"]["nearest_ape_prop"]["signed_gap_m"], 3),
              "gate", row["gate_replay"]["events"], flush=True)
    print("evidence integrity passed", output["evidence_integrity_pass"])
    if not output["evidence_integrity_pass"]:
        raise AssertionError("Coverage-gate audit integrity failed")


if __name__ == "__main__":
    main()
