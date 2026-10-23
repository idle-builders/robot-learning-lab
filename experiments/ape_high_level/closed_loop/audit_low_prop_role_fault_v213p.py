"""Audit V212p correlated low-source dropout against route and owner truth."""

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
from run import ROOT, sha


SOURCE = Path(__file__)
OUT = ROOT / "artifacts/ape-robocasa-low-prop-role-fault-v212p"
REPORT = OUT / "report.json"


def fault_schedule(case):
    checks = []
    for row, source in zip(case["fault_log"], case["raw_low_source_packets"]):
        with np.load(ROOT / source["path"], allow_pickle=False) as packet:
            status = packet["status"]
            index = int(packet["sample_index"])
        count = int(np.count_nonzero(status == 3))
        time_s = index * .4
        expected = 21 if 7.2 <= time_s <= 14.0 else 0
        checks.append({"index": index, "actual": count,
                       "logged": row["occluded_rays"],
                       "expected": expected,
                       "pass": count == expected == row["occluded_rays"]})
    return {"packets": len(checks),
            "all_pass": all(row["pass"] for row in checks),
            "first_failure": next((row for row in checks if not row["pass"]), None)}


def main():
    report = json.loads(REPORT.read_text())
    old_scene, old_path = route_audit.evaluator_scene, route_audit.prop_at
    route_audit.evaluator_scene = low_evaluator_scene
    route_audit.prop_at = evaluator_prop_at
    rows = []
    try:
        for case in report["episodes"]:
            geometry = route_audit.audit_case({
                "scenario": case["scenario"],
                "package": case["fault"],
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
                "owner": owner_boundary(case),
                "hold_actions": sum(d["decision"] ==
                                    "hold_and_reobserve_moving_object"
                                    for d in case["decision_history"]),
            })
    finally:
        route_audit.evaluator_scene = old_scene
        route_audit.prop_at = old_path
    output = {
        "schema": "robot-learning-lab.ape-low-prop-role-fault-audit.v1",
        "source_sha256": sha(SOURCE),
        "report_sha256": sha(REPORT),
        "experiment_source_match": report["source_sha256"] == sha(
            SOURCE.with_name("run_low_prop_role_fault_v212p.py")),
        "v210_source_match": report["v210_source_sha256"] == sha(
            SOURCE.with_name("run_low_prop_role_split_v210p.py")),
        "matched_scene_fingerprint": len({case["scene_fingerprint"]
                                          for case in report["episodes"]}) == 1,
        "episodes": rows,
    }
    output["evidence_integrity_pass"] = (
        output["experiment_source_match"] and output["v210_source_match"]
        and output["matched_scene_fingerprint"]
        and all(row["geometry"]["video_sha_pass"] and
                row["geometry"]["video_frames_pass"] and
                row["geometry"]["prop_fixture_nonoverlap_pass"] and
                row["mid_packets"]["count_match"] and
                row["mid_packets"]["all_checks_pass"] and
                row["low_packets"]["count_match"] and
                row["low_packets"]["checks_pass"] and
                row["fault_schedule"]["all_pass"] and
                row["owner"]["video_sha_pass"] and
                row["owner"]["video_frame_count_pass"] and
                row["owner"]["actor_outcome_matches_evidence"]
                for row in rows))
    (OUT / "audit.json").write_text(json.dumps(output, indent=2) + "\n")
    for row in rows:
        print(row["scenario"], row["fault"], row["route_state"],
              "gap", round(row["geometry"]["nearest_ape_prop"]["signed_gap_m"], 3),
              "holds", row["hold_actions"], flush=True)
    print("evidence integrity passed", output["evidence_integrity_pass"])
    if not output["evidence_integrity_pass"]:
        raise AssertionError("Fault audit integrity failed")


if __name__ == "__main__":
    main()
