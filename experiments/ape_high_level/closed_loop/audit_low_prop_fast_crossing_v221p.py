"""Independently replay the fast low-prop paths, raw packets and owner claims."""

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
from run import ROOT, sha


SOURCE = Path(__file__)
OUT = ROOT / "artifacts/ape-robocasa-low-prop-fast-crossing-v220p"
REPORT = OUT / "report.json"


def fault_schedule(case, window):
    checks = []
    for source in case["raw_low_source_packets"]:
        with np.load(ROOT / source["path"], allow_pickle=False) as packet:
            status = packet["status"]
            index = int(packet["sample_index"])
        time_s = index * .4
        expected = 21 if window[0] <= time_s <= window[1] else 0
        actual = int(np.count_nonzero(status == 3))
        checks.append({"index": index, "actual": actual,
                       "expected": expected, "pass": actual == expected})
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
                "scenario": case["scenario"], "package": "fast_crossing",
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
                "owner": owner_boundary(case),
            })
    finally:
        route_audit.evaluator_scene = old_scene
        route_audit.prop_at = old_path
    output = {
        "schema": "robot-learning-lab.ape-low-prop-fast-crossing-audit.v1",
        "source_sha256": sha(SOURCE),
        "report_sha256": sha(REPORT),
        "experiment_source_match": report["source_sha256"] == sha(
            SOURCE.with_name("run_low_prop_fast_crossing_v220p.py")),
        "v216_source_match": report["v216_source_sha256"] == sha(
            SOURCE.with_name("run_low_prop_coverage_recover_v216p.py")),
        "matched_scene_fingerprint": len({case["scene_fingerprint"]
                                          for case in report["episodes"]}) == 1,
        "episodes": rows,
    }
    output["evidence_integrity_pass"] = (
        output["experiment_source_match"] and output["v216_source_match"]
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
        raise AssertionError("Fast-crossing audit integrity failed")


if __name__ == "__main__":
    main()
