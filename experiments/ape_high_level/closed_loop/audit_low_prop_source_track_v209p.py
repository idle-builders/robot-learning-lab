"""Audit separate low tracker packets and full owner/geometry outcome."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

import audit_robocasa_dynamic_suite_v440 as route_audit
from audit_robocasa_layout4_wait_and_remap_v462 import evaluator_prop_at
from audit_low_prop_sensor_join_v207p import (
    low_evaluator_scene, owner_boundary, packets,
)
from run import ROOT, sha


SOURCE = Path(__file__)
OUT = ROOT / "artifacts/ape-robocasa-low-prop-source-track-v208p"
REPORT = OUT / "report.json"


def low_packets(case):
    scans = case["raw_low_source_packets"]
    checks = []
    for index, source in enumerate(scans):
        path = ROOT / source["path"]
        with np.load(path, allow_pickle=False) as packet:
            ranges = packet["ranges_m"]
            status = packet["status"]
            stored_index = int(packet["sample_index"])
        valid = status == 0
        checks.append({
            "index": index,
            "hash_match": sha(path) == source["sha256"],
            "index_match": stored_index == index,
            "beam_count_match": len(ranges) == len(status) == 72,
            "invalid_unmeasured": bool(np.all(np.isnan(ranges[~valid]))),
            "valid_finite": bool(np.all(np.isfinite(ranges[valid]) &
                                        (ranges[valid] >= .06) &
                                        (ranges[valid] <= 5.02))),
        })
    return {"count": len(scans),
            "count_match": len(scans) == len(case["route_episode"]["trace"]),
            "checks_pass": all(all(v for k, v in check.items() if k != "index")
                               for check in checks),
            "first_failure": next((check for check in checks
                                   if not all(v for k, v in check.items()
                                              if k != "index")), None)}


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
                "package": "separate_low_tracker",
                "episode": case["route_episode"]})
            rows.append({
                "scenario": case["scenario"]["id"],
                "route_state": case["route_state"],
                "outcome": case["outcome"],
                "geometry": geometry,
                "fused_packets": packets(case, 360),
                "low_packets": low_packets(case),
                "owner": owner_boundary(case),
                "tracked_frames": sum(t["status"] == "tracked"
                                      for t in case["track_history"]),
                "hold_actions": sum(d["decision"] ==
                                    "hold_and_reobserve_moving_object"
                                    for d in case["decision_history"]),
            })
    finally:
        route_audit.evaluator_scene = old_scene
        route_audit.prop_at = old_path
    output = {
        "schema": "robot-learning-lab.ape-low-prop-source-track-audit.v1",
        "source_sha256": sha(SOURCE),
        "report_sha256": sha(REPORT),
        "experiment_source_match": report["source_sha256"] == sha(
            SOURCE.with_name("run_low_prop_source_track_v208p.py")),
        "v206_source_match": report["v206_source_sha256"] == sha(
            SOURCE.with_name("run_low_prop_sensor_join_v206p.py")),
        "matched_scene_fingerprint": len({case["scene_fingerprint"]
                                          for case in report["episodes"]}) == 1,
        "episodes": rows,
    }
    output["evidence_integrity_pass"] = (
        output["experiment_source_match"] and output["v206_source_match"]
        and output["matched_scene_fingerprint"]
        and all(row["geometry"]["video_sha_pass"] and
                row["geometry"]["video_frames_pass"] and
                row["geometry"]["prop_fixture_nonoverlap_pass"] and
                row["fused_packets"]["count_match"] and
                row["fused_packets"]["all_checks_pass"] and
                row["low_packets"]["count_match"] and
                row["low_packets"]["checks_pass"] and
                row["owner"]["video_sha_pass"] and
                row["owner"]["video_frame_count_pass"] and
                row["owner"]["actor_outcome_matches_evidence"]
                for row in rows))
    (OUT / "audit.json").write_text(json.dumps(output, indent=2) + "\n")
    for row in rows:
        print(row["scenario"], row["route_state"], row["outcome"],
              "gap", round(row["geometry"]["nearest_ape_prop"]["signed_gap_m"], 3),
              "hold", row["hold_actions"], "track", row["tracked_frames"],
              flush=True)
    print("evidence integrity passed", output["evidence_integrity_pass"])
    if not output["evidence_integrity_pass"]:
        raise AssertionError("Source-track audit integrity failed")


if __name__ == "__main__":
    main()
