"""Audit source-role V210p against saved rays, 20 ms gaps and owner frames."""

from __future__ import annotations

import json
from pathlib import Path

import audit_robocasa_dynamic_suite_v440 as route_audit
from audit_robocasa_layout4_wait_and_remap_v462 import evaluator_prop_at
from audit_low_prop_sensor_join_v207p import (
    low_evaluator_scene, owner_boundary, packets,
)
from audit_low_prop_source_track_v209p import low_packets
from run import ROOT, sha


SOURCE = Path(__file__)
OUT = ROOT / "artifacts/ape-robocasa-low-prop-role-split-v210p"
REPORT = OUT / "report.json"


def main():
    report = json.loads(REPORT.read_text())
    old_scene, old_path = route_audit.evaluator_scene, route_audit.prop_at
    route_audit.evaluator_scene = low_evaluator_scene
    route_audit.prop_at = evaluator_prop_at
    rows = []
    try:
        for case in report["episodes"]:
            geometry = route_audit.audit_case({
                "scenario": case["scenario"], "package": "role_split",
                "episode": case["route_episode"]})
            rows.append({
                "scenario": case["scenario"]["id"],
                "route_state": case["route_state"],
                "outcome": case["outcome"],
                "geometry": geometry,
                "mid_packets": packets(case, 360),
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
        "schema": "robot-learning-lab.ape-low-prop-role-split-audit.v1",
        "source_sha256": sha(SOURCE),
        "report_sha256": sha(REPORT),
        "experiment_source_match": report["source_sha256"] == sha(
            SOURCE.with_name("run_low_prop_role_split_v210p.py")),
        "v206_source_match": report["v206_source_sha256"] == sha(
            SOURCE.with_name("run_low_prop_sensor_join_v206p.py")),
        "v208_source_match": report["v208_source_sha256"] == sha(
            SOURCE.with_name("run_low_prop_source_track_v208p.py")),
        "matched_scene_fingerprint": len({case["scene_fingerprint"]
                                          for case in report["episodes"]}) == 1,
        "episodes": rows,
    }
    output["evidence_integrity_pass"] = (
        output["experiment_source_match"] and output["v206_source_match"]
        and output["v208_source_match"] and
        output["matched_scene_fingerprint"]
        and all(row["geometry"]["video_sha_pass"] and
                row["geometry"]["video_frames_pass"] and
                row["geometry"]["prop_fixture_nonoverlap_pass"] and
                row["mid_packets"]["count_match"] and
                row["mid_packets"]["all_checks_pass"] and
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
        raise AssertionError("Role-split audit integrity failed")


if __name__ == "__main__":
    main()
