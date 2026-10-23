"""Audit later/nearer crossing, including a prop that enters the held pose."""

from __future__ import annotations

import json
from pathlib import Path

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
import run_low_prop_track_validity_v222p as v222


SOURCE = Path(__file__)
OUT = ROOT / "artifacts/ape-robocasa-low-prop-held-pose-v224p"
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
                "scenario": case["scenario"], "package": "held_pose",
                "episode": case["route_episode"]})
            rows.append({
                "scenario": case["scenario"]["id"],
                "route_state": case["route_state"],
                "outcome": case["outcome"],
                "geometry": geometry,
                "mid_packets": packets(case, 360),
                "low_packets": low_packets(case),
                "fault_schedule": fault_schedule(case, report["fault_window_s"]),
                "recovery": recovery_replay(case),
                "validity": validity_replay(case, {
                    "quiet_speed_m_s": v222.QUIET_SPEED_M_S,
                    "path_clear_gap_m": v222.PATH_CLEAR_GAP_M,
                    "speed_bound_m_s": v222.SPEED_BOUND_M_S,
                    "reach_horizon_s": v222.REACH_HORIZON_S,
                    "fresh_stable_packets": v222.FRESH_STABLE_PACKETS,
                    "max_reobserve_s": v222.MAX_REOBSERVE_S,
                }),
                "owner": owner_boundary(case),
            })
    finally:
        route_audit.evaluator_scene = old_scene
        route_audit.prop_at = old_path
    output = {
        "schema": "robot-learning-lab.ape-low-prop-held-pose-audit.v1",
        "source_sha256": sha(SOURCE),
        "report_sha256": sha(REPORT),
        "experiment_source_match": report["source_sha256"] == sha(
            SOURCE.with_name("run_low_prop_held_pose_v224p.py")),
        "v222_source_match": report["v222_source_sha256"] == sha(
            SOURCE.with_name("run_low_prop_track_validity_v222p.py")),
        "matched_scene_fingerprint": len({case["scene_fingerprint"]
                                          for case in report["episodes"]}) == 1,
        "episodes": rows,
    }
    output["evidence_integrity_pass"] = (
        output["experiment_source_match"] and output["v222_source_match"]
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
            row["owner"]["video_sha_pass"] and
            row["owner"]["video_frame_count_pass"] and
            row["owner"]["actor_outcome_matches_evidence"]
            for row in rows))
    output["all_5cm_margin_pass"] = all(
        row["geometry"]["ape_5cm_margin_pass"] for row in rows)
    (OUT / "audit.json").write_text(json.dumps(output, indent=2) + "\n")
    for row in rows:
        nearest = row["geometry"]["nearest_ape_prop"]
        print(row["scenario"], row["route_state"],
              "gap", round(nearest["signed_gap_m"], 3),
              "phase", nearest["phase"],
              "holds", row["validity"]["holds"], flush=True)
    print("evidence integrity passed", output["evidence_integrity_pass"],
          "all margins", output["all_5cm_margin_pass"])
    if not output["evidence_integrity_pass"]:
        raise AssertionError("Held-pose audit integrity failed")


if __name__ == "__main__":
    main()
