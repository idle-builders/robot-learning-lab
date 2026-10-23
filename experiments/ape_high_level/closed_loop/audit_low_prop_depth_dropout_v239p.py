"""Audit complete missing-depth fallback against the inward-bias baseline."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

import audit_robocasa_dynamic_suite_v440 as route_audit
from audit_robocasa_layout4_wait_and_remap_v462 import evaluator_prop_at
from audit_low_prop_sensor_join_v207p import low_evaluator_scene, owner_boundary
from audit_low_prop_capture_age_v228p import delivery_replay, age_gate_replay
from audit_low_prop_aligned_capture_v233p import actor_capture_replay
from audit_low_prop_capture_pose_bias_v235p import projection_replay
from audit_low_prop_depth_crosscheck_v237p import depth_replay
from run import ROOT, sha


SOURCE = Path(__file__)
OUT = ROOT / "artifacts/ape-robocasa-low-prop-depth-dropout-v238p"
REPORT = OUT / "report.json"
BASE = ROOT / "artifacts/ape-robocasa-low-prop-capture-pose-bias-v234p/report.json"


def missing_depth(case):
    checks = []
    for capture in case["low_depth_captures"]:
        if capture["packet"] is None:
            continue
        with np.load(ROOT / capture["packet"]["path"],
                     allow_pickle=False) as packet:
            checks.append(bool(np.isnan(packet["measured_depth_m"]).all()))
    return {"packets": len(checks), "all_missing": bool(checks) and all(checks)}


def same_route(case, baseline):
    current = case["route_episode"]["trace"]
    old = baseline["route_episode"]["trace"]
    fields = ("actual_xy_m", "estimated_xy_m", "plan_status",
              "requested_xy_m")
    checks = []
    for a, b in zip(current, old):
        checks.append(all(a[field] == b[field] for field in fields))
    return {"count_match": len(current) == len(old),
            "all_route_decisions_match": len(current) == len(old) and all(checks),
            "outcome_match": (case["route_state"] == baseline["route_state"]
                              and case["outcome"] == baseline["outcome"])}


def main():
    report = json.loads(REPORT.read_text())
    baseline = json.loads(BASE.read_text())
    inward = {case["scenario"]["id"]: case for case in baseline["episodes"]
              if case["low_projection_bias_name"] == "toward_route_3cm"}
    old_scene, old_path = route_audit.evaluator_scene, route_audit.prop_at
    route_audit.evaluator_scene = low_evaluator_scene
    route_audit.prop_at = evaluator_prop_at
    rows = []
    try:
        for case in report["episodes"]:
            geometry = route_audit.audit_case({
                "scenario": case["scenario"], "package": "depth_dropout",
                "episode": case["route_episode"]})
            rows.append({
                "scenario": case["scenario"]["id"],
                "route_state": case["route_state"],
                "outcome": case["outcome"],
                "geometry": geometry,
                "missing_depth": missing_depth(case),
                "depth": depth_replay(case),
                "route_pair": same_route(case, inward[case["scenario"]["id"]]),
                "delivery": delivery_replay(case, (10.4, 12.4)),
                "actor_capture": actor_capture_replay(case),
                "low_projection": projection_replay(case),
                "age_gate": age_gate_replay(case, .2, 3),
                "owner": owner_boundary(case),
            })
    finally:
        route_audit.evaluator_scene = old_scene
        route_audit.prop_at = old_path
    output = {
        "schema": "robot-learning-lab.ape-low-prop-depth-dropout-audit.v1",
        "source_sha256": sha(SOURCE),
        "report_sha256": sha(REPORT),
        "experiment_source_match": report["source_sha256"] == sha(
            SOURCE.with_name("run_low_prop_depth_dropout_v238p.py")),
        "depth_crosscheck_source_match": report[
            "depth_crosscheck_source_sha256"] == sha(
                SOURCE.with_name("run_low_prop_depth_crosscheck_v236p.py")),
        "depth_crosscheck_report_match": report[
            "depth_crosscheck_report_sha256"] == sha(
                ROOT / "artifacts/ape-robocasa-low-prop-depth-crosscheck-v236p/report.json"),
        "matched_scene_fingerprint": len({case["scene_fingerprint"]
                                          for case in report["episodes"]}) == 1,
        "episodes": rows,
    }
    output["evidence_integrity_pass"] = (
        output["experiment_source_match"] and
        output["depth_crosscheck_source_match"] and
        output["depth_crosscheck_report_match"] and
        output["matched_scene_fingerprint"] and
        all(row["geometry"]["video_sha_pass"] and
            row["geometry"]["video_frames_pass"] and
            row["geometry"]["prop_fixture_nonoverlap_pass"] and
            row["missing_depth"]["all_missing"] and
            row["depth"]["all_checks_pass"] and
            row["depth"]["video_sha_pass"] and
            row["depth"]["video_frame_count_pass"] and
            row["depth"]["depth_releases"] == 0 and
            row["route_pair"]["count_match"] and
            row["route_pair"]["all_route_decisions_match"] and
            row["route_pair"]["outcome_match"] and
            row["delivery"]["count_match"] and
            row["delivery"]["all_checks_pass"] and
            row["actor_capture"]["count_match"] and
            row["actor_capture"]["all_checks_pass"] and
            row["low_projection"]["count_match"] and
            row["low_projection"]["all_checks_pass"] and
            row["age_gate"]["all_checks_pass"] and
            row["owner"]["video_sha_pass"] and
            row["owner"]["video_frame_count_pass"] and
            row["owner"]["actor_outcome_matches_evidence"]
            for row in rows))
    output["all_5cm_margin_pass"] = all(
        row["geometry"]["ape_5cm_margin_pass"] for row in rows)
    (OUT / "audit.json").write_text(json.dumps(output, indent=2) + "\n")
    for row in rows:
        print(row["scenario"], row["route_state"], row["outcome"],
              "gap", round(row["geometry"]["nearest_ape_prop"]["signed_gap_m"], 3),
              "missing packets", row["missing_depth"]["packets"], flush=True)
    print("integrity", output["evidence_integrity_pass"],
          "margins", output["all_5cm_margin_pass"])
    if not output["evidence_integrity_pass"]:
        raise AssertionError("Depth-dropout evidence integrity failed")


if __name__ == "__main__":
    main()
