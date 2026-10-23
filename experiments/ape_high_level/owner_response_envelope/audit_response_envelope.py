"""Recheck complete response-envelope episodes and their saved media."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
import sys

import numpy as np


ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / "experiments/ape_high_level/closed_loop"))
import audit_robocasa_first_return_person_reacquire_v578 as geometry_audit

REPORT = ROOT / "artifacts/ape-owner-response-envelope/v2/report.json"


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def main():
    report = json.loads(REPORT.read_text())
    cases = report["cases"]
    episodes = report["episodes"]
    pre_show = [tuple(json.dumps(row, sort_keys=True) for row in ep["trace"]
                      if row["stage"] in ("choose_pre_show_standoff",
                                          "pre_show_backtrack",
                                          "pre_show_motion_assessment"))
                for ep in episodes]
    checks = {
        "run_source_sha_pass": report["source_sha256"] == sha256(
            ROOT / "experiments/ape_high_level/owner_response_envelope/run_response_envelope.py"),
        "geometry_audit_source_sha_pass": report["geometry_audit_source_sha256"] ==
            sha256(geometry_audit.__file__),
        "native_backward_matrix_sha_pass": report["native_backward_matrix_sha256"] ==
            sha256(report["native_backward_matrix_path"]),
        "matched_scene": len({ep["scene_fingerprint"] for ep in episodes}) == 1,
        "identical_pre_show_trajectory": len(set(pre_show)) == 1,
        "case_count": len(cases) == len(episodes) == 7,
    }
    rows = []
    for case, ep in zip(cases, episodes):
        audit = geometry_audit.audit_episode(ep)
        audit.pop("geometry_samples", None)
        ticks = [row for row in ep["trace"] if row["stage"] ==
                 "owner_reacquire_guard_tick"]
        start = np.asarray(ticks[0]["ape_xy_m"])
        alarm = case["alarm_time_s"]
        speed = case["response_speed_m_s"]
        delay = case["response_delay_s"]
        errors = []
        for tick in ticks:
            measured = float(np.linalg.norm(np.asarray(tick["ape_xy_m"]) - start))
            expected = min(.5, speed * max(0., tick["time_s"] - alarm - delay))
            errors.append(abs(measured - expected))
        comparison = {
            "name": case["name"],
            "independent_gap_matches_report": abs(
                audit["person_exact_nearest"]["gap_m"] -
                case["person_min_exact_gap_m"]) < 1e-8,
            "response_distance_trace_pass": max(errors) < .001,
            "maximum_distance_trace_error_m": max(errors),
            "upper_range_alarm": case["alarm_reason"] == "upper_owner_sector_motion",
            "mid_range_sector_fault_present": case["frozen_mid_range_packets"] > 0,
            "complete_retreat": abs(case["retreat_distance_at_cancel_m"] - .5) < .001,
            "video_and_packets_pass": all(audit[key] for key in (
                "video_sha_pass", "video_frames_pass", "depth_packet_sha_pass",
                "reacquire_packet_sha_pass")),
            "person_5cm_pass": audit["person_exact_nearest"]["gap_m"] >= .05,
        }
        rows.append(comparison)
    result = {"source_sha256": sha256(__file__), "report_sha256": sha256(REPORT),
              "checks": checks, "cases": rows,
              "all_evidence_checks_pass": all(checks.values()) and all(
                  all(row[key] for key in (
                      "independent_gap_matches_report", "response_distance_trace_pass",
                      "upper_range_alarm", "mid_range_sector_fault_present",
                      "complete_retreat", "video_and_packets_pass"))
                  for row in rows)}
    path = REPORT.with_name("audit.json")
    path.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps({"checks": checks,
                      "all_evidence_checks_pass": result["all_evidence_checks_pass"],
                      "cases": [{"name": row["name"],
                                 "person_5cm_pass": row["person_5cm_pass"],
                                 "maximum_distance_trace_error_m": row[
                                     "maximum_distance_trace_error_m"]}
                                for row in rows]}, indent=2))
    if not result["all_evidence_checks_pass"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
