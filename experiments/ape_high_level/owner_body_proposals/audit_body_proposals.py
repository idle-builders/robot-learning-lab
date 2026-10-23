"""Audit full-body proposal owner turns and saved media."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / "experiments/ape_high_level/closed_loop"))
import audit_robocasa_first_return_person_reacquire_v578 as geometry_audit

REPORT = ROOT / "artifacts/ape-owner-body-proposals/v1/report.json"


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def main():
    report = json.loads(REPORT.read_text())
    cases, episodes = report["cases"], report["episodes"]
    controls = {case["name"]: (case, ep) for case, ep in zip(cases, episodes)}
    checks = {
        "four_complete_cases": len(cases) == len(episodes) == 4,
        "expected_matrix": set(controls) == {"stationary_frame_control",
                                             "stationary_local_only",
                                             "stationary_body", "moving_body"},
        "same_room_and_route": (len({ep["scene_fingerprint"] for ep in episodes}) == 1
                                and len({json.dumps(ep["route_state"], sort_keys=True)
                                         for ep in episodes}) == 1),
        "source_hashes_unchanged": all(digest(ROOT / path) == expected
                                       for path, expected in report["source_hashes"].items()),
        "same_map_chosen_standoff": all(case["chosen_distance_m"] == .6 and
                                        case["chosen_path_known_free"]
                                        for case in cases),
    }
    rows = []
    for case, ep in zip(cases, episodes):
        audit = geometry_audit.audit_episode(ep)
        audit.pop("geometry_samples", None)
        decisions = report["candidate_depth_decisions"].get(ep["response"], [])
        proposals = report["person_proposal_decisions"].get(ep["response"], [])
        points = [np.asarray(item["actor_observation"]["point_xyz_est_m"])
                  for item in ep["trace"] if item["stage"] ==
                  "owner_person_rgbd_reacquire" and item["actor_observation"].get(
                      "point_xyz_est_m") is not None]
        row = {
            "name": case["name"], "outcome": ep["outcome"],
            "minimum_exact_person_gap_m": audit["person_exact_nearest"]["gap_m"],
            "independent_geometry_matches": abs(
                audit["person_exact_nearest"]["gap_m"] -
                case["minimum_exact_person_gap_m"]) < 1e-8,
            "video_and_packets_pass": all(audit[key] for key in (
                "video_sha_pass", "video_frames_pass", "depth_packet_sha_pass",
                "reacquire_packet_sha_pass")),
            "candidate_decisions": decisions,
            "proposal_decisions": proposals,
            "two_view_point_distance_m": (float(np.linalg.norm(points[0] - points[1]))
                                          if len(points) == 2 else None),
            "motion_alarm": case["alarm_reason"],
            "metric_find_point_error_m": ep["depth_point_error_m"],
        }
        rows.append(row)
    checks["control_has_no_overlay"] = not next(
        row for row in rows if row["name"] == "stationary_frame_control")[
            "candidate_decisions"]
    checks["local_depth_override_exercised"] = all(any(
        decision["local_override"] for decision in row["candidate_decisions"])
        for row in rows if row["name"] in ("stationary_local_only", "stationary_body"))
    body = next(row for row in rows if row["name"] == "stationary_body")
    checks["body_proposal_reordered"] = bool(body["proposal_decisions"] and
        body["proposal_decisions"][0]["detector_boxes"] and
        body["proposal_decisions"][0]["selected_box_xyxy"] !=
        body["proposal_decisions"][0]["detector_boxes"][0]["box_xyxy"])
    checks["moving_case_still_interrupts"] = all(
        row["motion_alarm"] == "upper_owner_sector_motion" and
        row["outcome"] == "owner_interaction_reset_after_reacquire_motion"
        for row in rows if row["name"] == "moving_body")
    result = {
        "source_sha256": digest(__file__), "report_sha256": digest(REPORT),
        "checks": checks, "cases": rows,
        "all_evidence_checks_pass": all(checks.values()) and all(
            row["independent_geometry_matches"] and row["video_and_packets_pass"]
            for row in rows),
    }
    REPORT.with_name("audit.json").write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps({"checks": checks,
                      "all_evidence_checks_pass": result["all_evidence_checks_pass"],
                      "cases": [{"name": row["name"], "outcome": row["outcome"],
                                 "gap_cm": round(100 * row[
                                     "minimum_exact_person_gap_m"], 2)}
                                for row in rows]}, indent=2))
    if not result["all_evidence_checks_pass"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
