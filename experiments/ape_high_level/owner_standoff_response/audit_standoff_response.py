"""Independently replay geometry and check the retained standoff episodes."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / "experiments/ape_high_level/closed_loop"))
import audit_robocasa_first_return_person_reacquire_v578 as geometry_audit

REPORT = ROOT / "artifacts/ape-owner-standoff-response/v2/report.json"


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def candidate_depth_views(case, episode):
    """Diagnostic only: inspect saved pixels, without changing actor decisions."""
    path = REPORT.parent / f"{case['name']}-depth-packets-person-reacquire.npz"
    if not path.exists():
        return []
    observations = [row["actor_observation"] for row in episode["trace"]
                    if row["stage"] == "owner_person_rgbd_reacquire"]
    views = []
    with np.load(path) as packets:
        for index, observation in enumerate(observations):
            depth = packets[f"depth_{index}"]
            proposals = []
            for detection in observation["detections"]:
                x1, y1, x2, y2 = (int(v) for v in detection["box_xyxy"])
                box = depth[max(0, y1):min(depth.shape[0], y2),
                            max(0, x1):min(depth.shape[1], x2)]
                if not box.size:
                    continue
                valid = np.isfinite(box) & (box >= .15) & (box <= 2.8)
                proposals.append({"score": detection["score"],
                                  "local_valid_depth_fraction": float(valid.mean())})
            views.append({"actor_status": observation["status"],
                          "central_valid_depth_fraction": observation[
                              "central_valid_fraction"],
                          "person_proposals": proposals})
    return views


def main():
    report = json.loads(REPORT.read_text())
    cases, episodes = report["cases"], report["episodes"]
    checks = {
        "seven_complete_cases": len(cases) == len(episodes) == 7,
        "matched_scene": len({ep["scene_fingerprint"] for ep in episodes}) == 1,
        "source_hashes_unchanged": all(digest(ROOT / path) == expected
                                       for path, expected in report["source_hashes"].items()),
        "same_route_state": len({json.dumps(ep["route_state"], sort_keys=True)
                                  for ep in episodes}) == 1,
    }
    rows = []
    for case, ep in zip(cases, episodes):
        audit = geometry_audit.audit_episode(ep)
        audit.pop("geometry_samples", None)
        candidate = next(row["chosen"] for row in ep["trace"]
                         if row["stage"] == "choose_pre_show_standoff")
        guard = [row for row in ep["trace"] if row["stage"] ==
                 "owner_reacquire_guard_tick"]
        alarm = case["alarm_time_s"]
        distances = []
        if guard and alarm is not None:
            start = np.asarray(guard[0]["ape_xy_m"])
            for row in guard:
                observed = float(np.linalg.norm(np.asarray(row["ape_xy_m"]) - start))
                expected = min(.5, .25 * max(0., row["time_s"] - alarm -
                                               case["response_delay_s"]))
                distances.append(abs(observed - expected))
        row = {
            "name": case["name"],
            "selected_known_free_path": candidate is not None and candidate["known_free_path"],
            "selected_meets_requested_reserve": candidate is not None and
                candidate["reserve_m"] >= case["required_reserve_m"],
            "selected_distance_matches": candidate is not None and
                candidate["distance_m"] == case["chosen_distance_m"],
            "independent_exact_gap_matches": abs(
                audit["person_exact_nearest"]["gap_m"] -
                case["minimum_exact_person_gap_m"]) < 1e-8,
            "response_law_matches": not distances or max(distances) < .001,
            "maximum_response_error_m": max(distances) if distances else None,
            "video_and_packets_pass": all(audit[key] for key in (
                "video_sha_pass", "video_frames_pass", "depth_packet_sha_pass",
                "reacquire_packet_sha_pass")),
            "frozen_mid_sector_present": case["frozen_mid_sector_packets"] > 0,
            "proxy_five_cm_gap_pass": audit["person_exact_nearest"]["gap_m"] >= .05,
            "outcome": ep["outcome"],
            "reacquire_candidate_depth_diagnostic": candidate_depth_views(case, ep),
        }
        rows.append(row)
    evidence_keys = ("selected_known_free_path", "selected_meets_requested_reserve",
                     "selected_distance_matches", "independent_exact_gap_matches",
                     "response_law_matches", "video_and_packets_pass",
                     "frozen_mid_sector_present")
    result = {
        "source_sha256": digest(__file__), "report_sha256": digest(REPORT),
        "checks": checks, "cases": rows,
        "all_evidence_checks_pass": all(checks.values()) and all(
            all(row[key] for key in evidence_keys) for row in rows),
    }
    path = REPORT.with_name("audit.json")
    path.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps({"checks": checks,
                      "all_evidence_checks_pass": result["all_evidence_checks_pass"],
                      "cases": [{"name": row["name"],
                                 "proxy_five_cm_gap_pass": row["proxy_five_cm_gap_pass"],
                                 "outcome": row["outcome"]} for row in rows]}, indent=2))
    if not result["all_evidence_checks_pass"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
