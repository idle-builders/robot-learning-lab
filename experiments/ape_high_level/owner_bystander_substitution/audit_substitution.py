"""Check the full bystander-substitution episodes and actor identity claim."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import imageio.v2 as imageio
import numpy as np

ROOT = Path(__file__).resolve().parents[3]
REPORT = ROOT / "artifacts/ape-owner-bystander-substitution/v3/report.json"


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def media(ep):
    video = Path(ep["video"]["path"])
    reader = imageio.get_reader(video)
    frames = reader.count_frames()
    reader.close()
    packet_rows = [row for row in ep["trace"] if row["stage"] ==
                   "owner_person_rgbd_packets"]
    packet_rows += [row["raw_packets"] for row in ep["trace"]
                    if row["stage"] == "owner_post_retreat_view" and
                    row.get("raw_packets")]
    packet_rows += [row["depth_packets"] for row in ep["trace"]
                    if row["stage"] == "owner_depth_change" and
                    row.get("depth_packets")]
    return {"video_sha_pass": digest(video) == ep["video"]["sha256"],
            "decoded_frames": frames,
            "frame_count_pass": frames == ep["video"]["frames"],
            "packet_sha_pass": all(digest(row["path"]) == row["sha256"]
                                   for row in packet_rows),
            "packet_count": len(packet_rows)}


def main():
    report = json.loads(REPORT.read_text())
    cases, episodes = report["cases"], report["episodes"]
    keyed = {case["name"]: (case, ep) for case, ep in zip(cases, episodes)}
    checks = {
        "two_complete_cases": len(keyed) == len(episodes) == 2,
        "expected_matrix": set(keyed) == {"owner_present_body",
                                           "bystander_substitute_body"},
        "same_route_state": len({json.dumps(ep["route_state"], sort_keys=True)
                                  for ep in episodes}) == 1,
        "identical_route_trace": len({json.dumps(ep["route_episode"]["trace"],
                                                 sort_keys=True)
                                      for ep in episodes}) == 1,
        "same_scene_before_substitution": len({ep["scene_fingerprint"]
                                                for ep in episodes}) == 1,
        "source_hashes_unchanged": all(digest(ROOT / path) == expected
                                       for path, expected in report["source_hashes"].items()),
        "substitution_fixture_logged": bool(report["evaluator_substitution"]),
        "both_choose_known_free_60cm": all(case["chosen_distance_m"] == .6 and
                                            case["chosen_path_known_free"]
                                            for case in cases),
    }
    rows = []
    bystander_xy = np.asarray(report["evaluator_substitution"][0][
        "bystander_torso_xy_evaluator_m"]) if report["evaluator_substitution"] else None
    for case, ep in zip(cases, episodes):
        associations = [item for item in ep["trace"] if item["stage"] ==
                        "owner_person_rgbd_association"]
        person_views = [item for item in ep["trace"] if item["stage"] ==
                        "owner_person_rgbd_reacquire"]
        selected_points = [np.asarray(item["actor_observation"]["point_xyz_est_m"])
                           for item in person_views if item["actor_observation"].get(
                               "point_xyz_est_m") is not None]
        claim_distance = None
        if bystander_xy is not None and associations:
            claim_distance = float(np.linalg.norm(np.asarray(
                associations[0]["new_owner_surface_xy_est_m"]) - bystander_xy))
        rows.append({
            "name": case["name"], "outcome": ep["outcome"],
            "media": media(ep),
            "person_view_statuses": [item["actor_observation"]["status"]
                                     for item in person_views],
            "two_view_point_distance_m": (float(np.linalg.norm(
                selected_points[0] - selected_points[1]))
                                          if len(selected_points) == 2 else None),
            "actor_owner_association_count": len(associations),
            "association_distance_to_bystander_torso_m_evaluator_only": claim_distance,
            "false_old_owner_association_evaluator": (
                case["name"] == "bystander_substitute_body" and
                claim_distance is not None and claim_distance < .4),
            "owner_association_rows": associations,
            "selected_proposals": report["person_proposal_decisions"].get(
                ep["response"], []),
            "candidate_depth_decisions": report["candidate_depth_decisions"].get(
                ep["response"], []),
        })
    checks["media_and_packets_pass"] = all(row["media"][key] for row in rows
                                          for key in ("video_sha_pass",
                                                      "frame_count_pass",
                                                      "packet_sha_pass"))
    result = {
        "source_sha256": digest(__file__), "report_sha256": digest(REPORT),
        "checks": checks, "cases": rows,
        "geometry_limit": (
            "The imported V578 owner-geometry summary embedded in report.json "
            "does not reconstruct evaluator replacement bystander geoms. Its "
            "person minimum for bystander_substitute_body is invalid and must "
            "not be used as a clearance claim. This audit verifies identity, "
            "route equality, raw packets and complete media only."),
        "all_evidence_checks_pass": all(checks.values()),
    }
    REPORT.with_name("audit.json").write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps({"checks": checks,
                      "all_evidence_checks_pass": result["all_evidence_checks_pass"],
                      "cases": [{"name": row["name"], "outcome": row["outcome"],
                                 "association_count": row["actor_owner_association_count"],
                                 "distance_to_bystander_m": row[
                                     "association_distance_to_bystander_torso_m_evaluator_only"]}
                                for row in rows]}, indent=2))
    if not result["all_evidence_checks_pass"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
