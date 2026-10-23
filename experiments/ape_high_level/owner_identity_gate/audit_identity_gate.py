"""Audit the full-route anonymous-person gate against owner and bystander."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import imageio.v2 as imageio

ROOT = Path(__file__).resolve().parents[3]
REPORT = ROOT / "artifacts/ape-owner-identity-gate/v1/report.json"


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def media(ep):
    video = Path(ep["video"]["path"])
    reader = imageio.get_reader(video)
    frames = reader.count_frames()
    reader.close()
    packets = [row for row in ep["trace"] if row["stage"] ==
               "owner_person_rgbd_packets"]
    packets += [row["raw_packets"] for row in ep["trace"]
                if row["stage"] == "owner_post_retreat_view" and
                row.get("raw_packets")]
    packets += [row["depth_packets"] for row in ep["trace"]
                if row["stage"] == "owner_depth_change" and
                row.get("depth_packets")]
    return {"video_sha_pass": digest(video) == ep["video"]["sha256"],
            "frame_count_pass": frames == ep["video"]["frames"],
            "decoded_frames": frames,
            "packet_sha_pass": all(digest(row["path"]) == row["sha256"]
                                   for row in packets)}


def main():
    combined = json.loads(REPORT.read_text())
    pair_path = Path(combined["owner_bystander_report"]["path"])
    moving_path = Path(combined["moving_guard_report"]["path"])
    pair = json.loads(pair_path.read_text())
    moving = json.loads(moving_path.read_text())
    checks = {
        "combined_source_sha": digest(ROOT / "experiments/ape_high_level/owner_identity_gate/run_identity_gate.py") == combined["source_sha256"],
        "guarded_actor_sha": digest(REPORT.parent / "guarded_owner_turn_v597.py") == combined["guarded_turn_source_sha256"],
        "paired_report_sha": digest(pair_path) == combined["owner_bystander_report"]["sha256"],
        "moving_report_sha": digest(moving_path) == combined["moving_guard_report"]["sha256"],
        "two_plus_one_complete_episodes": len(pair["episodes"]) == 2 and
                                          len(moving["episodes"]) == 1,
        "same_route_before_substitution": len({json.dumps(ep["route_episode"]["trace"], sort_keys=True)
                                                for ep in pair["episodes"]}) == 1,
        "same_scene_before_substitution": len({ep["scene_fingerprint"]
                                                for ep in pair["episodes"]}) == 1,
        "all_source_hashes_unchanged": all(
            digest(ROOT / path) == expected
            for report in (pair, moving)
            for path, expected in report["source_hashes"].items()),
    }
    rows = []
    for ep in [*pair["episodes"], *moving["episodes"]]:
        stages = [row["stage"] for row in ep["trace"]]
        anonymous = [row for row in ep["trace"] if row["stage"] ==
                     "anonymous_person_rgbd_consistent"]
        alarm = [row for row in ep["trace"] if row["stage"] ==
                 "owner_reacquire_image_cancelled"]
        after_gate = (ep["trace"][ep["trace"].index(anonymous[0]) + 1:]
                      if anonymous else [])
        rows.append({"response": ep["response"], "outcome": ep["outcome"],
                     "media": media(ep),
                     "anonymous_observation_count": len(anonymous),
                     "old_owner_association_count": stages.count(
                         "owner_person_rgbd_association"),
                     "new_show_or_restage_after_gate": any(
                         row["stage"] in ("choose_pre_show_restage",
                                          "ask_owner_to_show_before_standoff",
                                          "standoff_reached_after_owner_show")
                         for row in after_gate),
                     "actor_action": anonymous[0]["actor_action"] if anonymous else None,
                     "motion_alarm": alarm[0]["alarm_reason"] if alarm else None})
    checks["paired_identity_gate"] = all(
        row["anonymous_observation_count"] == 1 and
        row["old_owner_association_count"] == 0 and
        not row["new_show_or_restage_after_gate"] and
        row["actor_action"] == "request_explicit_reintroduction_before_show" and
        row["outcome"] == "anonymous_person_waiting_for_reintroduction"
        for row in rows[:2])
    checks["moving_alarm_survives"] = (rows[2]["motion_alarm"] ==
        "upper_owner_sector_motion" and rows[2]["outcome"] ==
        "owner_interaction_reset_after_reacquire_motion" and
        rows[2]["old_owner_association_count"] == 0)
    checks["all_media_and_packets_pass"] = all(
        row["media"][key] for row in rows
        for key in ("video_sha_pass", "frame_count_pass", "packet_sha_pass"))
    result = {"source_sha256": digest(__file__),
              "combined_report_sha256": digest(REPORT),
              "checks": checks, "episodes": rows,
              "all_evidence_checks_pass": all(checks.values())}
    REPORT.with_name("audit.json").write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps({"checks": checks,
                      "all_evidence_checks_pass": result["all_evidence_checks_pass"],
                      "episodes": [{"response": x["response"], "outcome": x["outcome"]}
                                   for x in rows]}, indent=2))
    if not result["all_evidence_checks_pass"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
