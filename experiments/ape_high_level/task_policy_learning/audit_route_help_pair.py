"""Check state continuity and source ordering in the frozen route-help pair."""

from __future__ import annotations

import hashlib
import gzip
import json
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[3]
OUT = ROOT / "assets/real_indoor_scenes/learner_option_pilot_20261016"
ART = ROOT / "artifacts/ape-real-room-next-captures-20261014"
RUN = "alt_pair_route_help_clear_v4"
SCENES = ("capture_white_hart_east_1m", "capture_blue_south_1m")


def read(path: Path) -> dict:
    if path.is_file():
        return json.loads(path.read_text())
    with gzip.open(path.with_name(path.name + ".gz"), "rt") as stream:
        return json.load(stream)


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main() -> None:
    rows = []
    for scene in SCENES:
        folder = OUT / scene / RUN
        summary_path = folder / "summary.json"
        first_path = folder / "walk/trace.json"
        second_path = folder / "walk_after_route_clear/trace.json"
        packets_path = folder / "independent_range/packets.json"
        video_path = folder / "whole_episode_review.mp4"
        summary, first, second = (read(path) for path in (
            summary_path, first_path, second_path))
        packets = read(packets_path)["packets"]
        phases = [(e["phase"], e["executed"]) for e in summary["events"]]
        assert summary["fault"] == "blocked_route_walk"
        assert summary["owner_clears_route"]
        assert phases.count(("route_blocked", "ASK_OWNER")) == 1
        assert phases.index(("route_blocked", "ASK_OWNER")) < phases.index(
            ("skill_result", "FINAL_REVIEW"))
        assert summary["route_owner_reply"] == "scripted_person_cleared_route"
        assert summary["route_clear_source_health"]["authorized"]
        assert summary["route_relook_source_health"]["authorized"]
        assert summary["route_relook_action"] == \
            "owner_confirmed_visual_identity"
        assert first["result"] == "near_body_swept_clearance"
        assert second["result"] == "observed_target_standoff"
        assert np.allclose(first["actual_final_xz_m"],
                           second["actions"][0]["evaluator_actual_xz_m"])
        correction = second["actions"][0].get("range_pose_correction") or {}
        expected_actor = np.asarray(first["actor_final_xz_m"])
        if correction.get("accepted"):
            expected_actor += np.asarray(correction["correction_xy_m"])
        assert np.allclose(expected_actor,
                           second["actions"][0]["actor_xz_m"])
        assert any(p["stationary_person_xz_m"] is not None for p in packets)
        cleared = [p for p in packets if p["stationary_person_xz_m"] is None
                   and p["capture_wall_s"] < second["actions"][0][
                       "observation_wall_s"]]
        assert cleared and any(p["stationary_person_xz_m"] is not None
            for p in packets if p["capture_wall_s"] < cleared[-1][
                "capture_wall_s"])
        assert all(a.get("source_guard", {}).get("authorized") for a in
            second["actions"] if a["decision"] == "forward")
        assert summary["final_source_health"]["authorized"]
        assert (summary["result"] == "completed") == (
            summary["final_action_raw"] ==
            "owner_confirmed_visual_identity")
        assert video_path.is_file() and video_path.stat().st_size > 0
        rows.append({"scene_key": scene, "result": summary["result"],
            "first_walk": first["result"],
            "renewed_walk": second["result"],
            "renewed_forward_steps": second["accepted_forward_steps"],
            "final_action": summary["final_action_raw"],
            "final_source_healthy": summary["final_source_health"][
                "authorized"],
            "summary_sha256": digest(summary_path),
            "first_walk_sha256": digest(first_path),
            "renewed_walk_sha256": digest(second_path),
            "range_packets_sha256": digest(packets_path),
            "video": str(video_path.relative_to(ROOT)),
            "video_sha256": digest(video_path)})
    report = {"schema": "ape.learner-option-route-help-audit.v1",
        "run": RUN, "processed": len(rows),
        "completed": sum(r["result"] == "completed" for r in rows),
        "safe_unresolved": sum(r["result"] == "safe_unresolved" for r in rows),
        "limits": "Frozen policy in two development-exposed captured rooms. Owner clearing is scripted and immediate; body is a historical kinematic proxy with ideal sensing. The audit checks ordering and state continuity, not real collision avoidance or human response.",
        "rows": rows}
    path = ART / "learner_option_route_help_pair_audit.json"
    path.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps({key: report[key] for key in (
        "processed", "completed", "safe_unresolved")}, indent=2))


if __name__ == "__main__":
    main()
