"""Audit the frozen V4 whole-task breadth and paired recovery sweeps."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
OUT = ROOT / "assets/real_indoor_scenes/learner_option_pilot_20261016"
ART = ROOT / "artifacts/ape-real-room-next-captures-20261014"


def read(path: Path) -> dict:
    return json.loads(path.read_text())


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def audited_rows(name: str) -> tuple[dict, dict[str, dict]]:
    report = read(OUT / f"{name}_sweep.json")
    freeze = read(ROOT / report["precommit"])
    assert report["precommit_sha256"] == digest(ROOT / report["precommit"])
    assert report["policy_sha256"] == freeze["policy_sha256"]
    assert report["processed"] == report["total"] == len(freeze["scenes"])
    cases = {r["scene_key"]: r for r in report["rows"]}
    assert len(cases) == len(freeze["scenes"])
    assert set(cases) == {r["scene_key"] for r in freeze["scenes"]}
    assert len({r["capture_group"] for r in freeze["scenes"]}) == len(cases)
    for scene, row in cases.items():
        episode = read(ROOT / row["summary"])
        assert episode["scene_key"] == scene
        assert row["wrong_candidate_questions"] in (None, 0)
        assert episode["result"] == row["result"]
        for event in episode["events"]:
            if event["phase"] == "pre_motion" and event["executed"] == \
                    "APPROACH":
                assert all(event["observation"][key] for key in (
                    "present_confirmed", "fresh_confirmed",
                    "source_healthy", "pose_healthy", "association_healthy"))
        if row["result"] == "completed":
            assert episode["final_action_raw"] == \
                "owner_confirmed_visual_identity"
            assert episode["final_source_health"]["authorized"]
            assert any(e["phase"] == "final_review" and
                       e["executed"] == "COMPLETE" for e in episode["events"])
        assert row["archive_manifest_sha256"]
        if row["video"]:
            assert (ROOT / row["video"]).is_file()
        else:
            assert not (ROOT / row["archived_video"]).exists()
    return report, cases


def main() -> None:
    development, dev = audited_rows("v4_breadth20_standard")
    control, before = audited_rows("v4_prior_transfer10_standard")
    recovery, after = audited_rows("v4_prior_transfer10_recovery")
    assert development["policy_sha256"] == control["policy_sha256"] == \
        recovery["policy_sha256"]
    assert not set(dev).intersection(before)
    assert set(before) == set(after)
    assert len(set(dev) | set(after)) == 30
    assert development["completed"] == 18
    assert control["completed"] == 5
    assert recovery["completed"] == 7
    improvements = [scene for scene in before if before[scene]["result"] !=
                    "completed" and after[scene]["result"] == "completed"]
    regressions = [scene for scene in before if before[scene]["result"] ==
                   "completed" and after[scene]["result"] != "completed"]
    assert set(improvements) == {
        "capture_cozy_apartment_night_east_0p8m",
        "capture_upper_vestibule_south_0p8m"}
    assert not regressions
    helped = []
    for source_name, scene in (("v4_breadth20_standard",
                                "capture_nomad_east_1m"),
                               ("v4_prior_transfer10_recovery",
                                "capture_morning_geometry_west_0p8m"),
                               ("v4_prior_transfer10_recovery",
                                "capture_dining_north_1m"),
                               ("v4_prior_transfer10_recovery",
                                "capture_billiards_east_0p8m")):
        destination = ("v4_breadth20_owner_reshow" if source_name ==
                       "v4_breadth20_standard" else
                       "v4_prior_transfer10_owner_reshow")
        folder = OUT / destination / scene
        continuation = read(folder / "learner_summary.json")
        assert continuation["source_episode"].endswith(
            f"{scene}/{source_name}/summary.json")
        assert continuation["authorized"] == "ASK_RESHOW"
        assert continuation["safe_visible_owner"]
        assert continuation["result"] == "completed"
        assert read(folder / "summary.json")["action"]["final_action"] == \
            "owner_confirmed_visual_identity"
        assert (folder / "whole_episode_review.mp4").is_file()
        helped.append({"scene_key": scene, "source_run": source_name,
                       "video": str((folder / "whole_episode_review.mp4").relative_to(ROOT))})
    assert dev["capture_nazaretian_inner_east_0p8m"]["result"] == \
        "safe_unresolved"
    result = {"schema": "ape.learner-option-v4-breadth-audit.v1",
        "policy_sha256": development["policy_sha256"],
        "distinct_capture_groups": 30,
        "development_direct": {"completed": 18, "attempted": 20},
        "prior_transfer_direct": {"completed": 5, "attempted": 10},
        "prior_transfer_generic_recovery": {"completed": 7,
            "attempted": 10, "improved_scenes": improvements,
            "regressed_scenes": regressions},
        "scripted_owner_reshow_completed": len(helped),
        "assisted_whole_task_completed": 29,
        "whole_task_attempts": 30,
        "safe_unresolved": ["capture_nazaretian_inner_east_0p8m"],
        "wrong_confident_candidate_questions": 0,
        "assisted_continuations": helped,
        "local_development_videos": sum(bool(r["video"]) for r in dev.values()),
        "local_transfer_recovery_videos": sum(bool(r["video"]) for r in after.values()),
        "limits": "The 20+10 capture groups are development-exposed static reconstructed rooms, not appearance-qualified photoreal or a blind test. The owner and response are scripted; initial owner approach, teaching, search, reshow execution and bounded motion remain authored. The learned policy selects post-teaching options and the owner request. Ideal RGB-D, yaw and independent 360-degree range plus rejected historical motion proxy do not qualify physical sensors or the current 50 cm body. No human animation, stop-tail physics, joint/contact/force integration or unseen live person behavior is established."}
    output = ART / "learner_option_v4_breadth30_audit.json"
    output.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps({k: result[k] for k in (
        "distinct_capture_groups", "development_direct",
        "prior_transfer_direct", "prior_transfer_generic_recovery",
        "assisted_whole_task_completed", "safe_unresolved")}, indent=2))


if __name__ == "__main__":
    main()
