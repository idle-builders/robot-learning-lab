"""Audit early teaching and one-point help as parts of the 20-room task."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[3]
ART = ROOT / "artifacts/ape-real-room-next-captures-20261014"
SOURCE_AUDIT = ART / "alt_pair_full_action_20_audit.json"
RETEACH = ART / "alt_pair_early_reteach_precommit.json"
RETEACH_REPORT = ART / "alt_pair_early_reteach_report.json"
HELP = ART / "alt_pair_reteach_help_precommit.json"
HELP_REPORT = ART / "alt_pair_reteach_help_report.json"
AUDIT = ART / "alt_pair_early_reteach_audit.json"
BASE = ROOT / "assets/real_indoor_scenes"
RETEACH_OUT = BASE / "alt_pair_early_reteach_20261015"
HELP_OUT = BASE / "alt_pair_reteach_help_20261015"
ACTION_A = BASE / "alt_pair_early_reteach_action_20261015"
ACTION_B = BASE / "alt_pair_reteach_help_action_20261015"


def read(path: Path) -> dict:
    return json.loads(path.read_text())


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def verify_precommit(path: Path) -> dict:
    x = read(path)
    for row in x["policy_sources"]:
        assert sha(ROOT / row["path"]) == row["sha256"], row["path"]
    assert sha(ROOT / x["source_precommit"]) == x["source_precommit_sha256"]
    assert sha(ROOT / x["source_report"]) == x["source_report_sha256"]
    return x


def media(path: str | None) -> None:
    if path:
        p = ROOT / path
        assert p.is_file() and p.stat().st_size > 1024, path


def action(path: Path, expected_arm: str) -> dict:
    x = read(path)
    assert x["arm"] == expected_arm
    assert x["fresh_trace"] and x["find_source_health"]["authorized"]
    assert x["motion_authorized"] and x["walk_trace"]
    fresh = read(ROOT / x["fresh_trace"])
    assert fresh["action"] == x["fresh_action"]
    media(fresh["video"])
    walk = read(ROOT / x["walk_trace"])
    assert walk["result"] == x["walk_result"]
    media(x["walk_video"])
    packets = read(path.parent / "independent_range_packets.json")["packets"]
    assert packets and all(p["index"] == i for i, p in enumerate(packets))
    assert all(p["delivery_wall_s"] >= p["capture_wall_s"] for p in packets)
    gap = any(p["source_health"] == "missing" for p in packets)
    if expected_arm == "healthy":
        assert not gap
        assert x["walk_result"] == "observed_target_standoff"
        assert x["scripted_owner_final_yes"]
        assert x["final_action"] == "owner_confirmed_visual_identity"
        assert x["final_selected_target_iou"] >= .25
    else:
        assert gap and not x["scripted_owner_final_yes"]
        assert x["final_action"] != "owner_confirmed_visual_identity"
    if x["final_trace"]:
        final = read(ROOT / x["final_trace"])
        media(final["video"])
    return {"arm": expected_arm, "fresh_action": x["fresh_action"],
        "walk_result": x["walk_result"], "final_action": x["final_action"],
        "final_correct": x["scripted_owner_final_yes"],
        "gap_seen": gap, "summary": str(path.relative_to(ROOT))}


def main() -> None:
    initial = read(SOURCE_AUDIT)
    assert initial["denominator"]["capture_groups"] == 20
    assert initial["score"]["healthy"]["final_correct"] == 15
    re = verify_precommit(RETEACH)
    hp = verify_precommit(HELP)
    rerows = read(RETEACH_REPORT)["rows"]
    hrows = read(HELP_REPORT)["rows"]
    assert len(re["cases"]) == len(rerows) == 3
    assert len(hp["cases"]) == len(hrows) == 2
    assert len({c["capture_group"] for c in re["cases"]}) == 3
    scores = {"early_source_cases": 3, "accurate_teaching_confirmed": 0,
        "wrong_teaching_confirmed": 0, "silent_teaching_confirmed": 0,
        "early_present_correct": 0, "early_absent_safe": 0,
        "help_eligible": 2, "help_accurate_confirmed": 0,
        "help_wrong_confirmed": 0, "help_silent_confirmed": 0,
        "help_wrong_absent_confirmed": 0, "new_healthy_final_correct": 0,
        "new_range_gap_final_claims": 0}
    rows = []
    for case in re["cases"]:
        key = case["scene_key"]
        arms = {arm: read(RETEACH_OUT / key / arm / "summary.json")
                for arm in re["owner_reply_arms"]}
        for arm, x in arms.items():
            assert x["scene_key"] == key and x["arm"] == arm
            if arm == "silent":
                assert x["owner_signal"] is None and x["present"] is None
                assert not x["teach_confirmed"]
            else:
                signal = x["owner_signal"]
                media(signal["rgb"])
                assert (ROOT / signal["depth"]).is_file()
                teach = read((RETEACH_OUT / key / "teach/trace.json") if
                    arm == "accurate" else (RETEACH_OUT / key / arm / "teach/trace.json"))
                assert (teach["evaluator_selected_target_iou"] >= .25) == x[
                    "teach_confirmed"]
            if not x["teach_confirmed"]:
                assert x["present"] is None and x["absent"] is None
            else:
                for condition in ("present", "absent"):
                    assert x[condition]
                    episode = read(ROOT / x[condition]["trace"])
                    assert episode["action"] == x[condition]["action"]
                    media(x[condition]["video"])
        scores["accurate_teaching_confirmed"] += arms["accurate"]["teach_confirmed"]
        scores["wrong_teaching_confirmed"] += arms["wrong"]["teach_confirmed"]
        scores["silent_teaching_confirmed"] += arms["silent"]["teach_confirmed"]
        scores["early_present_correct"] += (arms["accurate"]["present"] or {}).get(
            "action") == "owner_confirmed_visual_identity"
        scores["early_absent_safe"] += (arms["accurate"]["absent"] or {}).get(
            "action") == "unidentified_ask_new_view_or_help"
        paths = [ACTION_A / key / arm / "summary.json" for arm in ("healthy", "range_gap")]
        actions = ([action(path, arm) for path, arm in zip(paths,
                    ("healthy", "range_gap"))]
                   if all(p.exists() for p in paths) else [])
        scores["new_healthy_final_correct"] += bool(actions and actions[0]["final_correct"])
        scores["new_range_gap_final_claims"] += bool(actions and actions[1]["final_correct"])
        rows.append({"scene_key": key, "capture_group": case["capture_group"],
            "source_status": case["source_status"],
            "accurate_teach": arms["accurate"]["teach_confirmed"],
            "wrong_teach": arms["wrong"]["teach_confirmed"],
            "present": (arms["accurate"]["present"] or {}).get("action"),
            "absent": (arms["accurate"]["absent"] or {}).get("action"),
            "actions": actions})
    helped = []
    for case in hp["cases"]:
        key = case["scene_key"]
        arms = {arm: read(HELP_OUT / key / arm / "trace.json")
                for arm in hp["owner_reply_arms"]}
        for arm, x in arms.items():
            assert x["scene_key"] == key
            if x.get("fresh_search"):
                episode = read(ROOT / x["fresh_search"])
                assert episode["action"] == x["action"]
                media(episode["video"])
        scores["help_accurate_confirmed"] += arms["accurate_present"]["action"] == \
            "owner_confirmed_visual_identity"
        scores["help_wrong_confirmed"] += arms["wrong_present"]["action"] == \
            "owner_confirmed_visual_identity"
        scores["help_silent_confirmed"] += arms["silent_present"]["action"] == \
            "owner_confirmed_visual_identity"
        scores["help_wrong_absent_confirmed"] += arms["wrong_absent"]["action"] == \
            "owner_confirmed_visual_identity"
        paths = [ACTION_B / key / arm / "summary.json" for arm in ("healthy", "range_gap")]
        actions = ([action(path, arm) for path, arm in zip(paths,
                    ("healthy", "range_gap"))]
                   if all(p.exists() for p in paths) else [])
        scores["new_healthy_final_correct"] += bool(actions and actions[0]["final_correct"])
        scores["new_range_gap_final_claims"] += bool(actions and actions[1]["final_correct"])
        helped.append({"scene_key": key, "capture_group": case["capture_group"],
            "accurate_action": arms["accurate_present"]["action"],
            "wrong_action": arms["wrong_present"]["action"],
            "silent_action": arms["silent_present"]["action"],
            "wrong_absent_action": arms["wrong_absent"]["action"],
            "actions": actions})
    assert scores["new_range_gap_final_claims"] == 0
    result = {"schema": "ape.alt-pair-early-reteach-audit.v1",
        "early_precommit": str(RETEACH.relative_to(ROOT)),
        "help_precommit": str(HELP.relative_to(ROOT)),
        "source_full_action_audit": str(SOURCE_AUDIT.relative_to(ROOT)),
        "score": scores,
        "joined_20_capture_state_continuous_correct": 15 + scores[
            "new_healthy_final_correct"],
        "early_rows": rows, "help_rows": helped,
        "limits": "Joined count sums saved-state continuations, not one uninterrupted fresh 20-room policy rollout. Previously exposed rooms, scripted owner, static people/items, ideal camera/depth/range and unselected kinematic body; source placements are provisional."}
    AUDIT.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps({"score": scores,
        "joined_20_capture_state_continuous_correct": result[
            "joined_20_capture_state_continuous_correct"]}, indent=2))


if __name__ == "__main__":
    main()
