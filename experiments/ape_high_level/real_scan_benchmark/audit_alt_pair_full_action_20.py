"""Independently score the frozen alternate-pair task continuation."""

from __future__ import annotations

import hashlib
import json
from collections import Counter
from pathlib import Path


ROOT = Path(__file__).resolve().parents[3]
ART = ROOT / "artifacts/ape-real-room-next-captures-20261014"
PRECOMMIT = ART / "alt_pair_full_action_20_precommit.json"
REPORT = ART / "alt_pair_full_action_20_report.json"
AUDIT = ART / "alt_pair_full_action_20_audit.json"
OUT = ROOT / "assets/real_indoor_scenes/alt_pair_full_action_20_20261015"


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def check_media(path: str | None) -> None:
    if path is None:
        return
    media = ROOT / path
    assert media.is_file() and media.stat().st_size > 1024, path


def main() -> None:
    frozen = json.loads(PRECOMMIT.read_text())
    for entry in frozen["policy_sources"]:
        assert digest(ROOT / entry["path"]) == entry["sha256"], entry["path"]
    assert digest(ROOT / frozen["source"]) == frozen["source_sha256"]
    assert digest(ROOT / frozen["help_audit"]) == frozen["help_audit_sha256"]
    report = json.loads(REPORT.read_text())
    cases = frozen["cases"]
    assert len(cases) == 20 and len({c["capture_group"] for c in cases}) == 20
    assert len(report["rows"]) == 35
    assert sum(c["branch"] == "earlier_task_failure" for c in cases) == 5
    scores = {arm: Counter() for arm in frozen["arms"]}
    rows = []
    for case in cases:
        key = case["scene_key"]
        if case["branch"] == "earlier_task_failure":
            rows.append({"scene_key": key, "capture_group": case["capture_group"],
                "branch": case["branch"], "status": case["source_status"]})
            continue
        for arm in frozen["arms"]:
            path = OUT / key / arm / "summary.json"
            row = json.loads(path.read_text())
            assert row["scene_key"] == key and row["capture_group"] == case["capture_group"]
            assert row["branch"] == case["branch"] and row["arm"] == arm
            packets = json.loads((path.parent / "independent_range_packets.json").read_text())[
                "packets"]
            assert packets and all(p["index"] == i for i, p in enumerate(packets))
            assert all(p["delivery_wall_s"] >= p["capture_wall_s"] for p in packets)
            assert all(a["capture_wall_s"] <= b["capture_wall_s"] for a, b in zip(
                packets, packets[1:]))
            assert all(p["source_health"] in ("healthy", "missing") for p in packets)
            fresh = json.loads((ROOT / row["fresh_trace"]).read_text())
            assert fresh["action"] == row["fresh_action"]
            check_media(fresh["video"])
            if row["motion_authorized"]:
                assert row["fresh_action"] == "owner_confirmed_visual_identity"
                assert row["association_gap_m"] <= .1 + 1e-8
                assert row["find_source_health"]["authorized"]
                assert row["walk_trace"] and row["walk_video"]
                walk = json.loads((ROOT / row["walk_trace"]).read_text())
                assert walk["result"] == row["walk_result"]
                for decision in walk["actions"]:
                    if decision["decision"] != "hold":
                        assert decision["source_guard"]["authorized"]
                check_media(row["walk_video"])
            else:
                assert row["walk_trace"] is None and row["final_trace"] is None
            if row["final_trace"]:
                final = json.loads((ROOT / row["final_trace"]).read_text())
                assert final["action"] == row["final_action_raw"]
                check_media(row["final_video"])
                if row["scripted_owner_final_yes"]:
                    assert final["action"] == "owner_confirmed_visual_identity"
                    assert row["final_selected_target_iou"] >= .25
                    assert row["final_source_health"]["authorized"]
            else:
                assert not row["scripted_owner_final_yes"]
            if arm == "range_gap":
                assert row["gap_stage"] in ("walk", "final") or not row[
                    "motion_authorized"]
                assert not row["scripted_owner_final_yes"]
                if row["gap_stage"]:
                    assert any(p["source_health"] == "missing" for p in packets)
            score = scores[arm]
            score["runs"] += 1
            score["fresh_confirmed"] += row["fresh_action"] == "owner_confirmed_visual_identity"
            score["motion_authorized"] += row["motion_authorized"]
            score["standoff"] += row["walk_result"] == "observed_target_standoff"
            score["final_correct"] += row["scripted_owner_final_yes"]
            score["source_gap_seen"] += any(p["source_health"] == "missing"
                for p in packets)
            score["safe_hold_or_unknown"] += not row["scripted_owner_final_yes"]
            score["final_wrong_claim"] += bool(row["final_action"] ==
                "owner_confirmed_visual_identity" and row["final_selected_target_iou"] < .25)
            rows.append({"scene_key": key, "capture_group": case["capture_group"],
                "branch": case["branch"], "arm": arm,
                "fresh_action": row["fresh_action"],
                "motion_authorized": row["motion_authorized"],
                "walk_result": row["walk_result"], "gap_stage": row["gap_stage"],
                "final_action": row["final_action"],
                "scripted_owner_final_yes": row["scripted_owner_final_yes"],
                "fresh_video": fresh["video"], "walk_video": row["walk_video"],
                "final_video": row["final_video"]})
    result = {"schema": "ape.alt-pair-full-action-audit.v1",
        "precommit": str(PRECOMMIT.relative_to(ROOT)),
        "report": str(REPORT.relative_to(ROOT)),
        "denominator": {"capture_groups": 20, "earlier_task_failures": 5,
            "direct_confirmed": 9, "accurate_help_confirmed": 6,
            "continued_confirmed": 15},
        "score": {arm: dict(score) for arm, score in scores.items()},
        "rows": rows,
        "limits": frozen["limits"]}
    AUDIT.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps({"denominator": result["denominator"],
        "score": result["score"]}, indent=2))


if __name__ == "__main__":
    main()
