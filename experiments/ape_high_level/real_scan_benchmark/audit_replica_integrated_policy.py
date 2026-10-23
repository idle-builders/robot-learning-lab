"""Audit one frozen direct-plus-route policy across Replica task arms."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

from audit_replica_full_monitor import check_visual_stage, video_frames

ROOT = Path(__file__).resolve().parents[3]
RUNS = ROOT / "assets/real_indoor_scenes/replica_owner_integrated_20261015"
REPORT = ROOT / "artifacts/ape-replica-interactive-20261015/integrated_route_policy_audit.json"
REFERENCE = ROOT / "artifacts/ape-replica-interactive-20261015/monitored_reference_hashes.json"
ARMS = {"hotel_0": ("none",), "apartment_0": ("none",),
        "office_3": ("none", "walk", "route", "final")}


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def check_arm(room: str, arm: str, reference: dict) -> dict:
    folder = RUNS / room / arm
    summary = json.loads((folder / "summary.json").read_text())
    scenario = json.loads((folder / "scenario.json").read_text())
    walk = json.loads((folder / "walk/trace.json").read_text())
    source = json.loads((folder / "independent_range/packets.json").read_text())
    packets = source["packets"]
    assert summary["scene"] == scenario["scene"] == room
    assert summary["range_gap_stage"] == scenario["range_gap_stage"] == arm
    assert summary["route_fallback"] and scenario["route_fallback"]
    assert summary["present_action"] == "candidate_to_owner_for_confirmation"
    assert summary["absent_action"] == "report_unidentified"
    assert not summary["present_wrong_confident"] and not summary[
        "absent_wrong_confident"]
    assert summary["find_source_health"]["authorized"]
    stages = {stage: check_visual_stage(folder / stage) for stage in
              ("find_present", "find_absent")}
    assert video_frames(folder / "walk/first_person.mp4") == len(walk["actions"])
    assert len(walk["actions"]) <= walk["action_budget"] + 1
    observation_times = [a["observation_wall_s"] for a in walk["actions"]]
    assert all(b > a for a, b in zip(observation_times, observation_times[1:]))
    assert all(a["source_guard"]["checked_wall_s"] >=
               a["observation_wall_s"] for a in walk["actions"]
               if "source_guard" in a)
    assert len(list((folder / "rgb_requests").glob("frame_*.jpg"))) == summary[
        "rgb_requests"]
    assert [p["index"] for p in packets] == list(range(len(packets)))
    assert all(p["delivery_wall_s"] >= p["capture_wall_s"] for p in packets)
    accepted = [a for a in walk["actions"] if a["decision"] == "forward" and
                a.get("evaluator_motion_accepted")]
    assert len(accepted) == summary["walk_accepted_forward_steps"]
    assert all(a["source_guard"]["authorized"] and
               a["source_guard"]["last_packet_age_s"] <= .2 and
               0 < a["amount"] <= .04 for a in accepted)
    route = [a for a in walk["actions"] if a.get("route_decision") is not None]
    assert bool(route) == summary["route_fallback_used"] == walk[
        "route_fallback_used"]
    final = None
    if summary["walk_result"] == "observed_target_standoff":
        final = check_visual_stage(folder / "final_view")
        assert final["action"] == summary["final_action_raw"] == \
            "candidate_to_owner_for_confirmation"
        assert final["selected_target_iou"] >= .25
    for name in ("owner/owner_rgb.png", "teach/teach_rgb.png"):
        assert digest(folder / name) == reference[room][name]
    for stage in ("find_present", "find_absent"):
        assert [digest(folder / stage / f"rgb_{i:03d}.png")
                for i in range(stages[stage]["frames"])] == reference[room][stage]
    missing = [p for p in packets if p["source_health"] == "missing"]
    stop_after_gap_s = None
    if arm in ("walk", "route"):
        assert summary["walk_result"] == "source_gap"
        assert summary["final_action"] is None
        assert walk["actions"][-1]["source_guard"]["reason"] == "source_gap"
        assert missing
        stop_after_gap_s = (walk["actions"][-1]["source_guard"][
            "checked_wall_s"] - missing[0]["capture_wall_s"])
        assert 0 <= stop_after_gap_s <= .2
        if arm == "walk":
            assert len(accepted) == 3 and not route
        else:
            assert sum(a["decision"] == "forward" and a.get(
                "evaluator_motion_accepted") and a.get("route_decision")
                is not None for a in walk["actions"]) == 3
    else:
        assert summary["walk_result"] == "observed_target_standoff"
        assert summary["walk_source_health"]["authorized"]
        assert final is not None
        if arm == "final":
            assert summary["final_action"] == "withheld_range_source_unhealthy"
            assert summary["final_source_health"]["source_gap_seen"]
            assert missing
        else:
            assert summary["final_action"] == final["action"]
            assert summary["final_source_health"]["authorized"]
            assert not missing
    return {"arm": arm, "walk_result": summary["walk_result"],
        "accepted_forward_steps": len(accepted), "route_fallback_used": bool(route),
        "route_actions": len(route),
        "rejected_candidate_steps": sum(a.get("evaluator_motion_accepted") is False
                                        for a in walk["actions"]),
        "observation_span_wall_s": observation_times[-1] - observation_times[0],
        "final_visual_action": final["action"] if final else None,
        "effective_final_action": summary["final_action"],
        "evaluator_final_target_gap_m": summary["evaluator_final_target_gap_m"],
        "source_packets": len(packets), "missing_packets": len(missing),
        "stop_after_first_missing_s": stop_after_gap_s,
        "visual_frames": {key: value["frames"] for key, value in stages.items()},
        "source_media_audited": True}


def main() -> None:
    reference = json.loads(REFERENCE.read_text())["rooms"]
    rows = {}
    for room, arms in ARMS.items():
        rows[room] = {arm: check_arm(room, arm, reference) for arm in arms}
    assert all(not rows[r]["none"]["route_fallback_used"] for r in
               ("hotel_0", "apartment_0"))
    office = rows["office_3"]
    assert office["none"]["route_fallback_used"]
    assert office["final"]["route_fallback_used"]
    healthy = RUNS / "office_3/none"
    final_fault = RUNS / "office_3/final"
    assert all(digest(healthy / "final_view" / f"rgb_{i:03d}.png") ==
               digest(final_fault / "final_view" / f"rgb_{i:03d}.png")
               for i in range(check_visual_stage(healthy / "final_view")["frames"]))
    report = {"schema": "ape.replica-integrated-route-policy-audit.v1",
        "policy": "One direct-approach policy with a bounded two-plane known-free grid-route fallback on blocked forward motion",
        "visual_reference_hashes": str(REFERENCE),
        "rooms": rows, "room_count": len(rows),
        "office_healthy_and_final_gap_rgb_identical": True,
        "conclusion": "One integrated policy keeps hotel/apartment healthy completions, reaches and visually rechecks the office item after a blocked direct route, stops on source loss during direct or route motion, and withholds a visually identical final offer on source loss during final inference.",
        "limits": "Three exposed photographic rooms, synthetic source gaps, ideal RGB-aligned depth/range and command pose, static-room-only independent monitor, scripted owner and 17cm disk; no moving-person or selected-body qualification, capture-disjoint transfer, deployable sensor comparison or learned whole-task policy."}
    REPORT.parent.mkdir(parents=True, exist_ok=True)
    REPORT.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps({"room_count": len(rows), "office": office}, indent=2))


if __name__ == "__main__":
    main()
