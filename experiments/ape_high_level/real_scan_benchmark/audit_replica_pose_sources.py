"""Audit historical motion-shaped pose packets in full Replica owner tasks."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np

from audit_replica_full_monitor import check_visual_stage, video_frames

ROOT = Path(__file__).resolve().parents[3]
ART = ROOT / "artifacts/ape-replica-interactive-20261015"
RUNS = ROOT / "assets/real_indoor_scenes/replica_owner_pose_sources_20261015"
PRECOMMIT = ART / "pose_source_screen_precommit.json"
REFERENCE = ART / "monitored_reference_hashes.json"
REPORT = ART / "pose_source_screen_audit.json"


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def check_arm(room: str, source: dict, reference: dict) -> dict:
    name = source["name"]
    folder = RUNS / room / name
    summary = json.loads((folder / "summary.json").read_text())
    scenario = json.loads((folder / "scenario.json").read_text())
    walk = json.loads((folder / "walk/trace.json").read_text())
    stream = json.loads((folder / "independent_range/packets.json").read_text())
    actions = walk["actions"]
    packets = stream["packets"]
    assert summary["scene"] == scenario["scene"] == room
    assert summary["pose_source"] == scenario["pose_source"] == name
    assert summary["pose_source_trace_sha256"] == source["sha256"]
    assert summary["route_fallback"] and scenario["route_fallback"]
    assert summary["range_gap_stage"] == "none"
    assert summary["owner_intent"] == "hold_near_person"
    assert summary["teaching_confirmed"]
    assert summary["present_action"] == "candidate_to_owner_for_confirmation"
    assert summary["absent_action"] == "report_unidentified"
    assert not summary["present_wrong_confident"] and not summary[
        "absent_wrong_confident"]
    stages = {stage: check_visual_stage(folder / stage) for stage in
              ("find_present", "find_absent")}
    for item in ("owner/owner_rgb.png", "teach/teach_rgb.png"):
        assert sha(folder / item) == reference[room][item]
    for stage, result in stages.items():
        assert [sha(folder / stage / f"rgb_{i:03d}.png")
                for i in range(result["frames"])] == reference[room][stage]
    assert video_frames(folder / "walk/first_person.mp4") == len(actions)
    assert len(actions) <= walk["action_budget"] + 1
    assert len(list((folder / "rgb_requests").glob("frame_*.jpg"))) == summary[
        "rgb_requests"]
    assert [p["index"] for p in packets] == list(range(len(packets)))
    assert all(p["source_health"] == "healthy" for p in packets)
    assert summary["find_source_health"]["authorized"]
    times = [row["observation_wall_s"] for row in actions]
    assert all(b > a for a, b in zip(times, times[1:]))
    errors = [float(np.linalg.norm(np.asarray(row["actor_xz_m"]) -
                                   np.asarray(row["evaluator_actual_xz_m"])))
              for row in actions]
    assert all(abs(error - row["evaluator_capture_pose_error_m"]) < 1e-8
               for error, row in zip(errors, actions))
    assert abs(max(errors) - summary["max_evaluator_capture_pose_error_m"]) < 1e-8
    accepted = [a for a in actions if a["decision"] == "forward" and
                a.get("evaluator_motion_accepted")]
    assert len(accepted) == summary["walk_accepted_forward_steps"]
    for index, row in enumerate(actions):
        if row["decision"] == "forward" and row.get("evaluator_motion_accepted"):
            packet = row["pose_increment_packet"]
            assert packet["source"] == name
            assert abs(packet["commanded_step_m"] - row["amount"]) < 1e-8
            assert row["source_guard"]["authorized"]
            if packet["accepted"] and index + 1 < len(actions):
                actual_increment = (np.asarray(actions[index + 1]["actor_xz_m"]) -
                                    np.asarray(row["actor_xz_m"]))
                assert np.allclose(actual_increment,
                                   packet["measured_delta_world_m"], atol=1e-8)
        elif row["decision"] == "forward":
            assert row.get("evaluator_motion_accepted") is False
    denied = [a for a in accepted if not a["pose_increment_packet"]["accepted"]]
    if denied:
        assert actions[-1] is denied[-1]
        assert summary["walk_result"] == "pose_source_unavailable"
    else:
        assert summary["walk_result"] != "pose_source_unavailable"
    final = None
    if (folder / "final_view/trace.json").is_file():
        final = check_visual_stage(folder / "final_view")
        assert summary["final_action"] == final["action"]
        assert summary["final_source_health"]["authorized"]
    else:
        assert summary["final_action"] is None
    return {"room": room, "source": name,
        "walk_result": summary["walk_result"],
        "route_fallback_used": summary["route_fallback_used"],
        "accepted_forward_steps": len(accepted),
        "denied_pose_packets": len(denied),
        "max_capture_pose_error_m": max(errors),
        "final_pose_error_m": summary["evaluator_final_pose_error_m"],
        "evaluator_final_target_gap_m": summary["evaluator_final_target_gap_m"],
        "final_visual_action": final["action"] if final else None,
        "final_selected_target_iou": final["selected_target_iou"] if final else None,
        "range_packets": len(packets),
        "walk_frames": len(actions),
        "media_aligned": True}


def main() -> None:
    frozen = json.loads(PRECOMMIT.read_text())
    assert sha(ROOT / frozen["baseline_audit"]) == frozen[
        "baseline_audit_sha256"]
    for item in frozen["policy_sources"]:
        assert sha(ROOT / item["path"]) == item["sha256"]
    for item in frozen["native_motion_sources"]:
        assert sha(ROOT / item["path"]) == item["sha256"]
    reference = json.loads(REFERENCE.read_text())["rooms"]
    rows = [check_arm(room, source, reference) for room in frozen["rooms"]
            for source in frozen["native_motion_sources"]]
    assert len(rows) == 9
    by_source = {}
    for source in frozen["native_motion_sources"]:
        selected = [row for row in rows if row["source"] == source["name"]]
        by_source[source["name"]] = {
            "rooms": len(selected),
            "walk_standoff": sum(row["walk_result"] ==
                                 "observed_target_standoff" for row in selected),
            "fresh_correct_offers": sum(row["final_visual_action"] ==
                                        "candidate_to_owner_for_confirmation" and
                                        row["final_selected_target_iou"] is not None and
                                        row["final_selected_target_iou"] >= .25
                                        for row in selected),
            "pose_source_stops": sum(row["walk_result"] ==
                                     "pose_source_unavailable" for row in selected),
            "max_capture_pose_error_m": max(row["max_capture_pose_error_m"]
                                             for row in selected)}
    report = {"schema": "ape.replica-pose-source-screen-audit.v1",
        "precommit": str(PRECOMMIT.relative_to(ROOT)),
        "rooms": frozen["rooms"], "native_motion_sources": [s["name"] for s in
            frozen["native_motion_sources"]],
        "summary": by_source, "rows": rows,
        "limits": frozen["limits"]}
    REPORT.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(by_source, indent=2))


if __name__ == "__main__":
    main()
