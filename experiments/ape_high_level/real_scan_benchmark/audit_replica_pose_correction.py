"""Audit the frozen nine-arm adjacent-scan correction comparison."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np

from audit_replica_full_monitor import check_visual_stage, video_frames

ROOT = Path(__file__).resolve().parents[3]
ART = ROOT / "artifacts/ape-replica-interactive-20261015"
RUNS = ROOT / "assets/real_indoor_scenes/replica_owner_pose_corrected_20261015"
PRECOMMIT = ART / "range_pose_correction_precommit.json"
REPORT = ART / "range_pose_correction_audit.json"


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def check_arm(room: str, source: dict, reference: dict,
              uncorrected: dict) -> dict:
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
    assert summary["pose_correction"] == scenario["pose_correction"] == "range_scan"
    assert summary["pose_source_trace_sha256"] == source["sha256"]
    assert summary["route_fallback"] and scenario["route_fallback"]
    assert summary["range_gap_stage"] == "none"
    assert summary["teaching_confirmed"]
    assert summary["present_action"] == "candidate_to_owner_for_confirmation"
    assert summary["absent_action"] == "report_unidentified"
    assert not summary["present_wrong_confident"] and not summary["absent_wrong_confident"]
    for item in ("owner/owner_rgb.png", "teach/teach_rgb.png"):
        assert sha(folder / item) == reference[room][item]
    for stage in ("find_present", "find_absent"):
        visual = check_visual_stage(folder / stage)
        assert [sha(folder / stage / f"rgb_{i:03d}.png")
                for i in range(visual["frames"])] == reference[room][stage]
    assert video_frames(folder / "walk/first_person.mp4") == len(actions)
    assert len(actions) <= walk["action_budget"] + 1
    assert [p["index"] for p in packets] == list(range(len(packets)))
    assert all(p["source_health"] == "healthy" for p in packets)
    assert summary["walk_source_health"]["authorized"]
    assert [a["frame"] for a in actions] == list(range(len(actions)))
    times = [a["observation_wall_s"] for a in actions]
    assert all(b > a for a, b in zip(times, times[1:]))
    errors = [float(np.linalg.norm(np.asarray(a["actor_xz_m"]) -
                                   np.asarray(a["evaluator_actual_xz_m"])))
              for a in actions]
    assert all(abs(e - a["evaluator_capture_pose_error_m"]) < 1e-8
               for e, a in zip(errors, actions))
    assert abs(max(errors) - summary["max_evaluator_capture_pose_error_m"]) < 1e-8
    assert all("range_pose_correction" in a for a in actions)
    assert actions[0]["range_pose_correction"]["reason"] == "first_range_anchor"
    accepted = [a for a in actions if a["decision"] == "forward" and
                a.get("evaluator_motion_accepted")]
    assert len(accepted) == summary["walk_accepted_forward_steps"]
    for a in accepted:
        assert a["source_guard"]["authorized"]
        assert a["pose_increment_packet"]["source"] == name
    if (folder / "final_view/trace.json").is_file():
        final = check_visual_stage(folder / "final_view")
        assert final["action"] == summary["final_action"]
        assert summary["final_source_health"]["authorized"]
    else:
        final = None
        assert summary["final_action"] is None
    previous = uncorrected[(room, name)]
    return {"room": room, "source": name,
        "walk_result": summary["walk_result"],
        "accepted_forward_steps": len(accepted),
        "scan_matches": sum(a["range_pose_correction"]["reason"] ==
                            "range_scan_match" for a in actions),
        "ambiguous_scans": sum(not a["range_pose_correction"]["accepted"]
                               for a in actions),
        "max_capture_pose_error_m": max(errors),
        "max_uncorrected_pose_error_m": previous["max_capture_pose_error_m"],
        "final_pose_error_m": summary["evaluator_final_pose_error_m"],
        "final_uncorrected_pose_error_m": previous["final_pose_error_m"],
        "final_visual_action": final["action"] if final else None,
        "final_selected_target_iou": final["selected_target_iou"] if final else None,
        "range_packets": len(packets), "walk_frames": len(actions),
        "media_aligned": True}


def main() -> None:
    frozen = json.loads(PRECOMMIT.read_text())
    assert sha(ROOT / frozen["uncorrected_audit"]) == frozen["uncorrected_audit_sha256"]
    for item in frozen["policy_sources"] + frozen["historical_motion_sources"]:
        assert sha(ROOT / item["path"]) == item["sha256"]
    reference = json.loads((ART / "monitored_reference_hashes.json").read_text())["rooms"]
    old = json.loads((ROOT / frozen["uncorrected_audit"]).read_text())
    uncorrected = {(r["room"], r["source"]): r for r in old["rows"]}
    rows = [check_arm(room, source, reference, uncorrected)
            for room in frozen["rooms"]
            for source in frozen["historical_motion_sources"]]
    assert len(rows) == 9
    by_source = {}
    for source in frozen["historical_motion_sources"]:
        selected = [r for r in rows if r["source"] == source["name"]]
        by_source[source["name"]] = {
            "rooms": len(selected),
            "fresh_correct_offers": sum(r["final_visual_action"] ==
                "candidate_to_owner_for_confirmation" and
                r["final_selected_target_iou"] is not None and
                r["final_selected_target_iou"] >= .25 for r in selected),
            "uncorrected_fresh_correct_offers": sum(
                uncorrected[(r["room"], r["source"])]["final_visual_action"] ==
                "candidate_to_owner_for_confirmation" for r in selected),
            "max_capture_pose_error_m": max(r["max_capture_pose_error_m"]
                                             for r in selected),
            "ambiguous_scans": sum(r["ambiguous_scans"] for r in selected)}
    report = {"schema": "ape.replica-range-pose-correction-audit.v1",
        "precommit": str(PRECOMMIT.relative_to(ROOT)),
        "rooms": frozen["rooms"], "summary": by_source,
        "rows": rows, "limits": frozen["limits"]}
    REPORT.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(by_source, indent=2))


if __name__ == "__main__":
    main()
