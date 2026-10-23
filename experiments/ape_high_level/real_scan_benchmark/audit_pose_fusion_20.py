"""Check paired captured-room motion/localization outcomes and saved media."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import cv2
import numpy as np
from PIL import Image
from run_common_policy_20_conditional_search import composite_aligned
from run_common_policy_20_object_generalization import aligned


ROOT = Path(__file__).resolve().parents[3]
ART = ROOT / "artifacts/ape-real-room-next-captures-20261014"
PRECOMMIT = ART / "pose_fusion_20_precommit.json"
REPORT = ART / "pose_fusion_20_report.json"
AUDIT = ART / "pose_fusion_20_audit.json"


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def frames(path: Path) -> int:
    cap = cv2.VideoCapture(str(path))
    if not cap.isOpened():
        raise AssertionError(f"Video did not open: {path}")
    count = 0
    while True:
        ok, _ = cap.read()
        if not ok:
            break
        count += 1
    cap.release()
    return count


def arm(row: dict, corrected: bool) -> dict:
    walk = json.loads((ROOT / row["walk_trace"]).read_text())
    actions = walk["actions"]
    assert actions and [r["frame"] for r in actions] == list(range(len(actions)))
    assert frames(ROOT / row["walk_video"]) == len(actions)
    assert len(actions) <= walk["action_budget"] + 1
    assert walk["accepted_forward_steps"] == row["accepted_forward_steps"]
    assert walk["result"] == row["walk_result"]
    errors = []
    for action in actions:
        rgb = np.asarray(Image.open(ROOT / action["rgb"]))
        depth = np.asarray(Image.open(ROOT / action["depth"]))
        assert rgb.ndim == 3 and rgb.shape[:2] == depth.shape
        assert depth.dtype == np.uint16
        assert len(action["range_angles_deg"]) == 37
        assert len(action["range_planes_m"]) == 2
        assert all(len(plane) == 37 for plane in action["range_planes_m"])
        assert action["range_capture_wall_s"] == action["observation_wall_s"]
        error = float(np.linalg.norm(np.asarray(action["actor_xz_m"]) -
                                     np.asarray(action["evaluator_actual_xz_m"])))
        assert abs(error - action["evaluator_capture_pose_error_m"]) < 1e-8
        errors.append(error)
        if corrected:
            assert "range_pose_correction" in action
        else:
            assert "range_pose_correction" not in action
        if action["decision"] == "forward" and action.get("evaluator_motion_accepted"):
            packet = action["pose_increment_packet"]
            assert packet["source"] == row["motion_source"]
            assert packet["accepted"] or walk["result"] == "pose_source_unavailable"
    assert abs(max(errors) - row["max_evaluator_capture_pose_error_m"]) < 1e-8
    assert actions[0].get("range_pose_correction", {}).get("reason") == (
        "first_range_anchor" if corrected else None)
    if row["walk_result"] == "observed_target_standoff":
        assert row["final_trace"] and row["final_video"]
        final = json.loads((ROOT / row["final_trace"]).read_text())
        folder = (ROOT / row["final_trace"]).parent
        assert frames(ROOT / row["final_video"]) == len(final["video_frames"])
        assert composite_aligned(folder, final)
        assert all(aligned((ROOT / s["trace"]).parent)
                   for s in final["stage_reports"])
        assert final["action"] == row["final_action"]
        assert final["wrong_confident_candidate"] == row["final_wrong_confident"]
        iou = final["best_proposal_evaluator_target_iou"]
    else:
        assert row["final_trace"] is None and row["final_action"] is None
        iou = None
    return {"scene_key": row["scene_key"], "capture_group": row["capture_group"],
        "arm": row["arm"], "walk_result": row["walk_result"],
        "accepted_forward_steps": row["accepted_forward_steps"],
        "simulator_rejected_steps": sum(a["decision"] == "forward" and
                                        a.get("evaluator_motion_accepted") is False
                                        for a in actions),
        "denied_motion_packets": sum(a.get("pose_increment_packet", {}).get(
                                     "accepted") is False for a in actions),
        "ambiguous_scan_matches": sum(a.get("range_pose_correction", {}).get(
                                      "accepted") is False for a in actions),
        "missing_range_sectors": sum(v is None for a in actions for plane in
                                     a["range_planes_m"] for v in plane),
        "route_fallback_used": row["route_fallback_used"],
        "max_capture_pose_error_m": max(errors),
        "final_target_gap_m": row["evaluator_final_target_gap_m"],
        "final_action": row["final_action"],
        "final_selected_target_iou": iou,
        "correct_fresh_offer": bool(row["final_action"] ==
            "owner_confirmed_visual_identity" and iou is not None and iou >= .25),
        "wrong_confident_offer": row["final_wrong_confident"],
        "frames": len(actions), "media_aligned": True}


def main() -> None:
    frozen = json.loads(PRECOMMIT.read_text())
    for key, digest in (("cohort", "cohort_sha256"),
                        ("source_audit", "source_audit_sha256"),
                        ("command_approach_audit", "command_approach_audit_sha256"),
                        ("motion_trace", "motion_trace_sha256")):
        assert sha(ROOT / frozen[key]) == frozen[digest]
    for item in frozen["policy_sources"]:
        assert sha(ROOT / item["path"]) == item["sha256"]
    data = json.loads(REPORT.read_text())
    assert data["processed"] == 20 and len(data["rows"]) == 20
    assert {r["capture_group"] for r in data["rows"]} == {
        r["capture_group"] for r in frozen["rooms"]}
    rows = []
    for pair in data["rows"]:
        assert [a["arm"] for a in pair["arms"]] == frozen["paired_arms"]
        left, right = pair["arms"]
        assert left["source_owner_trace"] == right["source_owner_trace"]
        assert left["source_confirmed_trace"] == right["source_confirmed_trace"]
        assert left["source_absent_trace"] == right["source_absent_trace"]
        assert left["motion_source_sha256"] == right["motion_source_sha256"] == \
            frozen["motion_trace_sha256"]
        rows.extend((arm(left, False), arm(right, True)))
    summary = {}
    for name in frozen["paired_arms"]:
        selected = [r for r in rows if r["arm"] == name]
        summary[name] = {"rooms": len(selected),
            "fresh_correct_offers": sum(r["correct_fresh_offer"] for r in selected),
            "wrong_confident_offers": sum(r["wrong_confident_offer"] for r in selected),
            "observed_standoffs": sum(r["walk_result"] ==
                                      "observed_target_standoff" for r in selected),
            "honest_stops": sum(r["walk_result"] != "observed_target_standoff"
                                and not r["wrong_confident_offer"] for r in selected),
            "accepted_forward_steps": sum(r["accepted_forward_steps"] for r in selected),
            "simulator_rejected_steps": sum(r["simulator_rejected_steps"] for r in selected),
            "denied_motion_packets": sum(r["denied_motion_packets"] for r in selected),
            "ambiguous_scan_matches": sum(r["ambiguous_scan_matches"] for r in selected),
            "missing_range_sectors": sum(r["missing_range_sectors"] for r in selected),
            "max_capture_pose_error_m": max(r["max_capture_pose_error_m"]
                                             for r in selected)}
    AUDIT.write_text(json.dumps({"schema": "ape.pose-fusion-20-audit.v1",
        "precommit": str(PRECOMMIT.relative_to(ROOT)), "summary": summary,
        "rows": rows, "limits": frozen["limits"]}, indent=2) + "\n")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
