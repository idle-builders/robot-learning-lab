"""Audit two whole Replica owner-task transfers from retained sensor media."""

from __future__ import annotations

import json
from pathlib import Path

import cv2
import numpy as np
from PIL import Image

ROOT = Path(__file__).resolve().parents[3]
RUNS = ROOT / "assets/real_indoor_scenes/replica_owner_monitored_20261015"
OUT = ROOT / "artifacts/ape-replica-interactive-20261015/owner_task_audit.json"
ROOMS = ("hotel_0", "apartment_0")


def frames_in_video(path: Path) -> int:
    capture = cv2.VideoCapture(str(path))
    if not capture.isOpened():
        raise AssertionError(f"Video does not open: {path}")
    count = 0
    while True:
        ok, frame = capture.read()
        if not ok:
            break
        assert frame.shape[:2] == (480, 640), path
        count += 1
    capture.release()
    return count


def check_stage(room: Path, name: str) -> dict:
    folder = room / name
    trace = json.loads((folder / "trace.json").read_text())
    count = frames_in_video(folder / "first_person.mp4")
    assert count == trace["frames"], (name, count, trace["frames"])
    for index in range(count):
        rgb = np.asarray(Image.open(folder / f"rgb_{index:03d}.png"))
        depth = np.asarray(Image.open(folder / f"depth_{index:03d}.png"))
        assert rgb.shape == (480, 640, 3), (name, index, rgb.shape)
        assert depth.shape == (480, 640) and depth.dtype == np.uint16
        assert int(np.count_nonzero(depth)) > depth.size // 2
    return {"action": trace["action"], "video_frames": count,
        "wrong_confident_candidate": trace["wrong_confident_candidate"],
        "selected_target_iou": (trace.get("verification") or {}).get(
            "evaluator_selected_target_iou")}


def check_room(name: str) -> dict:
    room = RUNS / name / "none"
    summary = json.loads((room / "summary.json").read_text())
    scenario = json.loads((room / "scenario.json").read_text())
    teaching = json.loads((room / "teach/trace.json").read_text())
    owner = json.loads((room / "owner/decision.json").read_text())
    walk = json.loads((room / "walk/trace.json").read_text())
    assert summary["scene"] == scenario["scene"] == name
    assert all(row["local_floor_y_m"] is not None for row in
               scenario["support_screen"].values())
    assert owner["actor"]["intent"] == "hold_near_person"
    assert teaching["owner_confirmed"] and teaching["selected_box"]
    stages = {stage: check_stage(room, stage) for stage in
              ("find_present", "find_absent", "final_view")}
    assert stages["find_present"]["action"] == "candidate_to_owner_for_confirmation"
    assert stages["find_absent"]["action"] == "report_unidentified"
    assert stages["final_view"]["action"] == "candidate_to_owner_for_confirmation"
    assert not any(row["wrong_confident_candidate"] for row in stages.values())
    assert stages["find_present"]["selected_target_iou"] >= .25
    assert stages["final_view"]["selected_target_iou"] >= .25
    action_rows = walk["actions"]
    assert frames_in_video(room / "walk/first_person.mp4") == len(action_rows)
    assert len(action_rows) <= walk["action_budget"] + 1
    assert all(row["evaluator_motion_accepted"] and 0 < row["amount"] <= .04
               for row in action_rows if row["decision"] == "forward")
    assert action_rows[-1]["decision"] == "hold"
    assert walk["result"] == "observed_target_standoff"
    assert abs(summary["evaluator_final_target_gap_m"] -
               action_rows[-1]["evaluator_target_gap_m"]) < 1e-6
    render_requests = list((room / "rgb_requests").glob("frame_*.jpg"))
    assert len(render_requests) == summary["rgb_requests"]
    assert len(render_requests) >= (2 + 1 + sum(
        row["video_frames"] for row in stages.values()) + len(action_rows))
    return {"room": name, "owner_intent": owner["actor"]["intent"],
        "teaching_confirmed": True, "stages": stages,
        "bounded_forward_steps": walk["accepted_forward_steps"],
        "action_budget": walk["action_budget"],
        "actual_final_gap_m": summary["evaluator_final_target_gap_m"],
        "actor_requested_rgb_views": len(render_requests),
        "media_aligned": True}


def main() -> None:
    rows = [check_room(name) for name in ROOMS]
    old = json.loads((RUNS / "apartment_0/fixed24_failure/summary.json").read_text())
    assert old["walk_result"] == "action_cap"
    result = {"schema": "ape.replica-owner-task-audit.v1",
        "rooms": rows, "room_count": len(rows),
        "baseline_fixed24_apartment_result": old["walk_result"],
        "adaptive_budget_apartment_result": rows[1]["stages"]["final_view"]["action"],
        "qualified_claim": "Two development-exposed official Replica interiors complete scripted owner-to-item task continuations from preselected owner viewpoints, with aligned photographic RGB-D/range media.",
        "limits": "No capture-disjoint holdouts, moving human, independent range source through whole action, physical sensor faults, learned whole-task policy or selected articulated body."}
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
