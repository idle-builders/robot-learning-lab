"""Audit source quality and complete teach/find outcomes in the fresh capture set."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import cv2


ROOT = Path(__file__).resolve().parents[3]
ARTIFACTS = ROOT / "artifacts/ape-real-room-next-captures-20261014"
BASE = ROOT / "assets/real_indoor_scenes/scansceneverse/fresh_holdout_20261014"
GLBS = ROOT / "assets/real_indoor_scenes/scansceneverse/glbs"
OUT = ARTIFACTS / "fresh_eight_capture_policy_audit.json"


def read(path: Path) -> dict:
    return json.loads(path.read_text())


def relative(path: Path) -> str:
    return str(path.relative_to(ROOT))


def aligned(folder: Path) -> bool:
    trace = read(folder / "trace.json")
    video = cv2.VideoCapture(str(folder / "first_person.mp4"))
    frames = 0
    while video.read()[0]:
        frames += 1
    video.release()
    return frames > 0 and frames == trace["frames"] == len(list(folder.glob("rgb_*.png"))) == len(list(folder.glob("depth_*.png")))


def main() -> None:
    frozen = read(ARTIFACTS / "fresh_eight_capture_holdout_precommit.json")
    screen = read(ARTIFACTS / "fresh_eight_capture_screen_report.json")
    visual = read(ARTIFACTS / "fresh_eight_capture_visual_review.json")
    common = read(ARTIFACTS / "common_policy_20_group_full_loop_report.json")
    old_groups = {r["capture_group"] for r in common["rooms"]}
    visual_by_group = {r["capture_group"]: r for r in visual["rows"]}
    screens = {}
    for pair in frozen["object_pairs"]:
        target, distractor = pair["target"], pair["distractor"]
        key = f"{target.lower()}_{distractor.lower()}"
        folder = BASE / f"item_screen_{key}"
        screens[key] = read(folder / "screen.json")
    rows = []
    for source in frozen["capture_groups"]:
        scene_id = source["id"]
        group = f"scansceneverse/{scene_id}"
        if group in old_groups:
            raise ValueError(f"Fresh group already in development cohort: {group}")
        source_index = GLBS / f"{scene_id}.source.json"
        indexed = read(source_index)
        if hashlib.sha256(source_index.read_bytes()).hexdigest() != frozen["source_index_sha256"][scene_id] or \
                hashlib.sha256((GLBS / f"{scene_id}.glb").read_bytes()).hexdigest() != indexed["sha256"]:
            raise ValueError(f"Source hash mismatch: {scene_id}")
        visual_row = visual_by_group[group]
        room = BASE / scene_id
        row = {"capture_group": group, "name": source["name"],
               "purpose": source["purpose"], "source_index": relative(source_index),
               "visual_status": visual_row["visual_status"],
               "visual_reason": visual_row["reason"],
               "moving_route_video": visual_row.get("moving_route_video"),
               "owner": None, "pairs": {}}
        owner = room / "owner_approach"
        if (owner / "trace.json").is_file():
            trace = read(owner / "trace.json")
            row["owner"] = {"outcome": trace["outcome"],
                            "observed_far_view": trace["observed_far_conversation_band_met"],
                            "accepted_travel_m": trace["accepted_travel_m"],
                            "max_capture_pose_error_m": trace["max_pose_error_at_capture_m"],
                            "media_aligned": aligned(owner),
                            "trace": relative(owner / "trace.json"),
                            "video": relative(owner / "first_person.mp4")}
        for pair in frozen["object_pairs"]:
            target, distractor = pair["target"], pair["distractor"]
            key = f"{target.lower()}_{distractor.lower()}"
            placement = next(p for p in screens[key]["scenes"] if p["scene_id"] == group)
            result = {"placement_status": placement["status"],
                      "source_screen": relative(BASE / f"item_screen_{key}/screen.json")}
            if placement["status"] == "provisional_supported_visible":
                for mode in frozen["item_evaluation_arms"]:
                    folder = room / f"teach_find_{target.lower()}_{mode}"
                    summary_path = folder / "summary.json"
                    if not summary_path.is_file():
                        result[mode] = {"status": "not_run"}
                        continue
                    summary = read(summary_path)
                    arm = {"status": summary["status"], "summary": relative(summary_path),
                           "teaching_target_iou": (summary.get("teaching") or {}).get("evaluator_selected_target_iou")}
                    if summary["status"] == "teach_find_completed_far_viewpoint":
                        for condition in summary["find_conditions"]:
                            label = "find_present" if condition["target_present"] else "find_absent"
                            arm[label] = {"action": condition["action"],
                                          "wrong_confident_candidate": condition["wrong_confident_candidate"],
                                          "best_target_iou": condition["best_proposal_evaluator_target_iou"],
                                          "media_aligned": aligned(folder / label),
                                          "video": relative(folder / label / "first_person.mp4")}
                        arm["present_correct"] = (arm["find_present"]["action"] == "candidate_to_owner_for_confirmation"
                                                  and not arm["find_present"]["wrong_confident_candidate"])
                        arm["absent_safe"] = arm["find_absent"]["action"] == "report_unidentified"
                    result[mode] = arm
            row["pairs"][key] = result
        rows.append(row)
    supported = [(r, p) for r in rows for p in r["pairs"].values()
                 if p["placement_status"] == "provisional_supported_visible"]
    report = {"schema": "ape.fresh-eight-capture-policy-audit.v1",
              "precommit": relative(ARTIFACTS / "fresh_eight_capture_holdout_precommit.json"),
              "capture_group_count": len(rows),
              "visually_provisional_group_count": sum(r["visual_status"] == "provisional_primary_candidate" for r in rows),
              "completed_owner_group_count": sum(bool(r["owner"] and r["owner"]["outcome"] == "hold_viewpoint_far") for r in rows),
              "screened_pair_count": len(rows) * len(frozen["object_pairs"]),
              "supported_pair_count": len(supported),
              "completed_pair_arms": {mode: sum(p.get(mode, {}).get("status") == "teach_find_completed_far_viewpoint"
                                               for _, p in supported)
                                      for mode in frozen["item_evaluation_arms"]},
              "present_correct": {mode: sum(p.get(mode, {}).get("present_correct", False)
                                            for _, p in supported)
                                  for mode in frozen["item_evaluation_arms"]},
              "absent_safe": {mode: sum(p.get(mode, {}).get("absent_safe", False)
                                       for _, p in supported)
                              for mode in frozen["item_evaluation_arms"]},
              "limits": "The visual screen retained only one provisional moving-view candidate; this is not a broad or certified photoreal holdout. Inserted rigid person, HOPE props, ideal yaw/range and kinematic proxy remain.",
              "rooms": rows}
    OUT.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps({k: v for k, v in report.items() if k != "rooms"}, indent=2))


if __name__ == "__main__":
    main()
