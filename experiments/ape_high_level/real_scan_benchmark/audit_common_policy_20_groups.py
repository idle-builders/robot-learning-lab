"""Audit the frozen common-policy cohort and the later independent capture screens."""

from __future__ import annotations

import json
from pathlib import Path

import cv2


ROOT = Path(__file__).resolve().parents[3]
BASELINE = ROOT / "artifacts/ape-real-room-next-captures-20261014/common_policy_15_group_report.json"
NEW_ROOT = ROOT / "assets/real_indoor_scenes/scansceneverse/new_capture_owner_20261014"
BRIDGE_ROOT = ROOT / "assets/real_indoor_scenes/scansceneverse/dynamic_placement_screen/common_policy_bridge_20261014"
OUT = ROOT / "artifacts/ape-real-room-next-captures-20261014/common_policy_20_group_full_loop_report.json"


def relative(path: Path) -> str:
    return str(path.relative_to(ROOT))


def aligned(folder: Path) -> bool:
    trace = json.loads((folder / "trace.json").read_text())
    video = cv2.VideoCapture(str(folder / "first_person.mp4"))
    count = 0
    while video.read()[0]:
        count += 1
    video.release()
    return count > 0 and count == trace["frames"] == len(list(folder.glob("rgb_*.png"))) == len(list(folder.glob("depth_*.png")))


def add_screen(rows: list[dict], cohort: str, screen_path: Path, run_root: Path) -> None:
    screen = json.loads(screen_path.read_text())
    for placement in screen["scenes"]:
        group = placement["scene_id"]
        row = {"cohort": cohort, "scene_id": group, "capture_group": group,
               "name": placement["name"], "source_screen": relative(screen_path),
               "placement_status": placement["status"]}
        owner_trace = placement.get("source_delayed_owner_trace")
        if owner_trace:
            owner_folder = (ROOT / owner_trace).parent
            row["owner_trace"] = owner_trace
            row["owner_outcome"] = json.loads((owner_folder / "trace.json").read_text())["outcome"]
            row["owner_media_aligned"] = aligned(owner_folder)
            row["owner_video"] = relative(owner_folder / "first_person.mp4")
        if placement["status"] != "provisional_supported_visible":
            row["task_status"] = "not_run_no_supported_placement"
            row["dino"] = {"status": "not_run_no_supported_placement"}
            row["multi_cue"] = {"status": "not_run_no_supported_placement"}
            rows.append(row)
            continue
        key = placement["scene_key"]
        room_root = (run_root / key.removeprefix("capture_") if cohort == "new_capture"
                     else run_root / key)
        for mode, dirname in (("dino", "teach_find_dino"),
                              ("multi_cue", "teach_find_multi_cue")):
            folder = room_root / dirname
            summary_path = folder / "summary.json"
            if not summary_path.is_file():
                row[mode] = {"status": "not_run"}
                continue
            summary = json.loads(summary_path.read_text())
            if summary.get("source_capture"):
                row["source_capture"] = summary["source_capture"]
            result = {"status": summary["status"], "summary": relative(summary_path),
                      "identity_mode": summary["identity_mode"],
                      "teaching_target_iou": (summary.get("teaching") or {}).get("evaluator_selected_target_iou")}
            if summary["status"] == "teach_find_completed_far_viewpoint":
                conditions = {"find_present" if c["target_present"] else "find_absent": c
                              for c in summary["find_conditions"]}
                for label in ("find_present", "find_absent"):
                    c = conditions[label]
                    result[label] = {"action": c["action"],
                                     "wrong_confident_candidate": c["wrong_confident_candidate"],
                                     "best_target_iou": c["best_proposal_evaluator_target_iou"],
                                     "media_aligned": aligned(folder / label),
                                     "video": relative(folder / label / "first_person.mp4")}
                result["present_correct"] = (result["find_present"]["action"] == "candidate_to_owner_for_confirmation"
                                             and not result["find_present"]["wrong_confident_candidate"])
                result["absent_safe"] = result["find_absent"]["action"] == "report_unidentified"
            row[mode] = result
        row["task_status"] = row["dino"]["status"]
        rows.append(row)


def main() -> None:
    baseline = json.loads(BASELINE.read_text())
    rows = [{**r, "cohort": "baseline", "source_capture": (
        json.loads((ROOT / r["summary"]).read_text()).get("source_capture")
        if r.get("summary") else None), "dino": {
        "status": r["task_status"], "present_correct": r.get("present_correct"),
        "absent_safe": r.get("absent_safe"), "summary": r.get("summary")}}
        for r in baseline["rooms"]]
    add_screen(rows, "new_capture", NEW_ROOT / "teach_find_placement_screen/screen.json", NEW_ROOT)
    add_screen(rows, "bridge", BRIDGE_ROOT / "teach_find_placement_screen/screen.json", BRIDGE_ROOT)
    completed = [r for r in rows if r["dino"]["status"] == "teach_find_completed_far_viewpoint"]
    paired = [r for r in completed if r.get("multi_cue", {}).get("status") == "teach_find_completed_far_viewpoint"]
    groups = [r["capture_group"] for r in completed]
    if len(groups) != len(set(groups)):
        raise ValueError("Completed task scenes duplicate a physical capture group")
    report = {
        "schema": "ape.common-policy-20-group-full-loop-audit.v1",
        "baseline_report": relative(BASELINE),
        "scene_count": len(rows),
        "distinct_screened_capture_groups": len({r["capture_group"] for r in rows}),
        "distinct_completed_capture_groups": len(groups),
        "completed_by_cohort": {cohort: sum(r["cohort"] == cohort for r in completed)
                                for cohort in ("baseline", "new_capture", "bridge")},
        "dino_present_correct": sum(r["dino"].get("present_correct", False) for r in completed),
        "dino_absent_safe": sum(r["dino"].get("absent_safe", False) for r in completed),
        "new_capture_and_bridge_paired_count": len(paired),
        "paired_dino_present_correct": sum(r["dino"].get("present_correct", False) for r in paired),
        "paired_multi_cue_present_correct": sum(r["multi_cue"].get("present_correct", False) for r in paired),
        "paired_dino_absent_safe": sum(r["dino"].get("absent_safe", False) for r in paired),
        "paired_multi_cue_absent_safe": sum(r["multi_cue"].get("absent_safe", False) for r in paired),
        "all_completed_dino_media_aligned": all(
            r["owner_media_aligned"] and all(
                c["media_aligned"] for c in r["find"].values())
            if r["cohort"] == "baseline" else
            r["owner_media_aligned"] and all(
                r["dino"][label]["media_aligned"] for label in ("find_present", "find_absent"))
            for r in completed),
        "new_media_aligned": all(r.get("owner_media_aligned") and all(
            r[mode][label]["media_aligned"] for mode in ("dino", "multi_cue")
            for label in ("find_present", "find_absent")) for r in paired),
        "limits": ["Rooms are real-recorded textured scans with provisional visual quality, not certified photorealistic interaction.",
                   "The rigid scanned person and HOPE objects are inserted into the scene.",
                   "Owner stops use a 17 cm kinematic proxy with ideal yaw and mesh-derived depth/range, not a selected 50 cm articulated ape or physical sensors.",
                   "Placements are frozen by geometry and rendered visibility before item-model outcomes; all groups are development exposed, not a withheld test set."],
        "rooms": rows,
    }
    OUT.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps({k: v for k, v in report.items() if k != "rooms"}, indent=2))
    print(relative(OUT))


if __name__ == "__main__":
    main()
