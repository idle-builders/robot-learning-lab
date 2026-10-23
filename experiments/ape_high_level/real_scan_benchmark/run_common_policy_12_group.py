"""Evaluate one frozen teach/find policy across distinct captured indoor groups."""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

import cv2


ROOT = Path(__file__).resolve().parents[3]
HERE = Path(__file__).resolve().parent
SCREENS = (
    ROOT / "assets/real_indoor_scenes/scansceneverse/dynamic_placement_screen/teach_find_placement_screen/far_owner_11_placement_screen/screen.json",
    ROOT / "assets/real_indoor_scenes/habitat_test_scenes/teach_find_placement_screen/screen.json",
)
EXTENSION_SCREENS = tuple(ROOT / path for path in (
    "assets/real_indoor_scenes/scansceneverse/dynamic_placement_screen/additional_owner_screen/teach_find_screen/screen.json",
    "assets/real_indoor_scenes/scansceneverse/dynamic_placement_screen/second_pass_owner_screen/teach_find_screen/screen.json",
    "assets/real_indoor_scenes/scansceneverse/dynamic_placement_screen/other_living_holdout_owner_screen/teach_find_screen/screen.json",
    "assets/real_indoor_scenes/scansceneverse/dynamic_placement_screen/nazaretian_holdout_owner_screen/teach_find_screen/screen.json",
))
OUT = ROOT / "artifacts/ape-real-room-next-captures-20261014/common_policy_12_group_report.json"
RUNS = ROOT / "assets/real_indoor_scenes/common_policy_12_group"
EXISTING = {
    "habitat_test_scenes/skokloster-castle": ROOT / "assets/real_indoor_scenes/habitat_test_scenes/skokloster-castle/teach_find_depthlayers_delay850ms_v1",
    "habitat_test_scenes/apartment_1": ROOT / "assets/real_indoor_scenes/habitat_test_scenes/apartment_1/teach_find_depthlayers_delay850ms_v1",
}
PRIOR_AUDIT = json.loads((ROOT /
    "assets/real_indoor_scenes/real_capture_20_full_task_split_audit.json").read_text())
CAPTURE_GROUPS = {room["scene_id"]: room["capture_group"]
                  for room in PRIOR_AUDIT["rooms"]}


def video_frames(path: Path) -> int:
    video = cv2.VideoCapture(str(path))
    count = 0
    while video.read()[0]:
        count += 1
    video.release()
    return count


def phase_aligned(folder: Path) -> bool:
    trace = json.loads((folder / "trace.json").read_text())
    frames = trace["frames"]
    return (video_frames(folder / "first_person.mp4") == frames ==
            len(list(folder.glob("rgb_*.png"))) ==
            len(list(folder.glob("depth_*.png"))))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--extended", action="store_true")
    args = parser.parse_args()
    screens = SCREENS + (EXTENSION_SCREENS if args.extended else ())
    report_path = (ROOT / "artifacts/ape-real-room-next-captures-20261014/common_policy_15_group_report.json"
                   if args.extended else OUT)
    precommit = ("artifacts/ape-real-room-next-captures-20261014/common_policy_15_group_extension_precommit.json"
                 if args.extended else
                 "artifacts/ape-real-room-next-captures-20261014/common_policy_12_group_precommit.json")
    rows = []
    for screen_path in screens:
        screen = json.loads(screen_path.read_text())
        for placement in screen["scenes"]:
            row = {"scene_id": placement["scene_id"],
                   "capture_group": CAPTURE_GROUPS.get(
                       placement["scene_id"], placement["scene_id"]),
                   "name": placement["name"],
                   "source_screen": str(screen_path.relative_to(ROOT)),
                   "placement_status": placement["status"]}
            if placement["status"] != "provisional_supported_visible":
                row["task_status"] = "not_run_no_supported_placement"
                rows.append(row)
                continue
            key = placement["scene_key"]
            output = EXISTING.get(placement["scene_id"], RUNS / key)
            summary_path = output / "summary.json"
            if not summary_path.is_file():
                result = subprocess.run([
                    sys.executable, str(HERE / "run_real_capture_teach_find.py"),
                    "--scene", key, "--output", str(output),
                    "--simulated-inference-delay-s", "0.85",
                    "--delayed-conversation-screen", str(screen_path),
                    "--rerun-far-controller", "--depth-separated-overlap",
                    "--pose-mode", "rgbd_yaw",
                ], cwd=ROOT, capture_output=True, text=True, timeout=300)
                if result.returncode:
                    row.update(task_status="runner_failed", exit_code=result.returncode,
                               error_tail=result.stderr[-1600:])
                    rows.append(row)
                    print(key, row["task_status"], flush=True)
                    continue
            summary = json.loads(summary_path.read_text())
            row.update(task_status=summary["status"],
                       summary=str(summary_path.relative_to(ROOT)),
                       owner_outcome=summary["owner_approach_outcome"])
            owner = output / "owner_approach"
            row["owner_media_aligned"] = phase_aligned(owner)
            if summary["teaching"] is not None:
                find = {}
                for label in ("find_present", "find_absent"):
                    trace = json.loads((output / label / "trace.json").read_text())
                    find[label] = {
                        "action": trace["action"],
                        "best_similarity": trace["best_similarity"],
                        "best_proposal_evaluator_target_iou":
                            trace["best_proposal_evaluator_target_iou"],
                        "wrong_confident_candidate": trace["wrong_confident_candidate"],
                        "media_aligned": phase_aligned(output / label),
                        "video": str((output / label / "first_person.mp4").relative_to(ROOT)),
                    }
                row.update(teach_selected_target_iou=summary["teaching"]["evaluator_selected_target_iou"],
                           acceptance_similarity=summary["teaching"]["acceptance_similarity"],
                           find=find,
                           present_correct=bool(find["find_present"]["action"] ==
                               "candidate_to_owner_for_confirmation" and
                               not find["find_present"]["wrong_confident_candidate"]),
                           absent_safe=bool(find["find_absent"]["action"] ==
                               "report_unidentified"))
            rows.append(row)
            report = {"schema": "ape.common-policy-room-report.v2",
                      "precommit": precommit,
                      "capture_group_source": "assets/real_indoor_scenes/real_capture_20_full_task_split_audit.json",
                      "post_run_source_correction": "The precommit called the 12 supported room records independent groups. The Dining and Serving Rooms are two rooms from the same Hallwyl Museum capture; physical capture groups are counted separately from task scenes.",
                      "scene_count": len(rows),
                      "supported_count": sum(r["placement_status"] == "provisional_supported_visible" for r in rows),
                      "completed_full_loop_count": sum(r.get("task_status") == "teach_find_completed_far_viewpoint" for r in rows),
                      "present_correct_count": sum(r.get("present_correct", False) for r in rows),
                      "absent_safe_count": sum(r.get("absent_safe", False) for r in rows),
                      "rooms": rows}
            report_path.write_text(json.dumps(report, indent=2) + "\n")
            print(key, row["task_status"], row.get("present_correct"),
                  row.get("absent_safe"), flush=True)
    report.update(scene_count=len(rows),
                  distinct_supported_group_count=len({r["capture_group"] for r in rows
                      if r["placement_status"] == "provisional_supported_visible"}),
                  distinct_completed_group_count=len({r["capture_group"] for r in rows
                      if r.get("task_status") == "teach_find_completed_far_viewpoint"}),
                  groups_with_at_least_one_success_count=len({r["capture_group"] for r in rows
                      if r.get("present_correct") and r.get("absent_safe")}),
                  groups_with_all_supported_scenes_success_count=sum(
                      all(r.get("present_correct") and r.get("absent_safe")
                          for r in rows if r["capture_group"] == group and
                          r["placement_status"] == "provisional_supported_visible")
                      for group in {r["capture_group"] for r in rows if
                          r["placement_status"] == "provisional_supported_visible"}))
    report_path.write_text(json.dumps(report, indent=2) + "\n")
    print(report_path)


if __name__ == "__main__":
    main()
