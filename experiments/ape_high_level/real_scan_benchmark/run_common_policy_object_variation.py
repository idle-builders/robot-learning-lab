"""Replay one unchanged taught-object policy across screened object/room pairs."""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path

import cv2


ROOT = Path(__file__).resolve().parents[3]
HERE = Path(__file__).resolve().parent
BASE = ROOT / "assets/real_indoor_scenes/object_variation"
OUT = ROOT / "artifacts/ape-real-room-next-captures-20261014/common_policy_object_variation_report.json"
BASELINE = ROOT / "artifacts/ape-real-room-next-captures-20261014/common_policy_15_group_report.json"
PRECOMMIT = "artifacts/ape-real-room-next-captures-20261014/common_policy_object_variation_precommit.json"
PAIRS = (("Ketchup", "Mustard"), ("Yogurt", "Corn"))


def aligned(folder: Path) -> bool:
    trace = json.loads((folder / "trace.json").read_text())
    video = cv2.VideoCapture(str(folder / "first_person.mp4"))
    count = 0
    while video.read()[0]:
        count += 1
    video.release()
    return (count == trace["frames"] ==
            len(list(folder.glob("rgb_*.png"))) ==
            len(list(folder.glob("depth_*.png"))))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--camera-detail", choices=("standard", "high"),
                        default="standard")
    args = parser.parse_args()
    high = args.camera_detail == "high"
    base = (ROOT / "assets/real_indoor_scenes/object_variation_hires"
            if high else BASE)
    out = (ROOT / "artifacts/ape-real-room-next-captures-20261014/"
           "common_policy_camera_detail_report.json" if high else OUT)
    precommit = ("artifacts/ape-real-room-next-captures-20261014/"
                 "common_policy_camera_detail_precommit.json" if high else PRECOMMIT)
    camera_width, camera_height = (1280, 720) if high else (640, 360)
    baseline = {row["scene_id"]: row for row in
                json.loads(BASELINE.read_text())["rooms"]}
    results = []
    for target, distractor in PAIRS:
        folder = base / f"{target.lower()}_vs_{distractor.lower()}"
        screen_path = folder / "screen.json"
        screen = json.loads(screen_path.read_text())
        for placement in screen["scenes"]:
            row = {"scene_id": placement["scene_id"],
                   "capture_group": placement["capture_group"],
                   "name": placement["name"],
                   "target_name": target, "distractor_name": distractor,
                   "placement_status": placement["status"],
                   "source_screen": str(screen_path.relative_to(ROOT)),
                   "baseline_present_correct": baseline[placement["scene_id"]].get(
                       "present_correct"),
                   "baseline_absent_safe": baseline[placement["scene_id"]].get(
                       "absent_safe")}
            if placement["status"] != "provisional_supported_visible":
                row["task_status"] = "not_run_no_supported_object_view"
                results.append(row)
                continue
            output = folder / "runs" / placement["scene_key"]
            summary_path = output / "summary.json"
            if not summary_path.is_file():
                result = subprocess.run([
                    sys.executable, str(HERE / "run_real_capture_teach_find.py"),
                    "--scene", placement["scene_key"],
                    "--output", str(output),
                    "--simulated-inference-delay-s", "0.85",
                    "--delayed-conversation-screen", str(screen_path),
                    "--target-object", target,
                    "--distractor-object", distractor,
                    "--object-scale", "0.012",
                    "--camera-width", str(camera_width),
                    "--camera-height", str(camera_height),
                ], cwd=ROOT, capture_output=True, text=True, timeout=180,
                   env=({**os.environ, "PYTORCH_ENABLE_MPS_FALLBACK": "1"}
                        if high else None))
                if result.returncode:
                    row.update(task_status="runner_failed", exit_code=result.returncode,
                               error_tail=result.stderr[-1500:])
                    results.append(row)
                    print(target, placement["scene_key"], "runner_failed", flush=True)
                    continue
            summary = json.loads(summary_path.read_text())
            row.update(task_status=summary["status"],
                       summary=str(summary_path.relative_to(ROOT)),
                       reused_owner_stop=summary.get("reused_delayed_owner_stop"),
                       teaching_target_iou=(summary["teaching"]["evaluator_selected_target_iou"]
                           if summary.get("teaching") else None),
                       acceptance_similarity=(summary["teaching"]["acceptance_similarity"]
                           if summary.get("teaching") else None))
            if summary.get("teaching"):
                conditions = {}
                for label in ("find_present", "find_absent"):
                    trace = json.loads((output / label / "trace.json").read_text())
                    conditions[label] = {
                        "action": trace["action"],
                        "best_similarity": trace["best_similarity"],
                        "best_target_iou": trace["best_proposal_evaluator_target_iou"],
                        "target_visible_in_any_view": trace["target_visible_in_any_view"],
                        "wrong_confident_candidate": trace["wrong_confident_candidate"],
                        "media_aligned": aligned(output / label),
                        "video": str((output / label / "first_person.mp4").relative_to(ROOT)),
                        "trace": str((output / label / "trace.json").relative_to(ROOT)),
                    }
                row["find"] = conditions
                row["present_correct"] = bool(
                    conditions["find_present"]["action"] ==
                    "candidate_to_owner_for_confirmation" and
                    not conditions["find_present"]["wrong_confident_candidate"])
                row["absent_safe"] = bool(
                    conditions["find_absent"]["action"] == "report_unidentified")
                row["present_visible_but_abstained"] = bool(
                    conditions["find_present"]["target_visible_in_any_view"] and
                    conditions["find_present"]["action"] == "report_unidentified")
            results.append(row)
            report = {"schema": "ape.common-policy-object-variation-report.v1",
                      "precommit": precommit,
                      "baseline": str(BASELINE.relative_to(ROOT)),
                      "teach_find_camera_resolution_px": [camera_width, camera_height],
                      "tested_pair_count": len(PAIRS),
                      "screened_scene_pair_count": len(results),
                      "supported_scene_pair_count": sum(r["placement_status"] ==
                          "provisional_supported_visible" for r in results),
                      "continued_count": sum(r.get("task_status") ==
                          "teach_find_completed_far_viewpoint" for r in results),
                      "present_correct_count": sum(r.get("present_correct", False)
                          for r in results),
                      "absent_safe_count": sum(r.get("absent_safe", False)
                          for r in results),
                      "wrong_confident_count": sum(any(
                          x["wrong_confident_candidate"] for x in r.get("find", {}).values())
                          for r in results),
                      "rows": results}
            out.write_text(json.dumps(report, indent=2) + "\n")
            print(target, placement["scene_key"], row["task_status"],
                  row.get("present_correct"), row.get("absent_safe"), flush=True)
    report.update(screened_scene_pair_count=len(results),
                  supported_scene_pair_count=sum(r["placement_status"] ==
                      "provisional_supported_visible" for r in results),
                  geometry_failed_count=sum(r["placement_status"] ==
                      "placement_screen_failed" for r in results),
                  no_owner_viewpoint_count=sum(r["placement_status"] ==
                      "no_owner_viewpoint" for r in results),
                  per_pair={target: {
                      "supported": sum(r["placement_status"] ==
                          "provisional_supported_visible" for r in results
                          if r["target_name"] == target),
                      "present_correct": sum(r.get("present_correct", False)
                          for r in results if r["target_name"] == target),
                      "absent_safe": sum(r.get("absent_safe", False)
                          for r in results if r["target_name"] == target),
                  } for target, _ in PAIRS})
    out.write_text(json.dumps(report, indent=2) + "\n")
    print(out)


if __name__ == "__main__":
    main()
