"""Run frozen item-identity alternatives on two previously unused object pairs."""

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
SCREENS = ROOT / "assets/real_indoor_scenes/object_variation_holdout"
ARTIFACTS = ROOT / "artifacts/ape-real-room-next-captures-20261014"
PRECOMMIT = "artifacts/ape-real-room-next-captures-20261014/common_policy_multicue_holdout_precommit.json"
PAIRS = (("BBQSauce", "SaladDressing"), ("Peaches", "Pineapple"))


def aligned(folder: Path) -> bool:
    trace = json.loads((folder / "trace.json").read_text())
    video = cv2.VideoCapture(str(folder / "first_person.mp4"))
    frames = 0
    while video.read()[0]:
        frames += 1
    video.release()
    return (frames == trace["frames"] ==
            len(list(folder.glob("rgb_*.png"))) ==
            len(list(folder.glob("depth_*.png"))))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--identity-mode", choices=("dino", "multi_cue"),
                        required=True)
    parser.add_argument("--search-photometry", choices=("none", "cool_dim"),
                        default="none")
    args = parser.parse_args()
    if args.search_photometry != "none" and args.identity_mode != "multi_cue":
        raise ValueError("Frozen photometric stress applies to multi-cue policy")
    pairs = PAIRS[:1] if args.search_photometry != "none" else PAIRS
    tag = args.identity_mode + ("_cool_dim" if args.search_photometry != "none" else "")
    output = ARTIFACTS / f"common_policy_multicue_holdout_{tag}_report.json"
    rows = []

    def persist() -> None:
        report = {
            "schema": "ape.common-policy-multicue-holdout-report.v1",
            "precommit": PRECOMMIT,
            "identity_mode": args.identity_mode,
            "search_photometry": args.search_photometry,
            "screened_pair_count": len(pairs),
            "screened_scene_pair_count": len(rows),
            "geometry_supported_count": sum(
                row["placement_status"] == "provisional_supported_visible"
                for row in rows),
            "completed_count": sum(row["task_status"] ==
                "teach_find_completed_far_viewpoint" for row in rows),
            "present_correct_count": sum(row.get("present_correct", False)
                                         for row in rows),
            "absent_safe_count": sum(row.get("absent_safe", False)
                                    for row in rows),
            "wrong_confident_count": sum(row.get("wrong_confident", False)
                                         for row in rows),
            "rows": rows,
        }
        output.write_text(json.dumps(report, indent=2) + "\n")

    for target, distractor in pairs:
        pair_dir = SCREENS / f"{target.lower()}_vs_{distractor.lower()}"
        screen_path = pair_dir / "screen.json"
        screen = json.loads(screen_path.read_text())
        for placement in screen["scenes"]:
            row = {key: placement[key] for key in (
                "scene_id", "capture_group", "name", "scene_key")}
            row.update(target_name=target, distractor_name=distractor,
                       placement_status=placement["status"],
                       source_screen=str(screen_path.relative_to(ROOT)))
            if placement["status"] != "provisional_supported_visible":
                row["task_status"] = "not_run_no_supported_object_view"
                rows.append(row)
                persist()
                continue
            run = pair_dir / f"runs_{tag}" / placement["scene_key"]
            summary_path = run / "summary.json"
            if not summary_path.is_file():
                command = [sys.executable,
                    str(HERE / "run_real_capture_teach_find.py"),
                    "--scene", placement["scene_key"],
                    "--output", str(run),
                    "--simulated-inference-delay-s", "0.85",
                    "--delayed-conversation-screen", str(screen_path),
                    "--target-object", target,
                    "--distractor-object", distractor,
                    "--object-scale", "0.012",
                    "--identity-mode", args.identity_mode,
                    "--search-photometry", args.search_photometry]
                result = subprocess.run(command, cwd=ROOT, capture_output=True,
                    text=True, timeout=180, env={**os.environ,
                        "PYTORCH_ENABLE_MPS_FALLBACK": "1"})
                if result.returncode:
                    row.update(task_status="runner_failed",
                               exit_code=result.returncode,
                               error_tail=result.stderr[-1600:])
                    rows.append(row)
                    persist()
                    print(target, placement["scene_key"], "runner_failed",
                          flush=True)
                    continue
            summary = json.loads(summary_path.read_text())
            row.update(task_status=summary["status"],
                       summary=str(summary_path.relative_to(ROOT)),
                       reused_owner_stop=summary.get("reused_delayed_owner_stop"),
                       teaching_iou=(summary["teaching"][
                           "evaluator_selected_target_iou"]
                           if summary.get("teaching") else None),
                       dino_acceptance=(summary["teaching"][
                           "acceptance_similarity"]
                           if summary.get("teaching") else None),
                       colour_acceptance=(summary["teaching"].get(
                           "colour_acceptance_similarity")
                           if summary.get("teaching") else None))
            if summary.get("teaching"):
                conditions = {}
                for label in ("find_present", "find_absent"):
                    trace = json.loads((run / label / "trace.json").read_text())
                    conditions[label] = {
                        "action": trace["action"],
                        "selected_cue": trace.get("selected_cue"),
                        "cue_disagreement": trace.get("cue_disagreement"),
                        "target_visible_in_any_view": trace[
                            "target_visible_in_any_view"],
                        "wrong_confident_candidate": trace[
                            "wrong_confident_candidate"],
                        "best_target_iou": trace[
                            "best_proposal_evaluator_target_iou"],
                        "media_aligned": aligned(run / label),
                        "video": str((run / label / "first_person.mp4").relative_to(ROOT)),
                        "trace": str((run / label / "trace.json").relative_to(ROOT)),
                    }
                row["find"] = conditions
                row["present_correct"] = bool(
                    conditions["find_present"]["action"] ==
                    "candidate_to_owner_for_confirmation" and
                    not conditions["find_present"]["wrong_confident_candidate"])
                row["absent_safe"] = bool(
                    conditions["find_absent"]["action"] == "report_unidentified")
                row["wrong_confident"] = any(
                    condition["wrong_confident_candidate"]
                    for condition in conditions.values())
                row["present_visible_but_abstained"] = bool(
                    conditions["find_present"]["target_visible_in_any_view"] and
                    conditions["find_present"]["action"] == "report_unidentified")
            rows.append(row)
            persist()
            print(target, placement["scene_key"], row["task_status"],
                  row.get("present_correct"), row.get("absent_safe"),
                  flush=True)
    print(output)


if __name__ == "__main__":
    main()
