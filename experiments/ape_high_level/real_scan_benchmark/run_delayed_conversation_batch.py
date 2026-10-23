"""Continue frozen delayed owner stops from a farther conversation viewpoint."""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from collections import Counter
from pathlib import Path

from run_scanned_teach_find_batch import count_video_frames


HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
BASE = ROOT / "assets/real_indoor_scenes/scansceneverse/dynamic_placement_screen/teach_find_placement_screen"
SCREEN = BASE / "delayed_conversation_placement_screen/screen.json"
RUNNER = HERE / "run_real_capture_teach_find.py"


def aligned(folder: Path) -> int:
    trace = json.loads((folder / "trace.json").read_text())
    frames = count_video_frames(folder / "first_person.mp4")
    if not (frames == trace["frames"] ==
            len(list(folder.glob("rgb_*.png"))) ==
            len(list(folder.glob("depth_*.png")))):
        raise RuntimeError(f"Unaligned sensor media: {folder}")
    return frames


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--rerun-far-controller", action="store_true")
    parser.add_argument("--depth-separated-overlap", action="store_true")
    parser.add_argument("--screen", type=Path, default=SCREEN)
    parser.add_argument("--experiment-tag", default="",
                        help="Keep a changed decision policy's runs separate from the frozen baseline")
    parser.add_argument("--resume-completed", action="store_true",
                        help="Reuse only completed episodes for this same frozen screen")
    args = parser.parse_args()
    screen_path = args.screen.resolve()
    if args.experiment_tag and not all(c.isalnum() or c in "_-" for c in args.experiment_tag):
        parser.error("experiment tag must use letters, digits, underscores or hyphens")
    suffix = f"_{args.experiment_tag}" if args.experiment_tag else ""
    out_file = screen_path.parent / (("integrated_batch" if
        args.rerun_far_controller else "batch") + suffix + "_v1.json")
    screen = json.loads(screen_path.read_text())
    rows = []
    for placement in screen["scenes"]:
        row = {"scene_id": placement["scene_id"], "name": placement["name"],
               "placement_status": placement["status"]}
        if placement["status"] != "provisional_supported_visible":
            row["task_status"] = "not_run_no_supported_placement"
            rows.append(row)
            continue
        key = placement["scene_key"]
        out = screen_path.parent / (("integrated_runs" if args.rerun_far_controller
                               else "runs") + suffix) / key.removeprefix("capture_")
        print("run", placement["name"], flush=True)
        started = time.perf_counter()
        command = [sys.executable, str(RUNNER), "--scene", key,
            "--output", str(out), "--simulated-inference-delay-s", "0.85",
            "--delayed-conversation-screen", str(screen_path)]
        if args.rerun_far_controller:
            command.append("--rerun-far-controller")
        if args.depth_separated_overlap:
            command.append("--depth-separated-overlap")
        previous = None
        if args.resume_completed and (out / "summary.json").exists():
            previous = json.loads((out / "summary.json").read_text())
        reuse = bool(previous is not None and previous.get("status") in (
            "teach_find_completed", "teach_find_completed_far_viewpoint") and
            previous.get("scenario_placement_screen") == str(screen_path.relative_to(ROOT)) and
            previous.get("simulated_inference_delay_s") == .85 and
            previous.get("depth_separated_person_overlap", False) == args.depth_separated_overlap and
            previous.get("reused_delayed_owner_stop") == (not args.rerun_far_controller))
        result = (subprocess.CompletedProcess(command, 0, "", "") if reuse else
            subprocess.run(command, cwd=ROOT, capture_output=True,
                           text=True, timeout=180))
        row["runtime_wall_s"] = time.perf_counter() - started
        row["reused_completed_evidence"] = reuse
        if result.returncode:
            row.update(task_status="runner_failed", error_tail=result.stderr[-1500:])
        else:
            summary = json.loads((out / "summary.json").read_text())
            owner_dir = (out / "owner_approach" if args.rerun_far_controller
                         else (ROOT / placement["source_delayed_owner_trace"]).parent)
            owner_frames = aligned(owner_dir)
            row.update(task_status=summary["status"],
                owner_outcome=summary["owner_approach_outcome"],
                owner_frames=owner_frames,
                summary=str((out / "summary.json").relative_to(ROOT)),
                owner_video=str((owner_dir / "first_person.mp4").relative_to(ROOT)))
            if summary["teaching"] is not None:
                finds = {}
                for condition in ("find_present", "find_absent"):
                    trace = json.loads((out / condition / "trace.json").read_text())
                    frames = aligned(out / condition)
                    finds[condition] = {"action": trace["action"],
                        "frames": frames,
                        "best_similarity": trace["best_similarity"],
                        "best_target_iou": trace["best_proposal_evaluator_target_iou"],
                        "verification_target_iou": (trace["verification"]["evaluator_selected_target_iou"]
                            if trace["verification"] is not None else None),
                        "target_visible_in_any_view": trace["target_visible_in_any_view"],
                        "wrong_confident_candidate": trace["wrong_confident_candidate"],
                        "video": str((out / condition / "first_person.mp4").relative_to(ROOT)),
                        "trace": str((out / condition / "trace.json").relative_to(ROOT))}
                row.update(teach_target_iou=summary["teaching"]["evaluator_selected_target_iou"],
                    acceptance_similarity=summary["teaching"]["acceptance_similarity"],
                    find=finds)
        rows.append(row)
        report = {"schema": "ape.delayed-owner-conversation-teach-find-batch.v1",
            "placement_screen": str(screen_path.relative_to(ROOT)),
            "reuse_existing_delayed_owner_traces": not args.rerun_far_controller,
            "depth_separated_person_overlap": args.depth_separated_overlap,
            "scene_count": len(screen["scenes"]),
            "supported_placement_count": screen["supported_count"],
            "completed_count": sum(r["task_status"] in (
                "teach_find_completed_far_viewpoint", "teach_find_completed") for r in rows),
            "present_correct_candidate_count": sum(r.get("find", {}).get(
                "find_present", {}).get("action") ==
                "candidate_to_owner_for_confirmation" and r.get("find", {}).get(
                "find_present", {}).get("verification_target_iou", 0.) >= .25 for r in rows),
            "absent_false_candidate_count": sum(r.get("find", {}).get(
                "find_absent", {}).get("action") ==
                "candidate_to_owner_for_confirmation" for r in rows),
            "outcome_counts": dict(Counter(r["task_status"] for r in rows)),
            "scenes": rows}
        out_file.write_text(json.dumps(report, indent=2) + "\n")
        print(row["task_status"], row.get("find", {}).get(
            "find_present", {}).get("action"), flush=True)
    print(out_file)


if __name__ == "__main__":
    main()
