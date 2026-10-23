"""Run paired present/absent teach/find continuations in screened rooms."""

from __future__ import annotations

import json
import argparse
import subprocess
import sys
import time
from collections import Counter
from pathlib import Path

import cv2


HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
SCREEN = ROOT / "assets/real_indoor_scenes/scansceneverse/dynamic_placement_screen/teach_find_placement_screen/screen.json"
RUNNER = HERE / "run_real_capture_teach_find.py"
OUT = SCREEN.parent / "batch_v1.json"


def count_video_frames(path: Path) -> int:
    video = cv2.VideoCapture(str(path))
    count = 0
    while True:
        ok, _ = video.read()
        if not ok:
            break
        count += 1
    video.release()
    return count


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--simulated-inference-delay-s", type=float, default=0.)
    args = parser.parse_args()
    tag = (f"_delay{round(1000 * args.simulated_inference_delay_s)}ms"
           if args.simulated_inference_delay_s else "")
    out = SCREEN.parent / f"batch{tag}_v1.json"
    screen = json.loads(SCREEN.read_text())
    rows = []
    for placement in screen["scenes"]:
        row = {"scene_id": placement["scene_id"], "name": placement["name"],
               "placement_status": placement["status"]}
        if placement["status"] != "provisional_supported_visible":
            row["task_status"] = "not_run_no_supported_placement"
            rows.append(row)
            continue
        key = placement["scene_key"]
        output = SCREEN.parent / "runs" / (key.removeprefix("capture_") + tag)
        print("run", placement["name"], flush=True)
        started = time.perf_counter()
        result = subprocess.run([sys.executable, str(RUNNER), "--scene", key,
            "--output", str(output), "--simulated-inference-delay-s",
            str(args.simulated_inference_delay_s)], cwd=ROOT, capture_output=True,
            text=True, timeout=300)
        row["runtime_wall_s"] = time.perf_counter() - started
        if result.returncode:
            row.update(task_status="runner_failed", exit_code=result.returncode,
                       error_tail=result.stderr[-1500:])
        else:
            summary = json.loads((output / "summary.json").read_text())
            owner = json.loads((output / "owner_approach/trace.json").read_text())
            owner_frames = count_video_frames(output / "owner_approach/first_person.mp4")
            if not (owner_frames == owner["frames"] ==
                    len(list((output / "owner_approach").glob("rgb_*.png"))) ==
                    len(list((output / "owner_approach").glob("depth_*.png")))):
                raise RuntimeError(f"Unaligned owner media: {key}")
            row.update(task_status=summary.get("status", "teach_find_completed"),
                owner_outcome=owner["outcome"], owner_frames=owner_frames,
                summary=str((output / "summary.json").relative_to(ROOT)),
                owner_video=str((output / "owner_approach/first_person.mp4").relative_to(ROOT)))
            if summary["teaching"] is not None:
                finds = {}
                for condition in ("find_present", "find_absent"):
                    trace = json.loads((output / condition / "trace.json").read_text())
                    frames = count_video_frames(output / condition / "first_person.mp4")
                    if not (frames == trace["frames"] ==
                            len(list((output / condition).glob("rgb_*.png"))) ==
                            len(list((output / condition).glob("depth_*.png")))):
                        raise RuntimeError(f"Unaligned find media: {key}/{condition}")
                    finds[condition] = {"action": trace["action"], "frames": frames,
                        "best_similarity": trace["best_similarity"],
                        "best_target_iou": trace["best_proposal_evaluator_target_iou"],
                        "target_visible_in_any_view": trace["target_visible_in_any_view"],
                        "wrong_confident_candidate": trace["wrong_confident_candidate"],
                        "video": str((output / condition / "first_person.mp4").relative_to(ROOT)),
                        "trace": str((output / condition / "trace.json").relative_to(ROOT))}
                row.update(teach_target_iou=summary["teaching"]["evaluator_selected_target_iou"],
                    acceptance_similarity=summary["teaching"]["acceptance_similarity"],
                    find=finds)
        rows.append(row)
        report = {"schema": "ape.real-scan-teach-find-batch.v1",
            "placement_screen": str(SCREEN.relative_to(ROOT)),
            "scene_count": len(screen["scenes"]),
            "supported_placement_count": screen["supported_count"],
            "simulated_inference_delay_s": args.simulated_inference_delay_s,
            "owner_viewpoint_count": sum(r.get("owner_outcome") == "hold_viewpoint" for r in rows),
            "completed_count": sum(r["task_status"] in ("completed", "teach_find_completed") for r in rows),
            "present_candidate_count": sum(r.get("find", {}).get("find_present", {}).get(
                "action") == "candidate_to_owner_for_confirmation" for r in rows),
            "absent_false_candidate_count": sum(r.get("find", {}).get("find_absent", {}).get(
                "action") == "candidate_to_owner_for_confirmation" for r in rows),
            "outcome_counts": dict(Counter(r["task_status"] for r in rows)),
            "scenes": rows}
        out.write_text(json.dumps(report, indent=2) + "\n")
        print(row["task_status"], row.get("find", {}).get("find_present", {}).get(
            "action", ""), row.get("find", {}).get("find_absent", {}).get(
            "action", ""), flush=True)
    print(out)


if __name__ == "__main__":
    main()
