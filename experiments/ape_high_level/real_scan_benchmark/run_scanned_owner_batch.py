"""Run the same bounded scanned-owner loop in every supported capture room.

The placement screen is frozen before reading any owner-loop outcome. Failed
placements stay in the report rather than being counted as completed tasks.
"""

from __future__ import annotations

import argparse
import json
import re
import subprocess
import sys
import time
from collections import Counter
from pathlib import Path

import cv2
import numpy as np


HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
SCREEN = ROOT / "assets/real_indoor_scenes/scansceneverse/dynamic_placement_screen/screen.json"
RUNNER = HERE / "run_real_capture_owner_loop.py"


def video_frame_count(path: Path) -> int:
    video = cv2.VideoCapture(str(path))
    frames = 0
    while True:
        ok, _ = video.read()
        if not ok:
            break
        frames += 1
    video.release()
    return frames


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--pose-mode", choices=("command", "rgbd_yaw",
                                                "native_shaped", "rgbd_native_fallback"),
                        default="rgbd_yaw")
    parser.add_argument("--native-source", choices=("planar200v3-forward",
                                                     "varied150-forward-seed144",
                                                     "robust10v200-forward-seed144"),
                        default="varied150-forward-seed144")
    parser.add_argument("--without-pose-crosscheck", action="store_true")
    parser.add_argument("--person-mask-odometry", action="store_true")
    parser.add_argument("--simulated-inference-delay-s", type=float, default=0.)
    parser.add_argument("--fast-revalidate-during-inference", action="store_true")
    parser.add_argument("--spatial-range-association", action="store_true")
    parser.add_argument("--sam2-range-association", action="store_true")
    parser.add_argument("--far-conversation-viewpoint", action="store_true")
    parser.add_argument("--depth-separated-overlap", action="store_true")
    parser.add_argument("--receding-owner-target", action="store_true")
    parser.add_argument("--screen", type=Path, default=SCREEN)
    parser.add_argument("--steps", type=int, default=60)
    parser.add_argument("--run-label", default="",
                        help="Unique suffix for a frozen comparison; keeps earlier runs intact")
    parser.add_argument("--resume-existing", action="store_true",
                        help="Audit an already completed trace/video instead of rerunning it")
    args = parser.parse_args()
    if args.run_label and not re.fullmatch(r"[a-z0-9_-]+", args.run_label):
        parser.error("run label must use lowercase letters, digits, _ or -")
    screen = args.screen.resolve()
    if args.person_mask_odometry and args.pose_mode not in ("rgbd_yaw", "rgbd_native_fallback"):
        parser.error("person mask requires visual RGB-D pose")
    if args.fast_revalidate_during_inference and not args.simulated_inference_delay_s:
        parser.error("fast revalidation requires inference delay")
    if args.spatial_range_association and args.sam2_range_association:
        parser.error("choose either box or SAM 2 range association")
    crosscheck = args.pose_mode in ("rgbd_yaw", "rgbd_native_fallback") and not args.without_pose_crosscheck
    tag = (f"{args.pose_mode}_arbitrated" +
           ("_crosschecked" if crosscheck else "") +
           ("_personmasked" if args.person_mask_odometry else "") +
           (f"_delay{round(1000 * args.simulated_inference_delay_s)}ms"
            if args.simulated_inference_delay_s else "") +
           ("_fastrevalidate" if args.fast_revalidate_during_inference else "") +
           ("_rangeprojected" if args.spatial_range_association else "") +
           ("_rangesam2" if args.sam2_range_association else "") +
           ("_farconverse" if args.far_conversation_viewpoint else "") +
           ("_depthlayers" if args.depth_separated_overlap else "") +
           ("_recedingowner" if args.receding_owner_target else ""))
    if args.pose_mode in ("native_shaped", "rgbd_native_fallback"):
        tag += "_" + args.native_source
    if args.run_label:
        tag += "_" + args.run_label
    out = screen.parent / f"owner_batch_{tag}_v1.json"
    suffix = f"dynamic_owner_{tag}_scanned_placement_screen_v1_episode"
    screen_data = json.loads(screen.read_text())
    placements = screen_data.get("scenes", [screen_data])
    rows = []
    for placement in placements:
        scene = "capture_" + Path(placement["source_trace"]).parent.name
        if placement["status"] != "provisional_supported_visible":
            rows.append({"scene_id": placement["scene_id"], "name": placement["name"],
                         "placement_status": placement["status"],
                         "task_status": "not_run_no_supported_placement"})
            continue
        print("run", placement["name"], flush=True)
        started = time.perf_counter()
        folder = (ROOT / placement["source_trace"]).parent / suffix
        trace_path = folder / "trace.json"
        resumed = bool(args.resume_existing and trace_path.is_file())
        result = None if resumed else subprocess.run([
            sys.executable, str(RUNNER), "--scene", scene,
            "--person-style", "scanned", "--pose-mode", args.pose_mode,
            "--steps", str(args.steps), "--device", "mps",
            "--native-source", args.native_source,
            "--output-dir", str(folder),
            "--gaze-route-arbitration",
            "--simulated-inference-delay-s", str(args.simulated_inference_delay_s),
            *(["--pose-crosscheck"] if crosscheck else []),
            *(["--person-mask-odometry"] if args.person_mask_odometry else []),
            *(["--fast-revalidate-during-inference"] if args.fast_revalidate_during_inference else []),
            *(["--spatial-range-association"] if args.spatial_range_association else []),
            *(["--sam2-range-association"] if args.sam2_range_association else []),
            *(["--far-conversation-viewpoint"] if args.far_conversation_viewpoint else []),
            *(["--depth-separated-overlap"] if args.depth_separated_overlap else []),
            *(["--receding-owner-target"] if args.receding_owner_target else [])],
            cwd=ROOT, capture_output=True, text=True, timeout=300)
        row = {"scene_id": placement["scene_id"], "name": placement["name"],
               "placement_status": placement["status"],
               "runtime_wall_s": time.perf_counter() - started,
               "resumed_existing": resumed}
        if result is not None and result.returncode:
            row.update(task_status="runner_failed", exit_code=result.returncode,
                       error_tail=result.stderr[-1500:])
        else:
            # Open3D's native renderer may print device diagnostics on stdout.
            # The saved trace is the complete authoritative report.
            report = json.loads(trace_path.read_text())
            trace = report
            frames = video_frame_count(folder / "first_person.mp4")
            media_aligned = bool(frames == trace["frames"] ==
                                 len(list(folder.glob("rgb_*.png"))) ==
                                 len(list(folder.glob("depth_*.png"))))
            if not media_aligned:
                raise RuntimeError(f"Unaligned video/sensors for {scene}")
            monitor_frames = None
            if args.fast_revalidate_during_inference:
                ticks = sum(len(f["fast_revalidation"]) for f in trace["trace"])
                monitor_frames = video_frame_count(folder / "inference_monitor.mp4")
                if not (monitor_frames == frames + ticks ==
                        frames + len(list(folder.glob("fast_rgb_*.png"))) ==
                        frames + len(list(folder.glob("fast_depth_*.png")))):
                    raise RuntimeError(f"Unaligned fast monitor video/sensors for {scene}")
            row.update(task_status="completed", outcome=report["outcome"],
                       stop_reason=trace["trace"][-1]["planner_decision"].get("reason"),
                       frames=frames, accepted_travel_m=report["accepted_travel_m"],
                       observed_standoff_band_met=report["observed_standoff_band_met"],
                       observed_far_conversation_band_met=report.get(
                           "observed_far_conversation_band_met", False),
                       max_pose_error_m=report["max_pose_error_m"],
                       max_actor_pose_disagreement_m=report["max_actor_pose_disagreement_m"],
                       median_inference_wall_s=float(np.median([
                           f["inference_wall_s"] for f in trace["trace"]])),
                       monitor_frames=monitor_frames,
                       monitor_video=(str((folder / "inference_monitor.mp4").relative_to(ROOT))
                                      if monitor_frames is not None else None),
                       video=str((folder / "first_person.mp4").relative_to(ROOT)),
                       trace=str(trace_path.relative_to(ROOT)))
        rows.append(row)
        out.write_text(json.dumps({"schema": "ape.scanned-owner-batch.v1",
            "placement_screen": str(screen.relative_to(ROOT)),
            "pose_mode": args.pose_mode,
            "run_label": args.run_label,
            "native_source": (args.native_source if args.pose_mode in
                              ("native_shaped", "rgbd_native_fallback") else None),
            "pose_crosscheck": crosscheck,
            "person_masked_odometry": args.person_mask_odometry,
            "simulated_inference_delay_s": args.simulated_inference_delay_s,
            "fast_revalidate_during_inference": args.fast_revalidate_during_inference,
            "spatial_range_association": args.spatial_range_association,
            "sam2_range_association": args.sam2_range_association,
            "far_conversation_viewpoint": args.far_conversation_viewpoint,
            "receding_owner_target": args.receding_owner_target,
            "depth_separated_overlap": args.depth_separated_overlap,
            "scene_count": len(placements),
            "started_count": sum(r["placement_status"] == "provisional_supported_visible" for r in rows),
            "completed_count": sum(r["task_status"] == "completed" for r in rows),
            "viewpoint_count": sum(r.get("observed_standoff_band_met", False) or
                                   r.get("observed_far_conversation_band_met", False)
                                   for r in rows),
            "outcome_counts": dict(Counter(r.get("outcome", r["task_status"]) for r in rows)),
            "scenes": rows}, indent=2) + "\n")
        print(row["task_status"], row.get("outcome", ""),
              row.get("stop_reason", ""), flush=True)
    print(out)


if __name__ == "__main__":
    main()
