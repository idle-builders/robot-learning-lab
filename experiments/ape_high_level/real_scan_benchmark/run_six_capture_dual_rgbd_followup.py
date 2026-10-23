"""Compare an added downward RGB-D view on six frozen owner routes."""

from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

from pack_rgb_phase import pack, sha256, video_frames
from run_six_capture_range_ablation import ROOT, ARTIFACTS, ROOMS, relative


PRECOMMIT = ARTIFACTS / "six_capture_dual_rgbd_followup_precommit.json"
CONTROL = ARTIFACTS / "six_independent_capture_range_ablation_report.json"
OUT = ARTIFACTS / "six_capture_dual_rgbd_followup_report.json"
HERE = Path(__file__).resolve().parent


def aligned(folder: Path, trace: dict) -> bool:
    frames = trace["frames"]
    for prefix, video, depth in (("rgb", "first_person.mp4", "depth"),
                                 ("down_rgb", "downward_view.mp4", "down_depth")):
        manifest = folder / f"{prefix}_frames_manifest.json"
        archive = folder / f"{prefix}_frames.tar.gz"
        if video_frames(folder / video) != frames or len(list(folder.glob(f"{depth}_*.png"))) != frames:
            return False
        if not manifest.is_file() or not archive.is_file():
            return False
        record = json.loads(manifest.read_text())
        if record["frame_count"] != frames or record["archive_sha256"] != sha256(archive):
            return False
    return True


def main() -> None:
    frozen = json.loads(PRECOMMIT.read_text())
    control = json.loads(CONTROL.read_text())
    controls = {r["capture_group"]: r for r in control["rooms"]}
    assert len(ROOMS) == len(frozen["capture_groups"]) == len(controls) == 6
    rows = []
    for capture_id, scene_key, base in ROOMS:
        group = f"scansceneverse/{capture_id}"
        assert group in frozen["capture_groups"] and group in controls
        room = base / (capture_id if base.name == "new_capture_owner_20261014" else scene_key)
        candidate = room / "owner_dual_rgbd_projected"
        if not (candidate / "trace.json").is_file():
            command = [sys.executable, str(HERE / "run_real_capture_owner_loop.py"),
                       "--scene", scene_key, "--pose-mode", "rgbd_yaw",
                       "--range-mode", "dual_rgbd_projection", "--person-style", "scanned",
                       "--steps", "60", "--device", "mps", "--gaze-route-arbitration",
                       "--pose-crosscheck", "--person-mask-odometry",
                       "--simulated-inference-delay-s", "0.85", "--spatial-range-association",
                       "--far-conversation-viewpoint", "--depth-separated-overlap",
                       "--output-dir", str(candidate)]
            result = subprocess.run(command, cwd=ROOT,
                                    env={**os.environ, "PYTORCH_ENABLE_MPS_FALLBACK": "1"},
                                    capture_output=True, text=True, timeout=600)
            if result.returncode:
                raise RuntimeError(f"{scene_key}: {result.stderr[-1500:]}")
        upright = pack(candidate)
        downward = pack(candidate, prefix="down_rgb", video_name="downward_view.mp4",
                        depth_prefix="down_depth")
        trace = json.loads((candidate / "trace.json").read_text())
        first = trace["trace"][0]["planner_decision"]
        row = {"capture_group": group, "scene_key": scene_key,
               "independent_range_far_view": controls[group]["independent_two_height"]["observed_far_view"],
               "upright_rgbd_far_view": controls[group]["upright_rgbd_projected"]["observed_far_view"],
               "dual_rgbd": {"outcome": trace["outcome"],
                             "observed_far_view": trace["observed_far_conversation_band_met"],
                             "frames": trace["frames"],
                             "accepted_travel_m": trace["accepted_travel_m"],
                             "max_capture_pose_error_m": trace["max_pose_error_at_capture_m"],
                             "first_plan_reason": first.get("reason"),
                             "first_reachable_cells": first.get("reachable_cells"),
                             "media_aligned": aligned(candidate, trace),
                             "upright_video": relative(candidate / "first_person.mp4"),
                             "downward_video": relative(candidate / "downward_view.mp4"),
                             "trace": relative(candidate / "trace.json"),
                             "upright_rgb_archive": relative(candidate / upright["archive"]),
                             "downward_rgb_archive": relative(candidate / downward["archive"])}}
        rows.append(row)
        report = {"schema": "ape.six-capture-dual-rgbd-followup-report.v1",
                  "precommit": relative(PRECOMMIT), "control_report": relative(CONTROL),
                  "completed_group_count": len(rows),
                  "summary": {"independent_range_far_views": sum(r["independent_range_far_view"] for r in rows),
                              "upright_rgbd_far_views": sum(r["upright_rgbd_far_view"] for r in rows),
                              "dual_rgbd_far_views": sum(r["dual_rgbd"]["observed_far_view"] for r in rows),
                              "all_dual_media_aligned": all(r["dual_rgbd"]["media_aligned"] for r in rows)},
                  "limits": frozen["limits"], "rooms": rows}
        OUT.write_text(json.dumps(report, indent=2) + "\n")
        print(scene_key, trace["outcome"], trace["frames"], trace["accepted_travel_m"],
              flush=True)
    print(json.dumps(report["summary"], indent=2))


if __name__ == "__main__":
    main()
