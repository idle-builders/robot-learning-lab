"""Pair the six new/bridged owner routes with upright RGB-D projected range."""

from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

import cv2
from pack_rgb_phase import pack, sha256


ROOT = Path(__file__).resolve().parents[3]
HERE = Path(__file__).resolve().parent
ARTIFACTS = ROOT / "artifacts/ape-real-room-next-captures-20261014"
NEW = ROOT / "assets/real_indoor_scenes/scansceneverse/new_capture_owner_20261014"
BRIDGE = ROOT / "assets/real_indoor_scenes/scansceneverse/dynamic_placement_screen/common_policy_bridge_20261014"
ROOMS = (
    ("b61ae63b60484b7fa0a44eb2f1313f75", "capture_b61ae63b60484b7fa0a44eb2f1313f75", NEW),
    ("9a0aca1b562a4e3a8a8b0bf56861a447", "capture_9a0aca1b562a4e3a8a8b0bf56861a447", NEW),
    ("538d390577164a15a3b180e17a37679a", "capture_538d390577164a15a3b180e17a37679a", NEW),
    ("d7b8d3c06b1d4b928c7c30f9b9a66026", "capture_d7b8d3c06b1d4b928c7c30f9b9a66026", NEW),
    ("0502edd28d574a0a856ac4c582a93029", "capture_0502edd28d574a0a856ac4c582a93029", NEW),
    ("e7d97782c31448bb83418f01a5b8b4b3", "capture_restaurant_kitchen_south_0p8m", BRIDGE),
)


def aligned(folder: Path) -> bool:
    trace = json.loads((folder / "trace.json").read_text())
    video = cv2.VideoCapture(str(folder / "first_person.mp4"))
    frames = 0
    while video.read()[0]:
        frames += 1
    video.release()
    if not (frames > 0 and frames == trace["frames"] == len(list(folder.glob("depth_*.png")))):
        return False
    if len(list(folder.glob("rgb_*.png"))) == frames:
        return True
    manifest = folder / "rgb_frames_manifest.json"
    archive = folder / "rgb_frames.tar.gz"
    if not manifest.is_file() or not archive.is_file():
        return False
    record = json.loads(manifest.read_text())
    return record["frame_count"] == frames and record["archive_sha256"] == sha256(archive)


def relative(path: Path) -> str:
    return str(path.relative_to(ROOT))


def main() -> None:
    precommit = ARTIFACTS / "six_independent_capture_range_ablation_precommit.json"
    frozen = json.loads(precommit.read_text())
    assert len(ROOMS) == len(frozen["capture_groups"]) == 6
    rows = []
    for capture_id, scene_key, base in ROOMS:
        assert f"scansceneverse/{capture_id}" in frozen["capture_groups"]
        room = base / (capture_id if base == NEW else scene_key)
        control = room / "owner_approach"
        candidate = room / "owner_rgbd_projected"
        if not (candidate / "trace.json").is_file():
            command = [sys.executable, str(HERE / "run_real_capture_owner_loop.py"),
                       "--scene", scene_key, "--pose-mode", "rgbd_yaw",
                       "--range-mode", "rgbd_projection", "--person-style", "scanned",
                       "--steps", "60", "--device", "mps", "--gaze-route-arbitration",
                       "--pose-crosscheck", "--person-mask-odometry",
                       "--simulated-inference-delay-s", "0.85", "--spatial-range-association",
                       "--far-conversation-viewpoint", "--depth-separated-overlap",
                       "--output-dir", str(candidate)]
            env = {**os.environ, "PYTORCH_ENABLE_MPS_FALLBACK": "1"}
            process = subprocess.run(command, cwd=ROOT, env=env, capture_output=True,
                                     text=True, timeout=300)
            if process.returncode:
                raise RuntimeError(f"{scene_key}: {process.stderr[-1200:]}")
        packed = pack(candidate)
        traces = {arm: json.loads((folder / "trace.json").read_text())
                  for arm, folder in (("independent_two_height", control),
                                      ("upright_rgbd_projected", candidate))}
        row = {"capture_group": f"scansceneverse/{capture_id}", "scene_key": scene_key}
        for arm, trace in traces.items():
            folder = control if arm == "independent_two_height" else candidate
            row[arm] = {"outcome": trace["outcome"],
                        "observed_far_view": trace["observed_far_conversation_band_met"],
                        "max_capture_pose_error_m": trace["max_pose_error_at_capture_m"],
                        "accepted_travel_m": trace["accepted_travel_m"],
                        "frames": trace["frames"], "media_aligned": aligned(folder),
                        "trace": relative(folder / "trace.json"),
                        "video": relative(folder / "first_person.mp4")}
            if arm == "upright_rgbd_projected":
                row[arm]["lossless_rgb_archive"] = relative(candidate / packed["archive"])
        rows.append(row)
        print(scene_key, row["independent_two_height"]["outcome"],
              row["upright_rgbd_projected"]["outcome"], flush=True)
    report = {"schema": "ape.six-capture-range-ablation.v1",
              "precommit": relative(precommit), "capture_group_count": len(rows),
              "summary": {arm: {
                  "far_views": sum(r[arm]["outcome"] == "hold_viewpoint_far" and
                                   r[arm]["observed_far_view"] for r in rows),
                  "max_capture_pose_error_m": max(r[arm]["max_capture_pose_error_m"]
                                                  for r in rows),
                  "all_media_aligned": all(r[arm]["media_aligned"] for r in rows),
              } for arm in ("independent_two_height", "upright_rgbd_projected")},
              "limits": frozen["limits"], "rooms": rows}
    out = ARTIFACTS / "six_independent_capture_range_ablation_report.json"
    out.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report["summary"], indent=2))
    print(relative(out))


if __name__ == "__main__":
    main()
