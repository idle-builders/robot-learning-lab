"""Replay saved Isaac RGB-D and commands; use proxy pose only for scoring."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import cv2
import numpy as np

from rgbd_visual_odometry import RgbdVisualOdometry


def replay(run: Path, fov: int, scale: float, depth_available: bool,
           *, minimum_radial_depth_m: float = .3,
           minimum_inlier_fraction: float = .55,
           minimum_inliers: int = 45) -> dict:
    evidence = run / "evidence"
    video = evidence / f"ape_social_{fov}deg.mp4"
    depth = evidence / f"ape_social_{fov}deg-depth-sequence-stride3.npz"
    actions_path = evidence / "online-action-trace.json"
    scene_path = evidence / "scene-info.json"
    depths = np.load(depth)["depth_m"]
    actions = json.loads(actions_path.read_text())["actions"]
    initial = json.loads(scene_path.read_text())["camera_eye_xyz_m"][:2]
    cap = cv2.VideoCapture(str(video))
    actor = RgbdVisualOdometry(np.asarray(initial), fov, odometry_scale=scale,
        minimum_radial_depth_m=minimum_radial_depth_m,
        minimum_inlier_fraction=minimum_inlier_fraction,
        minimum_inliers=minimum_inliers)
    rows = []
    for frame, radial_depth in enumerate(depths):
        ok, bgr = cap.read()
        if not ok or frame >= len(actions):
            raise ValueError("RGB/depth/action streams are not aligned")
        rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
        previous_command = actions[frame - 1]["command_for_next_frame"] if frame else None
        estimate = actor.update(rgb, radial_depth if depth_available else
                                np.full_like(radial_depth, np.nan), previous_command)
        truth = (np.asarray(actions[frame - 1]["proxy_eye_xyz_m_evaluator_only"][:2])
                 if frame else np.asarray(initial))
        estimate["frame"] = frame
        estimate["error_m_evaluator_only"] = float(np.linalg.norm(actor.xy - truth))
        rows.append(estimate)
    if cap.read()[0] or len(rows) != len(actions):
        raise ValueError("RGB/depth/action streams have different lengths")
    return {"run": str(run), "fov_deg": fov, "odometry_scale": scale,
            "depth_available": depth_available,
            "minimum_radial_depth_m": minimum_radial_depth_m,
            "minimum_inlier_fraction": minimum_inlier_fraction,
            "minimum_inliers": minimum_inliers,
            "final_error_m": rows[-1]["error_m_evaluator_only"],
            "max_error_m": max(row["error_m_evaluator_only"] for row in rows),
            "visual_registered": sum(row["status"] == "visual_registered" for row in rows),
            "odometry_fallback": sum(row["status"] == "odometry_fallback" for row in rows),
            "sha256": {path.name: hashlib.sha256(path.read_bytes()).hexdigest()
                       for path in (video, depth, actions_path, scene_path)},
            "frames": rows}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("runs", nargs="+", type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    report = {"scope": "Saved-packet actor-input visual odometry diagnostic; kinematic eye, calibrated ideal radial depth, no planner feedback or physical body",
              "results": [replay(run, fov, scale, depth_available)
                          for run in args.runs for fov in (70, 90)
                          for scale in (1.0, .8)
                          for depth_available in (True, False)]}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n")
    for result in report["results"]:
        print(Path(result["run"]).name, result["fov_deg"], result["odometry_scale"],
              "depth", result["depth_available"],
              "final_cm", round(100 * result["final_error_m"], 1),
              "accepted", result["visual_registered"],
              "fallback", result["odometry_fallback"])


if __name__ == "__main__":
    main()
