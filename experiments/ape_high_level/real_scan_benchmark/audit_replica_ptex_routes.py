"""Summarize actual ape-height Replica RGB and paired Habitat-depth coverage."""

import argparse
import json
from pathlib import Path

import numpy as np
from PIL import Image


SCENES = (
    "apartment_0", "apartment_1", "apartment_2",
    "frl_apartment_0", "frl_apartment_1", "frl_apartment_2",
    "frl_apartment_3", "frl_apartment_4", "frl_apartment_5",
    "hotel_0", "office_0", "office_1", "office_2", "office_3", "office_4",
    "room_0", "room_1", "room_2",
)


def audit(root):
    rows = []
    for scene in SCENES:
        folder = root / scene
        trace_path = folder / "trace.json"
        if not trace_path.is_file():
            rows.append({"scene_id": scene, "status": "missing_trace"})
            continue
        trace = json.loads(trace_path.read_text())
        frames = trace["frames"]
        if not trace.get("geometry_frame", "").startswith("Source PLY explicitly rotated"):
            rows.append({"scene_id": scene, "status": "unverified_geometry_frame"})
            continue
        black = []
        depth_valid = []
        for frame in frames:
            path = folder / frame["image"]
            depth_path = folder / frame.get("depth_mm_png", "missing_depth")
            if not path.is_file() or not depth_path.is_file():
                break
            rgb = np.asarray(Image.open(path).convert("RGB"))
            depth = np.asarray(Image.open(depth_path))
            if depth.dtype != np.uint16 or depth.shape != rgb.shape[:2]:
                break
            valid_fraction = float((depth > 0).mean())
            if abs(valid_fraction - frame["valid_depth_fraction"]) > 0.005:
                break
            black.append(float((rgb.mean(axis=-1) < 10).mean()))
            depth_valid.append(valid_fraction)
        else:
            rows.append({
                "scene_id": scene,
                "physical_space_id": "frl_apartment" if scene.startswith("frl_apartment_") else scene,
                "source": "https://github.com/facebookresearch/Replica-Dataset",
                "appearance": "Official Replica HDR PTex renderer",
                "camera_height_m": trace["camera_height_above_nav_m"],
                "route_m": trace["rendered_path_polyline_m"],
                "frame_count": len(frames),
                "rgb_black_mean": float(np.mean(black)),
                "rgb_black_max": float(np.max(black)),
                "rgb_black_frames_gt_10pct": sum(value > 0.10 for value in black),
                "paired_mesh_depth_valid_mean": float(np.mean(depth_valid)),
                "paired_mesh_depth_valid_min": float(np.min(depth_valid)),
                "metric_depth_frames_complete": True,
                "video": str(folder / "ape_height_route.mp4"),
                "status": "rendered" if (folder / "ape_height_route.mp4").is_file() else "missing_video",
            })
            continue
        rows.append({"scene_id": scene, "status": "missing_or_invalid_frame_or_depth",
                     "frame": frame["frame"]})
    return rows


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("routes", type=Path)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    rows = audit(args.routes)
    args.output.write_text(json.dumps({"schema": "ape.replica-ptex-route-audit.v1", "scenes": rows}, indent=2) + "\n")
    for row in rows:
        print(row["scene_id"], row["status"],
              "black=%.3f" % row["rgb_black_mean"] if row["status"] == "rendered" else "")


if __name__ == "__main__":
    main()
