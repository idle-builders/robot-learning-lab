"""Render a short moving RGB-D camera probe at a frozen owner placement."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import imageio.v2 as imageio
import numpy as np
from PIL import Image

from real_capture_dynamic_scene import TexturedCaptureScene


ROOT = Path(__file__).resolve().parents[3]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--screen", type=Path, required=True)
    parser.add_argument("--distance-m", type=float, default=.8)
    parser.add_argument("--frames", type=int, default=9)
    args = parser.parse_args()
    screen_path = args.screen.resolve()
    placement = json.loads(screen_path.read_text())
    source = json.loads((ROOT / placement["source_trace"]).read_text())
    if placement["status"] != "provisional_supported_visible":
        raise ValueError("No supported placement")
    scene = TexturedCaptureScene(
        ROOT / source["source_mesh"], floor_y=source["floor_y_1pct_m"],
        scale=source["scale_to_metres"], source_z_up=source["source_z_up"])
    heading = math.radians(placement["start_heading_deg"])
    forward = np.array([math.sin(heading), 0., math.cos(heading)])
    start = np.array([placement["start_xz"][0],
                      source["floor_y_1pct_m"] + .42,
                      placement["start_xz"][1]])
    output = screen_path.parent
    rgb_frames = []
    rows = []
    for index in range(args.frames):
        travel = args.distance_m * index / max(1, args.frames - 1)
        eye = start + travel * forward
        rgb, depth = scene.observe(eye, forward, owner_xz=None,
                                   width=640, height=360)
        rgb_frames.append(rgb)
        rows.append({"frame": index, "travel_m": travel,
                     "camera_xyz_m": eye.tolist(),
                     "valid_depth_fraction": float(np.mean(np.isfinite(depth))),
                     "near_black_fraction": float(np.mean(np.max(rgb, axis=-1) < 8))})
    stem = f"static_approach_{args.distance_m:.1f}m"
    video = output / f"{stem}.mp4"
    imageio.mimsave(video, rgb_frames, fps=5, codec="libx264", quality=8,
                    macro_block_size=None)
    triplet = output / f"{stem}_triplet.jpg"
    Image.fromarray(np.concatenate([rgb_frames[0], rgb_frames[len(rgb_frames)//2],
                                    rgb_frames[-1]], axis=1)).save(triplet, quality=90)
    trace = output / f"{stem}_trace.json"
    trace.write_text(json.dumps({
        "schema": "ape.supported-static-route.v1",
        "source_screen": str(screen_path.relative_to(ROOT)),
        "distance_m": args.distance_m, "frames": rows,
        "video": str(video.relative_to(ROOT)),
        "triplet": str(triplet.relative_to(ROOT)),
    }, indent=2) + "\n")
    print(video, len(rows), "min_depth_valid",
          min(row["valid_depth_fraction"] for row in rows))


if __name__ == "__main__":
    main()
