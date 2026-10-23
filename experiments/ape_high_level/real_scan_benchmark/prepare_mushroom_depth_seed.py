"""Make a camera-aligned splat seed from MuSHRoom's measured iPhone RGB-D frames.

The upstream Polycam point cloud is stored in a different coordinate frame from
the released COLMAP camera poses. Back-projecting the registered depth avoids
silently training a model with that inconsistent initialization. Only frames
outside the room's test split contribute points.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import open3d as o3d
from PIL import Image


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--room-dir", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--max-points", type=int, default=1_000_000)
    parser.add_argument("--seed", type=int, default=19)
    args = parser.parse_args()
    data = args.room_dir / "iphone/long_capture"
    scene = json.loads((data / "transformations_colmap.json").read_text())
    test = set((data / "test.txt").read_text().splitlines())
    frames = [f for f in scene["frames"] if Path(f["file_path"]).stem not in test]
    rng = np.random.default_rng(args.seed)
    per_frame = max(1, args.max_points // len(frames))
    points, colours = [], []

    for frame in frames:
        depth = np.asarray(Image.open(data / frame["depth_file_path"]), dtype=np.float32) / 1000.0
        rgb = np.asarray(Image.open(data / frame["file_path"]).convert("RGB"))
        if depth.shape != rgb.shape[:2]:
            raise ValueError(f"RGB/depth size mismatch: {frame['file_path']}")
        valid = np.flatnonzero((depth.ravel() > 0.2) & (depth.ravel() < 8.0))
        if len(valid) == 0:
            continue
        choice = rng.choice(valid, size=min(per_frame, len(valid)), replace=False)
        v, u = np.divmod(choice, depth.shape[1])
        z = depth.ravel()[choice]
        fx = float(frame.get("fl_x", scene["fl_x"]))
        fy = float(frame.get("fl_y", scene["fl_y"]))
        cx = float(frame.get("cx", scene["cx"]))
        cy = float(frame.get("cy", scene["cy"]))
        camera = np.stack(((u - cx) * z / fx, -(v - cy) * z / fy, -z), axis=1)
        pose = np.asarray(frame["transform_matrix"], dtype=np.float64)
        points.append(camera @ pose[:3, :3].T + pose[:3, 3])
        colours.append(rgb.reshape(-1, 3)[choice].astype(np.float32) / 255.0)

    xyz = np.concatenate(points)
    color = np.concatenate(colours)
    cloud = o3d.geometry.PointCloud()
    cloud.points = o3d.utility.Vector3dVector(xyz)
    cloud.colors = o3d.utility.Vector3dVector(color)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    if not o3d.io.write_point_cloud(str(args.output), cloud, write_ascii=False):
        raise RuntimeError(f"Could not write {args.output}")
    print(json.dumps({"output": str(args.output), "training_frames": len(frames),
                      "points": len(xyz), "bounds_m": [xyz.min(0).tolist(), xyz.max(0).tolist()]}, indent=2))


if __name__ == "__main__":
    main()
