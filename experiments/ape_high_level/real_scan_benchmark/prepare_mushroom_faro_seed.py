"""Put one room's Faro reference points in the iPhone camera metric frame."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import open3d as o3d


ROOT = Path(__file__).resolve().parents[3] / "assets/real_indoor_scenes/mushroom"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--room", required=True)
    parser.add_argument("--points", type=int, default=1_000_000)
    args = parser.parse_args()
    room = ROOT / args.room / "extracted" / "room_datasets" / args.room
    output = room / "iphone/long_capture/faro_aligned_pointcloud.ply"
    # The dataset's gt_transformation maps iPhone coordinates to Faro. Camera
    # poses and depth in transformations_colmap.json are in iPhone coordinates.
    transform = np.linalg.inv(np.asarray(json.loads((room / "icp_iphone.json").read_text())["gt_transformation"]))
    cloud = o3d.io.read_point_cloud(str(room / "gt_pd.ply"))
    if not cloud.has_colors():
        raise ValueError("Faro reference point cloud has no seed colours")
    cloud.transform(transform)
    rng = np.random.default_rng(50)
    index = rng.choice(len(cloud.points), size=args.points, replace=False)
    seed = cloud.select_by_index(index.tolist())
    if not o3d.io.write_point_cloud(str(output), seed, write_ascii=False, compressed=False):
        raise RuntimeError(f"Could not write {output}")
    z = np.asarray(seed.points)[:, 2]
    print(f"{len(seed.points)} aligned Faro seed points; z 1–99% [{np.quantile(z, .01):.3f}, {np.quantile(z, .99):.3f}] m")
    print(output)


if __name__ == "__main__":
    main()
