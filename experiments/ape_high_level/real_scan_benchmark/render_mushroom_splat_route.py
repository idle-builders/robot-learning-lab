"""Render one real-scan splat along a Faro-screened 0.42 m ape-camera route."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import cv2
import imageio.v2 as imageio
import numpy as np
import open3d as o3d


ROOT = Path(__file__).resolve().parents[3] / "assets/real_indoor_scenes/mushroom"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--room", required=True)
    parser.add_argument("--splat", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--test-list", type=Path)
    parser.add_argument("--max-scale-m", type=float)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    route = json.loads((ROOT / args.room / "proxy_route.json").read_text())
    xy = np.asarray(route["route_xy_m"], dtype=float)
    if len(xy) < 2 or route["route_length_m"] < 1.0:
        raise ValueError("No meaningful Faro-screened route")
    floor = route["floor_z_m"]
    data = ROOT / args.room / "extracted" / "room_datasets" / args.room / "iphone/long_capture"
    scene = json.loads((data / "transformations_colmap.json").read_text())
    test = set((args.test_list or data / "test.txt").read_text().splitlines())
    training_centers = np.asarray([np.asarray(f["transform_matrix"])[:3, 3] for f in scene["frames"]
                                   if Path(f["file_path"]).stem not in test])
    splats = o3d.t.io.read_point_cloud(str(args.splat))
    for key in ("positions", "f_dc", "opacity", "scale", "rot"):
        if key not in splats.point:
            raise ValueError(f"Missing Gaussian property {key}")
    if args.max_scale_m is not None:
        keep = np.max(splats.point.scale.numpy(), axis=1) <= args.max_scale_m
        splats = splats.select_by_mask(o3d.core.Tensor(keep))
    opacity = splats.point.opacity.numpy()
    splats.point.opacity = o3d.core.Tensor((1 / (1 + np.exp(-np.clip(opacity, -30, 30)))).astype(np.float32))
    renderer = o3d.visualization.rendering.OffscreenRenderer(640, 480)
    renderer.scene.set_background([0, 0, 0, 1])
    material = o3d.visualization.rendering.MaterialRecord()
    material.shader = "gaussianSplat"
    material.gaussian_splat_sh_degree = 3
    material.gaussian_splat_max_tiles_per_splat = 4096
    renderer.scene.add_geometry("room", splats, material)

    lengths = np.r_[0.0, np.cumsum(np.linalg.norm(np.diff(xy, axis=0), axis=1))]
    total = float(lengths[-1])
    frames = max(2, math.ceil(total / 0.06) + 1)
    samples = np.linspace(0, total, frames)
    positions = np.stack([np.interp(samples, lengths, xy[:, i]) for i in range(2)], axis=1)
    trace = []
    writer = imageio.get_writer(str(args.output / "ape_height_route.mp4"), fps=8,
                                codec="libx264", quality=7, macro_block_size=8)
    try:
        for index, (distance, point) in enumerate(zip(samples, positions)):
            before = np.array([np.interp(max(0, distance - .12), lengths, xy[:, i]) for i in range(2)])
            after = np.array([np.interp(min(total, distance + .12), lengths, xy[:, i]) for i in range(2)])
            direction = after - before
            direction /= np.linalg.norm(direction)
            eye = np.r_[point, floor + 0.42]
            forward = np.r_[direction, 0]
            renderer.setup_camera(75.0, eye + forward, eye, [0, 0, 1])
            rgb = np.asarray(renderer.render_to_image())[:, :, :3].copy()
            writer.append_data(rgb)
            if index in (0, frames // 2, frames - 1):
                cv2.imwrite(str(args.output / f"frame_{index:03d}.png"), cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR))
            trace.append({"frame": index, "travel_m": float(distance), "eye_world_m": eye.tolist(),
                          "forward_world": forward.tolist(),
                          "nearest_training_camera_m": float(np.linalg.norm(training_centers - eye, axis=1).min()),
                          "nonblack_fraction": float(np.mean(np.max(rgb, axis=2) > 8))})
    finally:
        writer.close()
    summary = {"schema": "ape.mushroom-splat-ape-height-route.v1", "room": args.room,
               "splat": str(args.splat), "gaussians": int(splats.point.positions.shape[0]),
               "max_scale_m": args.max_scale_m,
               "source": "real MuSHRoom iPhone RGB-D and Faro scan",
               "camera_height_above_depth_floor_m": .42, "route_length_m": total,
               "frames": trace, "qualification": "appearance probe only; no owner task or selected-body motion"}
    (args.output / "trace.json").write_text(json.dumps(summary, indent=2) + "\n")
    print({"room": args.room, "route_m": total, "frames": frames,
           "minimum_nonblack_fraction": min(x["nonblack_fraction"] for x in trace),
           "nearest_training_camera_median": float(np.median([x["nearest_training_camera_m"] for x in trace])),
           "video": str(args.output / "ape_height_route.mp4")})


if __name__ == "__main__":
    main()
