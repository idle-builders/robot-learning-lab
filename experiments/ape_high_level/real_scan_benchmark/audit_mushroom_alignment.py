"""Compare registered iPhone depth with the transformed Faro reference mesh."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import open3d as o3d
from PIL import Image


ROOT = Path(__file__).resolve().parents[3] / "assets/real_indoor_scenes/mushroom"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--room", required=True)
    args = parser.parse_args()
    room = args.room
    base = ROOT / room / "extracted/room_datasets" / room
    iphone = base / "iphone/long_capture"
    transform = np.linalg.inv(np.asarray(json.loads((base / "icp_iphone.json").read_text())["gt_transformation"]))
    mesh = o3d.io.read_triangle_mesh(str(base / "gt_mesh.ply"))
    mesh.transform(transform)
    rays = o3d.t.geometry.RaycastingScene()
    rays.add_triangles(o3d.t.geometry.TriangleMesh.from_legacy(mesh))
    metadata = json.loads((iphone / "transformations_colmap.json").read_text())
    frames = {Path(f["file_path"]).name: f for f in metadata["frames"]}
    screen = json.loads((ROOT / room / "metric_low_view_screen.json").read_text())
    names = screen["floor_estimation_frames"]
    chosen = [names[0], names[len(names)//2], names[-1]]
    results = []
    for name in chosen:
        frame = frames[name]
        depth = np.asarray(Image.open(iphone / frame["depth_file_path"]), dtype=np.float32)[::12, ::12] / 1000
        h, w = Image.open(iphone / frame["depth_file_path"]).size[::-1]
        v, u = np.mgrid[0:h:12, 0:w:12]
        cx = frame.get("cx", metadata.get("cx"))
        cy = frame.get("cy", metadata.get("cy"))
        fx = frame.get("fl_x", metadata.get("fl_x"))
        fy = frame.get("fl_y", metadata.get("fl_y"))
        direction_cv = np.stack(((u - cx)/fx, -(v - cy)/fy, -np.ones_like(u)), axis=-1).reshape(-1, 3)
        pose = np.asarray(frame["transform_matrix"])
        direction_world = direction_cv @ pose[:3, :3].T
        origin = np.broadcast_to(pose[:3, 3], direction_world.shape)
        ray = np.concatenate((origin, direction_world), axis=1).astype(np.float32)
        predicted = rays.cast_rays(o3d.core.Tensor(ray))["t_hit"].numpy()
        observed = depth.ravel()
        valid = np.isfinite(predicted) & (observed > 0.2) & (observed < 8)
        error = np.abs(predicted[valid] - observed[valid])
        results.append({"frame": name, "valid_fraction": float(np.mean(valid)),
                        "median_absolute_depth_error_m": float(np.median(error)),
                        "p90_absolute_depth_error_m": float(np.quantile(error, .9)),
                        "samples": int(np.sum(valid))})
    result = {"schema": "ape.mushroom-faro-depth-alignment.v1", "room": room,
              "faro_to_iphone_transform": "inverse icp_iphone.json gt_transformation",
              "views": results}
    output = ROOT / room / "faro_depth_alignment.json"
    output.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
