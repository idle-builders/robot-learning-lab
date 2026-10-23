"""Compare on-demand mesh rays with saved Replica RGB-D route depth."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import numpy as np
import open3d as o3d
import trimesh
from PIL import Image


def ray_depth(scene, position, yaw, eye_height, shape):
    height, width = shape
    eye = np.asarray(position, np.float32) + [0., eye_height, 0.]
    forward = np.array([-math.sin(yaw), 0., -math.cos(yaw)], np.float32)
    right = np.array([math.cos(yaw), 0., -math.sin(yaw)], np.float32)
    up = np.array([0., 1., 0.], np.float32)
    xx, yy = np.meshgrid(np.arange(width), np.arange(height))
    focal = width / 2.
    rays = (forward + ((xx - (width-1)/2.) / focal)[..., None] * right
            + (((height-1)/2. - yy) / focal)[..., None] * up)
    rays /= np.linalg.norm(rays, axis=-1, keepdims=True)
    origins = np.broadcast_to(eye, rays.shape)
    packet = np.concatenate((origins, rays), axis=-1).astype(np.float32)
    hit = scene.cast_rays(o3d.core.Tensor(packet))["t_hit"].numpy()
    optical = hit * np.sum(rays * forward, axis=-1)
    optical[~np.isfinite(hit) | (optical <= 0) | (optical > 10.)] = np.nan
    return optical


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--converted-mesh", type=Path, required=True)
    p.add_argument("--reference", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    args = p.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    mesh = trimesh.load(args.converted_mesh, process=False)
    if not isinstance(mesh, trimesh.Trimesh):
        raise ValueError("Replica aligned mesh is not one triangle mesh")
    scene = o3d.t.geometry.RaycastingScene()
    scene.add_triangles(o3d.core.Tensor(np.asarray(mesh.vertices, np.float32)),
                        o3d.core.Tensor(np.asarray(mesh.faces, np.uint32)))
    trace = json.loads((args.reference / "trace.json").read_text())
    frames = trace["frames"]
    rows = []
    for index in sorted(set((0, len(frames)//2, len(frames)-1))):
        frame = frames[index]
        reference = np.asarray(Image.open(args.reference /
            frame["depth_mm_png"]), np.uint16).astype(float) / 1000.
        predicted = ray_depth(scene, frame["position_m"], frame["yaw_rad"],
            trace["camera_height_above_nav_m"], reference.shape)
        valid = np.isfinite(predicted) & (reference > .1) & (reference < 6.)
        difference = np.abs(predicted[valid] - reference[valid])
        if not difference.size:
            raise ValueError("No aligned valid depth pixels")
        saved = np.where(np.isfinite(predicted),
                         np.clip(np.rint(predicted * 1000.), 0, 65535), 0)
        Image.fromarray(saved.astype(np.uint16)).save(args.output /
            f"predicted_depth_{index:04d}.png")
        rows.append({"frame": index, "valid_overlap_fraction": float(valid.mean()),
            "median_abs_depth_error_m": float(np.median(difference)),
            "p90_abs_depth_error_m": float(np.percentile(difference, 90)),
            "fraction_over_5cm_on_overlap": float((difference > .05).mean())})
    result = {"schema": "ape.replica-source-depth-probe.v1",
        "source": str(args.converted_mesh), "reference": str(args.reference),
        "mesh_vertices": len(mesh.vertices), "mesh_triangles": len(mesh.faces),
        "rows": rows,
        "limits": "Same recorded route poses; source mesh rays are an ideal depth/lidar proxy, not a measured sensor or a closed owner task."}
    (args.output / "report.json").write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
