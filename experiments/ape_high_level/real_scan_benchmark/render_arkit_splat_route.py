"""Render a freely moving ape-height camera through a captured Apple room.

This is an appearance/coverage probe. The provisional floor raster supplies a
collision-free camera path; the learned splat supplies RGB. It is not an owner
interaction or a body-qualified navigation result.
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import cv2
import imageio.v2 as imageio
import numpy as np

from prepare_arkit_corpus import ASSETS
from run_mesh_probe import route_pair
from evaluate_trained_arkit_splat import load_renderer


def run(scene_id: str, splat_path: Path, output: Path) -> dict:
    scene = ASSETS / scene_id
    nav = dict(np.load(scene / "nav_candidate.npz"))
    start, end = route_pair(nav)
    floor_z = float(nav["floor_z_m"])
    renderer, splats = load_renderer(splat_path, 480, 360)
    output.mkdir(parents=True, exist_ok=True)
    travel = float(np.linalg.norm(end - start))
    frame_count = max(2, math.ceil(travel / .06) + 1)
    forward_xy = (end - start) / travel
    stats = []
    writer = imageio.get_writer(str(output / "ape_height_route.mp4"), fps=5,
                                codec="libx264", quality=7, macro_block_size=8)
    try:
        for index, fraction in enumerate(np.linspace(0., 1., frame_count)):
            xy = start * (1 - fraction) + end * fraction
            eye = np.r_[xy, floor_z + .42]
            forward = np.r_[forward_xy, 0.]
            renderer.setup_camera(75., eye + forward, eye, np.array([0., 0., 1.]))
            rgb = np.asarray(renderer.render_to_image())[:, :, :3].copy()
            writer.append_data(rgb)
            if index in (0, frame_count // 2, frame_count - 1):
                cv2.imwrite(str(output / f"frame_{index:03d}.png"),
                            cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR))
            stats.append({"frame": index, "travel_m": float(travel * fraction),
                          "eye_world_m": eye.tolist(),
                          "nonblack_fraction": float(np.mean(np.max(rgb, axis=2) > 8))})
    finally:
        writer.close()
    result = {"scene_id": scene_id, "splat_path": str(splat_path),
              "gaussians": int(splats.point.positions.shape[0]),
              "source": "real ARKitScenes RGB photographs, learned Gaussian splat",
              "camera_height_above_provisional_floor_m": .42,
              "route_travel_m": travel, "frames": stats,
              "status": "appearance_coverage_probe_not_owner_or_body_qualification"}
    (output / "trace.json").write_text(json.dumps(result, indent=2) + "\n")
    return {"scene_id": scene_id, "route_travel_m": travel,
            "frame_count": frame_count,
            "minimum_nonblack_fraction": min(x["nonblack_fraction"] for x in stats),
            "video": str(output / "ape_height_route.mp4")}


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--scene", required=True)
    parser.add_argument("--splat", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    print(json.dumps(run(args.scene, args.splat, args.output), indent=2))
