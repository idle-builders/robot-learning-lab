"""Replay exact Habitat-GS route observations as clean RGB and metric depth.

The original videos contain text and a virtual pink navigation target. This
replay removes those overlays, saves the collision-mesh depth and separately
measures its disagreement with the Gaussian-splat appearance depth.
"""

import argparse
import gc
import json
import math
from pathlib import Path

import imageio.v2 as imageio
import numpy as np
from PIL import Image

from run_habitat_gs_probe import make_scene, observe
from screen_habitat_gs import ASSETS


SCENES = (12, 14, 32, 36, 37, 38, 54, 55, 63, 64)


def camera_rotation(heading):
    c, s = math.cos(heading), math.sin(heading)
    # OpenCV right/down/forward columns in the Open3D world frame.
    return [[-c, 0.0, -s], [0.0, -1.0, 0.0], [-s, 0.0, c]]


def observation_pose(trace, index):
    if index == 0:
        start = np.asarray(trace["start_xz_m"], dtype=float)
        goal = np.asarray(trace["target_xz_m"], dtype=float)
        heading = math.atan2(start[0] - goal[0], goal[1] - start[1]) + math.radians(30)
        return start, heading
    preceding = trace["trace"][index - 1]
    return np.asarray(preceding["truth_xz_m"], dtype=float), float(preceding["truth_heading_rad"])


def depth_png(depth, output):
    valid = np.isfinite(depth) & (depth > 0.0) & (depth <= 65.535)
    millimetres = np.where(valid, np.rint(depth * 1000.0), 0)
    Image.fromarray(millimetres.astype(np.uint16)).save(output)


def replay(number):
    split = "train" if number <= 55 else "val"
    name = f"scene{number:02d}"
    folder = ASSETS / split / name / "gs_probe_extended"
    source_trace = folder / "trace.json"
    trace = json.loads(source_trace.read_text())
    level = float(trace["navmesh_floor_y_m"])
    output = folder / "sensor_replay"
    output.mkdir(exist_ok=True)
    renderer, ray_scene, _ = make_scene(number, None, level)
    frames = []
    video = imageio.get_writer(str(output / "clean_first_person.mp4"), fps=5,
                               codec="libx264", quality=7, macro_block_size=8)
    try:
        for index in range(len(trace["trace"])):
            xz, heading = observation_pose(trace, index)
            rgb, mesh_depth, _, _ = observe(renderer, ray_scene, xz, heading, level)
            splat_depth = np.asarray(renderer.render_to_depth_image(z_in_view_space=True))
            rgb_name = f"rgb_{index:04d}.jpg"
            mesh_name = f"depth_{index:04d}.png"
            splat_name = f"splat_depth_{index:04d}.png"
            Image.fromarray(rgb).save(output / rgb_name, quality=93)
            depth_png(mesh_depth, output / mesh_name)
            depth_png(splat_depth, output / splat_name)
            video.append_data(rgb)
            valid = (np.isfinite(mesh_depth) & np.isfinite(splat_depth) &
                     (mesh_depth > 0.15) & (mesh_depth < 6.0) &
                     (splat_depth > 0.15) & (splat_depth < 6.0))
            difference = np.abs(mesh_depth[valid] - splat_depth[valid])
            frames.append({
                "frame": index, "image": rgb_name,
                "depth_mm_png": mesh_name,
                "splat_depth_mm_png": splat_name,
                "position_m": [float(xz[0]), level, float(xz[1])],
                "camera_center_m": [float(xz[0]), level + 0.42, float(xz[1])],
                "camera_to_world_rotation": camera_rotation(heading),
                "heading_rad": heading,
                "mesh_depth_valid_fraction": float(np.isfinite(mesh_depth).mean()),
                "splat_mesh_comparable_fraction": float(valid.mean()),
                "splat_mesh_median_abs_depth_difference_m":
                    float(np.median(difference)) if len(difference) else None,
                "splat_mesh_within_10cm_fraction":
                    float((difference < 0.1).mean()) if len(difference) else None,
            })
    finally:
        video.close()
        del renderer, ray_scene
        gc.collect()
    width, height = 480, 360
    horizontal_fov = 2 * math.degrees(math.atan(
        width / height * math.tan(math.radians(75) / 2)))
    medians = [row["splat_mesh_median_abs_depth_difference_m"]
               for row in frames if row["splat_mesh_median_abs_depth_difference_m"] is not None]
    result = {
        "schema": "ape.habitat-gs-route-sensor-replay.v1",
        "source_trace": str(source_trace),
        "source": "https://huggingface.co/datasets/RukawaY/gs_scenes",
        "scene_id": name, "camera_height_above_nav_m": 0.42,
        "camera_resolution_px": [width, height],
        "camera_hfov_deg": horizontal_fov,
        "camera_vfov_deg": 75.0,
        "depth_encoding": "uint16_millimetres_zero_invalid",
        "rgb_source": "Gaussian splat, clean replay without virtual target or caption",
        "mesh_depth_source": "registered collision mesh, per-pixel raycast",
        "route_splat_mesh_median_depth_difference_m": float(np.median(medians)) if medians else None,
        "frames": frames,
    }
    (output / "sensor_trace.json").write_text(json.dumps(result, indent=2) + "\n")
    print(name, len(frames), "median_splat_mesh_depth_difference_m",
          round(result["route_splat_mesh_median_depth_difference_m"], 3), flush=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--scene-number", type=int)
    args = parser.parse_args()
    for number in (args.scene_number,) if args.scene_number else SCENES:
        replay(number)


if __name__ == "__main__":
    main()
