"""Estimate floor and true ape-height camera coverage from captured RGB-D."""

from __future__ import annotations

import argparse
import io
import json
import tarfile
from pathlib import Path

import numpy as np
from PIL import Image

from screen_mushroom_camera_heights import POSES, camera_centres


ROOT = Path(__file__).resolve().parents[3] / "assets/real_indoor_scenes/mushroom"


def max_separation(xy: np.ndarray) -> float:
    if len(xy) < 2:
        return 0.0
    sample = xy[::max(1, len(xy)//200)]
    return float(np.max(np.linalg.norm(sample[:, None] - sample[None, :], axis=2)))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--room", required=True)
    args = parser.parse_args()
    room = args.room
    archive_path = ROOT / room / f"{room}_iphone.tar.gz"
    pose_path = POSES / f"{room}_images.txt"
    names = [tokens[-1] for line in pose_path.read_text().splitlines()
             if len(tokens := line.replace(",", "").split()) == 10 and tokens[0].isdigit()
             and tokens[-1].endswith(".jpg")]
    xyz = camera_centres(pose_path)
    if len(names) != len(xyz):
        raise ValueError("Pose list and image list have different lengths")
    candidate = np.flatnonzero(xyz[:, 2] <= xyz[:, 2].min() + 0.35)
    chosen = [int(candidate[np.argmin(xyz[candidate, 2])])]
    while len(chosen) < min(12, len(candidate)):
        xy = xyz[candidate, :2]
        separation = np.min(np.linalg.norm(xy[:, None] - xyz[np.asarray(chosen), :2][None, :], axis=2), axis=1)
        separation[np.isin(candidate, chosen)] = -1
        chosen.append(int(candidate[np.argmax(separation)]))

    base = f"room_datasets/{room}/iphone/long_capture/"
    z_samples = []
    with tarfile.open(archive_path) as archive:
        transformations = json.load(archive.extractfile(archive.getmember(base + "transformations_colmap.json")))
        frames = {Path(f["file_path"]).name: f for f in transformations["frames"]}
        for index in chosen:
            name = names[index]
            frame = frames[name]
            depth_name = Path(name).with_suffix(".png").name
            data = archive.extractfile(archive.getmember(base + "depth/" + depth_name))
            if data is None:
                raise ValueError(depth_name)
            depth = np.asarray(Image.open(io.BytesIO(data.read())), dtype=np.float32) / 1000.0
            h, w = depth.shape
            v, u = np.mgrid[0:h:8, 0:w:8]
            depth = depth[::8, ::8]
            cx = frame.get("cx", transformations.get("cx"))
            cy = frame.get("cy", transformations.get("cy"))
            fx = frame.get("fl_x", transformations.get("fl_x"))
            fy = frame.get("fl_y", transformations.get("fl_y"))
            points = np.stack(((u - cx) * depth / fx,
                               -(v - cy) * depth / fy, -depth), axis=-1).reshape(-1, 3)
            pose = np.asarray(frame["transform_matrix"])
            z = (points @ pose[:3, :3].T + pose[:3, 3])[:, 2]
            valid = (depth.ravel() > 0.2) & (depth.ravel() < 8.0)
            z_samples.append(z[valid])
    heights = np.concatenate(z_samples)
    edges = np.arange(float(xyz[:, 2].min() - 2.0), float(xyz[:, 2].min() - 0.15), 0.02)
    count, bins = np.histogram(heights, bins=edges)
    peak = int(np.argmax(count))
    floor = float((bins[peak] + bins[peak + 1]) / 2)
    ape_band = xyz[(xyz[:, 2] >= floor + 0.35) & (xyz[:, 2] <= floor + 0.65)]
    result = {
        "schema": "ape.mushroom-metric-low-view-screen.v1", "room": room,
        "floor_from_captured_depth_mode_z_m": floor,
        "floor_mode_2cm_sample_count": int(count[peak]),
        "floor_mode_fraction_of_sampled_depth": float(count[peak] / len(heights)),
        "camera_frames": len(xyz),
        "ape_height_camera_frames_035_to_065_m": len(ape_band),
        "ape_height_max_xy_separation_m": max_separation(ape_band[:, :2]),
        "lowest_camera_height_above_depth_floor_m": float(xyz[:, 2].min() - floor),
        "floor_estimation_frames": [names[i] for i in chosen],
        "caution": "Floor estimated from twelve registered iPhone depth frames, not yet compared to Faro reference for this room.",
    }
    output = ROOT / room / "metric_low_view_screen.json"
    output.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
