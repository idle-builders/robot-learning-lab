"""Fetch a spatially spread set of real low-height ARKitScenes photographs.

This is input for appearance reconstruction, not simulated camera evidence.
The poses are nearest published trajectory samples; RGB time offsets remain
explicit in the manifest. Keep source data under ignored local assets.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from remotezip import RemoteZip
from scipy.spatial.transform import Rotation

from prepare_arkit_corpus import ASSETS, SOURCE, timestamp


def select_pose_rows(scene: Path, count: int) -> tuple[np.ndarray, float]:
    trajectory = np.loadtxt(scene / "camera.traj")
    rotation = Rotation.from_rotvec(trajectory[:, 1:4]).as_matrix()
    center = -np.einsum("nji,nj->ni", rotation, trajectory[:, 4:7])
    floor_z = float(np.load(scene / "nav_candidate.npz")["floor_z_m"])
    heights = center[:, 2] - floor_z
    candidates = np.flatnonzero((heights >= .35) & (heights <= .65))
    if len(candidates) == 0:
        raise ValueError("No recorded 35–65 cm camera pose")
    # Maximin spatial sampling avoids selecting a hundred stationary start frames.
    chosen = [int(candidates[np.argmin(heights[candidates])])]
    while len(chosen) < min(count, len(candidates)):
        xy = center[candidates, :2]
        distances = np.linalg.norm(xy[:, None, :] - center[chosen, :2][None, :, :], axis=2)
        nearest_m = distances.min(axis=1)
        # Reward a different view direction at almost the same position.
        look = rotation[candidates, 2, :2]
        selected_look = rotation[chosen, 2, :2]
        direction_difference = (1 - np.clip(look @ selected_look.T, -1, 1)).min(axis=1)
        score = nearest_m + .1 * direction_difference
        score[np.isin(candidates, chosen)] = -1
        next_index = int(candidates[np.argmax(score)])
        if score[np.argmax(score)] < .035:
            break
        chosen.append(next_index)
    return trajectory[np.sort(chosen)], floor_z


def fetch(scene_id: str, count: int) -> dict:
    scene = ASSETS / scene_id
    record = json.loads((scene / "scene.json").read_text())
    poses, floor_z = select_pose_rows(scene, count)
    base = f"{SOURCE}/raw/{record['fold']}/{scene_id}"
    output = scene / "low_height_views"
    output.mkdir(exist_ok=True)
    rows = []
    with RemoteZip(base + "/vga_wide.zip") as image_zip, RemoteZip(base + "/vga_wide_intrinsics.zip") as intr_zip:
        images = [x for x in image_zip.namelist() if x.endswith(".png")]
        intrinsics = [x for x in intr_zip.namelist() if x.endswith(".pincam")]
        for pose in poses:
            rgb_name = min(images, key=lambda x: abs(timestamp(x) - pose[0]))
            image_time = timestamp(rgb_name)
            intr_name = min(intrinsics, key=lambda x: abs(timestamp(x) - image_time))
            offset = image_time - float(pose[0])
            if abs(offset) > .05 or abs(timestamp(intr_name) - image_time) > .005:
                raise ValueError(f"{scene_id}: RGB/pose/intrinsics mismatch {offset:.3f}s")
            key = Path(rgb_name).stem
            (output / f"{key}.png").write_bytes(image_zip.read(rgb_name))
            (output / f"{key}.pincam").write_bytes(intr_zip.read(intr_name))
            camera_center = -Rotation.from_rotvec(pose[1:4]).as_matrix().T @ pose[4:7]
            rows.append({"key": key, "image_time_s": image_time,
                         "pose_time_s": float(pose[0]), "offset_s": offset,
                         "world_to_camera_rotvec": pose[1:4].tolist(),
                         "world_to_camera_t_m": pose[4:7].tolist(),
                         "center_world_m": camera_center.tolist(),
                         "height_above_floor_m": float(camera_center[2] - floor_z),
                         "image_source": base + "/vga_wide.zip#" + rgb_name})
    manifest = {"schema": "ape.arkit-low-camera-photographs.v1",
                "scene_id": scene_id, "visit_id": record["visit_id"],
                "floor_z_m": floor_z, "frame_count": len(rows),
                "selection": "maximin XY with modest look-direction preference within 0.35–0.65 m",
                "status": "recorded_photographs_not_simulated_views", "frames": rows}
    (output / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    return manifest


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--scene", required=True)
    parser.add_argument("--count", type=int, default=48)
    args = parser.parse_args()
    result = fetch(args.scene, args.count)
    print(f"{args.scene}: {result['frame_count']} captured low-height photographs")
