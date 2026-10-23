"""Build a posed real-photo training bundle for one Apple room reconstruction.

The bundle keeps Apple's metre-scale world frame. It never uses held-out low
camera images as training inputs. The actual splat fit runs on Runpod; no model
or simulated view is claimed by preparing this dataset alone.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from remotezip import RemoteZip
from scipy.spatial.transform import Rotation

from prepare_arkit_corpus import ASSETS, SOURCE, timestamp


def prepare(scene_id: str, count: int, heldout_indices: list[int]) -> dict:
    scene = ASSETS / scene_id
    record = json.loads((scene / "scene.json").read_text())
    trajectory = np.loadtxt(scene / "camera.traj")
    low_frames = json.loads((scene / "low_height_views" / "manifest.json").read_text())["frames"]
    heldout = [low_frames[i] for i in heldout_indices]
    bundle = scene / "splat_training"
    image_dir = bundle / "images"
    image_dir.mkdir(parents=True, exist_ok=True)
    base = f"{SOURCE}/raw/{record['fold']}/{scene_id}"
    source_rows = trajectory[np.unique(np.rint(np.linspace(0, len(trajectory) - 1, count)).astype(int))]
    frames = []
    with RemoteZip(base + "/vga_wide.zip") as image_zip, RemoteZip(base + "/vga_wide_intrinsics.zip") as intr_zip:
        images = [x for x in image_zip.namelist() if x.endswith(".png")]
        intrinsics = [x for x in intr_zip.namelist() if x.endswith(".pincam")]
        used_image_names: set[str] = set()
        for pose in source_rows:
            if any(abs(float(pose[0]) - h["pose_time_s"]) < .5 for h in heldout):
                continue
            image_name = min(images, key=lambda x: abs(timestamp(x) - pose[0]))
            image_time = timestamp(image_name)
            if image_name in used_image_names:
                continue
            intr_name = min(intrinsics, key=lambda x: abs(timestamp(x) - image_time))
            if abs(image_time - pose[0]) > .05 or abs(timestamp(intr_name) - image_time) > .005:
                raise ValueError(f"{scene_id}: image/pose/intrinsics mismatch")
            used_image_names.add(image_name)
            path = image_dir / Path(image_name).name
            if not path.exists():
                path.write_bytes(image_zip.read(image_name))
            w, h, fx, fy, cx, cy = np.fromstring(intr_zip.read(intr_name), sep=" ")
            R = Rotation.from_rotvec(pose[1:4]).as_matrix()
            c2w = np.eye(4)
            c2w[:3, :3] = R.T @ np.diag([1., -1., -1.])
            c2w[:3, 3] = -R.T @ pose[4:7]
            frames.append({"file_path": "images/" + path.name,
                           "transform_matrix": c2w.tolist(),
                           "fl_x": float(fx), "fl_y": float(fy),
                           "cx": float(cx), "cy": float(cy),
                           "w": int(w), "h": int(h),
                           "source_rgb_time_s": image_time,
                           "source_pose_time_s": float(pose[0])})
            if len(frames) % 25 == 0:
                print(f"{scene_id}: {len(frames)} training photos", flush=True)
    transforms = {"camera_model": "OPENCV", "frames": frames,
                  "orientation_override": "none"}
    (bundle / "transforms.json").write_text(json.dumps(transforms, indent=2) + "\n")
    manifest = {"scene_id": scene_id, "visit_id": record["visit_id"],
                "source": base + "/vga_wide.zip", "frame_count": len(frames),
                "heldout_low_height_indices": heldout_indices,
                "heldout_low_height_keys": [x["key"] for x in heldout],
                "heldout_exclusion_s": .5,
                "world_frame": "Apple ARKitScenes captured mesh frame, metres, +Z up",
                "camera_convention": "OpenCV source pose converted to nerfstudio OpenGL c2w",
                "status": "training_bundle_only_no_model_or_ape_view"}
    (bundle / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    return manifest


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--scene", required=True)
    parser.add_argument("--count", type=int, default=220)
    parser.add_argument("--heldout-indices", type=int, nargs="+", default=[4, 18, 36])
    args = parser.parse_args()
    print(json.dumps(prepare(args.scene, args.count, args.heldout_indices), indent=2))
