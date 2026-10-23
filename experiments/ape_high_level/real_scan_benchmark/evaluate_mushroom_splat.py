"""Screen one MuSHRoom splat against held-out photos and ape-height views.

This is an appearance gate. A good image alone does not establish navigation,
collision safety, object finding, or selected-body qualification.
"""

from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path

import numpy as np
import open3d as o3d
from PIL import Image


ROOT = Path(__file__).resolve().parents[3]
ASSETS = ROOT / "assets/real_indoor_scenes/mushroom"
DEFAULTS = {
    "coffee_room": (("frame_00041", "frame_00166"), ("frame_00041", "frame_00167")),
    "vr_room": (("frame_00045", "frame_00076"), ("frame_00048", "frame_00079")),
}
CV_FROM_GL = np.diag([1.0, -1.0, -1.0, 1.0])


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--room", choices=DEFAULTS, default="coffee_room")
    parser.add_argument("--splat", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--test-list", type=Path)
    parser.add_argument("--max-scale-m", type=float)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    data = ASSETS / args.room / "extracted" / "room_datasets" / args.room / "iphone/long_capture"
    floor_z = json.loads((ASSETS / args.room / "metric_low_view_screen.json").read_text())["floor_from_captured_depth_mode_z_m"]
    heldout, anchors = DEFAULTS[args.room]
    test_list = args.test_list or data / "test.txt"
    test_names = set(test_list.read_text().splitlines())
    if not set(heldout).issubset(test_names):
        raise ValueError(f"Selected held-out frames missing from {test_list}")

    scene = json.loads((data / "transformations_colmap.json").read_text())
    frames = {Path(f["file_path"]).stem: f for f in scene["frames"]}
    first = frames[heldout[0]]
    width = int(first.get("w", scene.get("w")))
    height = int(first.get("h", scene.get("h")))
    splats = o3d.t.io.read_point_cloud(str(args.splat))
    required = ("positions", "f_dc", "opacity", "scale", "rot")
    for name in required:
        if name not in splats.point:
            raise ValueError(f"Gaussian PLY missing {name}")
    if args.max_scale_m is not None:
        keep = np.max(splats.point.scale.numpy(), axis=1) <= args.max_scale_m
        splats = splats.select_by_mask(o3d.core.Tensor(keep))
    logits = splats.point.opacity.numpy()
    splats.point.opacity = o3d.core.Tensor(
        (1.0 / (1.0 + np.exp(-np.clip(logits, -30, 30)))).astype(np.float32)
    )
    renderer = o3d.visualization.rendering.OffscreenRenderer(width, height)
    renderer.scene.set_background([0, 0, 0, 1])
    material = o3d.visualization.rendering.MaterialRecord()
    material.shader = "gaussianSplat"
    material.gaussian_splat_sh_degree = 3
    material.gaussian_splat_max_tiles_per_splat = 4096
    renderer.scene.add_geometry("mushroom", splats, material)
    results = {"schema": "ape.mushroom-splat-appearance-screen.v1", "room": args.room,
               "test_list": str(test_list), "gaussians_rendered": int(splats.point.positions.shape[0]),
               "max_scale_m": args.max_scale_m, "heldout": [], "upright": []}

    for name in heldout:
        frame = frames[name]
        intrinsic = o3d.camera.PinholeCameraIntrinsic(
            width, height, frame.get("fl_x", scene.get("fl_x")), frame.get("fl_y", scene.get("fl_y")),
            frame.get("cx", scene.get("cx")), frame.get("cy", scene.get("cy"))
        )
        pose_gl = np.asarray(frame["transform_matrix"], dtype=float)
        camera_to_world_cv = pose_gl @ CV_FROM_GL
        renderer.setup_camera(intrinsic, np.linalg.inv(camera_to_world_cv))
        image = np.asarray(renderer.render_to_image())[:, :, :3]
        target = args.output / f"{name}_splat.png"
        Image.fromarray(image).save(target)
        depth = np.asarray(renderer.render_to_depth_image(z_in_view_space=True))
        np.save(args.output / f"{name}_splat_depth.npy", depth)
        photo_path = data / frame["file_path"]
        shutil.copy2(photo_path, args.output / f"{name}_real.jpg")
        real = np.asarray(Image.open(photo_path).convert("RGB"), dtype=np.float32)
        diff = real - image.astype(np.float32)
        mse = float(np.mean(diff * diff))
        results["heldout"].append({
            "frame": name,
            "is_dataset_test_frame": True,
            "camera_height_above_floor_m": float(pose_gl[2, 3] - floor_z),
            "psnr_db": float(10 * np.log10(255**2 / max(mse, 1e-12))),
            "mean_abs_rgb_255": float(np.mean(np.abs(diff))),
            "nonblack_fraction": float(np.mean(np.any(image > 4, axis=2))),
            "render": str(target),
            "real": str(args.output / f"{name}_real.jpg"),
        })

    for name in anchors:
        pose = np.asarray(frames[name]["transform_matrix"], dtype=float)
        eye = pose[:3, 3].copy()
        eye[2] = floor_z + 0.42
        forward = -pose[:3, 2].copy()
        forward[2] = 0
        forward /= np.linalg.norm(forward)
        for degrees in (0, 90, 180, 270):
            rad = np.deg2rad(degrees)
            fwd = np.array([
                np.cos(rad) * forward[0] - np.sin(rad) * forward[1],
                np.sin(rad) * forward[0] + np.cos(rad) * forward[1],
                0.0,
            ])
            renderer.setup_camera(63.0, eye + fwd, eye, [0, 0, 1])
            image = np.asarray(renderer.render_to_image())[:, :, :3]
            target = args.output / f"{name}_ape_042m_yaw{degrees:03d}.png"
            Image.fromarray(image).save(target)
            results["upright"].append({
                "anchor": name,
                "eye_world_m": eye.tolist(),
                "yaw_offset_degrees": degrees,
                "nonblack_fraction": float(np.mean(np.any(image > 4, axis=2))),
                "render": str(target),
            })
    (args.output / "appearance_screen.json").write_text(json.dumps(results, indent=2) + "\n")
    print(args.output / "appearance_screen.json")


if __name__ == "__main__":
    main()
