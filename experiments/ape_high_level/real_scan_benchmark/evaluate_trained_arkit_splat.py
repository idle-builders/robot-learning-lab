"""Qualify one locally trained real-room splat at withheld and upright ape views."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import cv2
import numpy as np
import open3d as o3d

from prepare_arkit_corpus import ASSETS
from render_projective_arkit import frame_geometry
from run_mesh_probe import in_free


def load_renderer(splat_path: Path, width: int, height: int):
    splats = o3d.t.io.read_point_cloud(str(splat_path))
    for key in ("positions", "f_dc", "opacity", "scale", "rot"):
        if key not in splats.point:
            raise ValueError(f"Missing splat attribute {key}")
    logit = splats.point.opacity.numpy()
    splats.point.opacity = o3d.core.Tensor((1 / (1 + np.exp(-np.clip(logit, -30, 30)))).astype(np.float32))
    renderer = o3d.visualization.rendering.OffscreenRenderer(width, height)
    renderer.scene.set_background([0., 0., 0., 1.])
    material = o3d.visualization.rendering.MaterialRecord()
    material.shader = "gaussianSplat"
    material.gaussian_splat_sh_degree = 2
    renderer.scene.add_geometry("room", splats, material)
    return renderer, splats


def render(renderer, intrinsic: np.ndarray, rotation: np.ndarray, translation: np.ndarray) -> np.ndarray:
    width, height, fx, fy, cx, cy = intrinsic
    camera = o3d.camera.PinholeCameraIntrinsic(int(width), int(height), fx, fy, cx, cy)
    extrinsic = np.eye(4)
    extrinsic[:3, :3] = rotation
    extrinsic[:3, 3] = translation
    renderer.setup_camera(camera, extrinsic)
    return np.asarray(renderer.render_to_image())[:, :, :3].copy()


def run(scene_id: str, splat_path: Path, output: Path,
        withheld_indices: list[int]) -> dict:
    scene = ASSETS / scene_id
    photos = scene / "low_height_views"
    frames = json.loads((photos / "manifest.json").read_text())["frames"]
    training = json.loads((scene / "splat_training" / "transforms.json").read_text())["frames"]
    training_centers = np.array([np.asarray(row["transform_matrix"])[:3, 3]
                                 for row in training])
    output.mkdir(parents=True, exist_ok=True)
    renderer, splats = load_renderer(splat_path, 480, 360)
    stats = []
    for index in withheld_indices:
        frame = frames[index]
        intrinsic, R, t = frame_geometry(frame, photos, .75)
        rgb = render(renderer, intrinsic, R, t)
        recorded = cv2.cvtColor(cv2.imread(str(photos / f"{frame['key']}.png")), cv2.COLOR_BGR2RGB)
        recorded = cv2.resize(recorded, (480, 360), interpolation=cv2.INTER_AREA)
        mae = float(np.abs(rgb.astype(np.float32) - recorded.astype(np.float32)).mean())
        rmse = float(np.sqrt(np.mean((rgb.astype(np.float32) - recorded.astype(np.float32)) ** 2)))
        psnr = float(20 * np.log10(255. / max(rmse, 1e-6)))
        cv2.imwrite(str(output / f"withheld_{index:02d}_render.png"), cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR))
        cv2.imwrite(str(output / f"withheld_{index:02d}_recorded.png"), cv2.cvtColor(recorded, cv2.COLOR_RGB2BGR))
        stats.append({"index": index, "height_m": frame["height_above_floor_m"],
                      "nearest_training_camera_m": float(np.linalg.norm(
                          training_centers - np.asarray(frame["center_world_m"]), axis=1).min()),
                      "mae_0_255": mae, "psnr_db": psnr})
    nav = dict(np.load(scene / "nav_candidate.npz"))
    eligible = [f for f in frames if in_free(nav, np.asarray(f["center_world_m"][:2]))]
    if not eligible:
        raise ValueError("No low-camera source position inside provisional free raster")
    xy = np.median(np.array([f["center_world_m"][:2] for f in eligible]), axis=0)
    eye = np.r_[xy, float(nav["floor_z_m"]) + .42]
    upright = []
    intrinsic, _, _ = frame_geometry(frames[0], photos, .75)
    for degrees in (0, 90, 180, 270):
        theta = math.radians(degrees)
        forward = np.array([math.sin(theta), math.cos(theta), 0.])
        right = np.array([math.cos(theta), -math.sin(theta), 0.])
        R = np.stack((right, np.array([0., 0., -1.]), forward))
        rgb = render(renderer, intrinsic, R, -R @ eye)
        cv2.imwrite(str(output / f"upright_{degrees:03d}.png"), cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR))
        upright.append({"heading_deg": degrees,
                        "nonblack_fraction": float(np.mean(np.max(rgb, axis=2) > 8))})
    result = {"scene_id": scene_id, "splat_path": str(splat_path),
              "gaussians": int(splats.point.positions.shape[0]),
              "withheld_views": stats, "upright_eye_world_m": eye.tolist(),
              "upright_views": upright,
              "status": "reconstruction_visual_screen_not_owner_or_body_qualification"}
    (output / "metrics.json").write_text(json.dumps(result, indent=2) + "\n")
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--scene", required=True)
    parser.add_argument("--splat", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--withheld-indices", type=int, nargs="+", default=[4, 18, 36])
    args = parser.parse_args()
    print(json.dumps(run(args.scene, args.splat, args.output, args.withheld_indices), indent=2))
