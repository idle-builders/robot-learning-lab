"""Evaluate a cheap photo-projection renderer against withheld ape-height RGB.

This is deliberately a baseline, not a Gaussian reconstruction. Only source
photographs and Apple's captured mesh enter rendering; the withheld image is
used for scoring after the novel view exists.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import cv2
import numpy as np
import open3d as o3d
from scipy.spatial.transform import Rotation

from prepare_arkit_corpus import ASSETS


def frame_geometry(frame: dict, directory: Path, scale: float) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    intrinsics_path = Path(frame.get("intrinsics_path", directory / f"{frame['key']}.pincam"))
    w, h, fx, fy, cx, cy = np.loadtxt(intrinsics_path)
    intrinsic = np.array([w, h, fx, fy, cx, cy], float)
    intrinsic *= scale
    rotation = Rotation.from_rotvec(frame["world_to_camera_rotvec"]).as_matrix()
    translation = np.asarray(frame["world_to_camera_t_m"], float)
    return intrinsic, rotation, translation


def sampled_vga_sources(scene_dir: Path) -> list[dict]:
    record = json.loads((scene_dir / "scene.json").read_text())
    trajectory = np.loadtxt(scene_dir / "camera.traj")
    out = []
    for frame in record["selected_frames"]:
        folder = scene_dir / "frames" / f"{frame['index']:02d}"
        if not (folder / "rgb_vga.png").exists():
            continue
        pose = trajectory[np.argmin(np.abs(trajectory[:, 0] - frame["timestamp_s"]))]
        rotation = Rotation.from_rotvec(pose[1:4]).as_matrix()
        center = -rotation.T @ pose[4:7]
        out.append({"key": f"sampled_{frame['index']:02d}",
                    "image_time_s": float(frame["timestamp_s"]),
                    "center_world_m": center.tolist(),
                    "world_to_camera_rotvec": pose[1:4].tolist(),
                    "world_to_camera_t_m": pose[4:7].tolist(),
                    "image_path": str(folder / "rgb_vga.png"),
                    "intrinsics_path": str(folder / "intrinsics_vga.pincam")})
    return out


def raycast_depth(scene, intrinsic, rotation, translation) -> tuple[np.ndarray, np.ndarray]:
    width, height, fx, fy, cx, cy = intrinsic
    u, v = np.meshgrid(np.arange(int(width)), np.arange(int(height)))
    cam_dir = np.stack(((u - cx) / fx, (v - cy) / fy, np.ones_like(u)), -1)
    world_dir = cam_dir @ rotation
    eye = -rotation.T @ translation
    rays = np.concatenate((np.broadcast_to(eye, world_dir.shape), world_dir), -1).astype(np.float32)
    depth = scene.cast_rays(o3d.core.Tensor(rays))["t_hit"].numpy()
    point = eye + world_dir * depth[..., None]
    return depth, point


def evaluate(scene_id: str, withheld_index: int, output: Path, max_source: int,
             min_source_displacement_m: float = .15,
             upright_eye_heading: tuple[np.ndarray, float] | None = None,
             include_sampled_vga: bool = False) -> dict:
    directory = ASSETS / scene_id / "low_height_views"
    manifest = json.loads((directory / "manifest.json").read_text())
    frames = manifest["frames"]
    held = frames[withheld_index].copy()
    if upright_eye_heading is not None:
        upright_eye, heading = upright_eye_heading
        forward = np.array([np.sin(heading), np.cos(heading), 0.])
        right = np.array([np.cos(heading), -np.sin(heading), 0.])
        rotation = np.stack((right, np.array([0., 0., -1.]), forward))
        held["world_to_camera_rotvec"] = Rotation.from_matrix(rotation).as_rotvec().tolist()
        held["world_to_camera_t_m"] = (-rotation @ upright_eye).tolist()
        held["center_world_m"] = upright_eye.tolist()
    eye = np.asarray(held["center_world_m"])
    candidates = frames + (sampled_vga_sources(ASSETS / scene_id) if include_sampled_vga else [])
    source = [f for f in candidates
              if (upright_eye_heading is not None or abs(f["image_time_s"] - held["image_time_s"]) > .5)
              and np.linalg.norm(np.asarray(f["center_world_m"]) - eye) >= min_source_displacement_m]
    source.sort(key=lambda f: np.linalg.norm(np.asarray(f["center_world_m"]) - eye))
    source = source[:max_source]
    mesh = o3d.io.read_triangle_mesh(str(ASSETS / scene_id / "mesh.ply"), enable_post_processing=False)
    rays = o3d.t.geometry.RaycastingScene()
    rays.add_triangles(o3d.t.geometry.TriangleMesh.from_legacy(mesh))
    target_intr, target_rot, target_t = frame_geometry(held, directory, .75)
    target_depth, point_world = raycast_depth(rays, target_intr, target_rot, target_t)
    height, width = target_depth.shape
    output_rgb = np.zeros((height, width, 3), np.uint8)
    best_score = np.full((height, width), np.inf, np.float32)
    valid_target = np.isfinite(target_depth) & (target_depth > .1) & (target_depth < 8.)
    # Raycast at quarter-size source resolution to reject surfaces hidden from
    # each source photograph. This avoids copying a wall image onto a chair.
    for frame in source:
        intrinsic, rotation, translation = frame_geometry(frame, directory, 1.)
        source_intr = intrinsic * .25
        source_depth, _ = raycast_depth(rays, source_intr, rotation, translation)
        camera = np.where(valid_target[..., None], point_world, 0.) @ rotation.T + translation
        z = camera[..., 2]
        projected_u = camera[..., 0] * intrinsic[2] / np.maximum(z, 1e-6) + intrinsic[4]
        projected_v = camera[..., 1] * intrinsic[3] / np.maximum(z, 1e-6) + intrinsic[5]
        depth_at_source = cv2.remap(source_depth, (projected_u * .25).astype(np.float32),
                                    (projected_v * .25).astype(np.float32),
                                    cv2.INTER_NEAREST, borderMode=cv2.BORDER_CONSTANT,
                                    borderValue=0)
        image_path = Path(frame.get("image_path", directory / f"{frame['key']}.png"))
        image = cv2.cvtColor(cv2.imread(str(image_path)), cv2.COLOR_BGR2RGB)
        color = cv2.remap(image, projected_u.astype(np.float32), projected_v.astype(np.float32),
                          cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT, borderValue=0)
        seen = (valid_target & (z > .1) & (projected_u >= 2) &
                (projected_u < intrinsic[0] - 2) & (projected_v >= 2) &
                (projected_v < intrinsic[1] - 2) & np.isfinite(depth_at_source) &
                (np.abs(z - depth_at_source) < .15))
        source_eye = np.asarray(frame["center_world_m"])
        displacement = float(np.linalg.norm(source_eye - eye))
        radial = ((projected_u - intrinsic[4]) / intrinsic[0]) ** 2 + \
                 ((projected_v - intrinsic[5]) / intrinsic[1]) ** 2
        score = displacement + .2 * radial
        replace = seen & (score < best_score)
        output_rgb[replace] = color[replace]
        best_score[replace] = score[replace]
    reference = None
    if upright_eye_heading is None:
        reference = cv2.cvtColor(cv2.imread(str(directory / f"{held['key']}.png")), cv2.COLOR_BGR2RGB)
        reference = cv2.resize(reference, (width, height), interpolation=cv2.INTER_AREA)
    filled = np.isfinite(best_score)
    # Score only pixels backed by the captured mesh and a visible source photo.
    mae = (float(np.abs(output_rgb[filled].astype(np.float32) - reference[filled]).mean())
           if reference is not None and filled.any() else None)
    output.mkdir(parents=True, exist_ok=True)
    if reference is not None:
        cv2.imwrite(str(output / "recorded_reference.png"), cv2.cvtColor(reference, cv2.COLOR_RGB2BGR))
    cv2.imwrite(str(output / "projected_render.png"), cv2.cvtColor(output_rgb, cv2.COLOR_RGB2BGR))
    cv2.imwrite(str(output / "coverage.png"), (filled.astype(np.uint8) * 255))
    result = {"scene_id": scene_id, "withheld_index": withheld_index,
              "withheld_key": held["key"], "height_m": held["height_above_floor_m"],
              "source_count": len(source), "source_exclusion_s": .5,
              "include_sampled_high_vga": include_sampled_vga,
              "minimum_source_camera_displacement_m": min_source_displacement_m,
              "photo_visible_pixel_fraction": float(filled.mean()),
              "recorded_vs_projected_mae_0_255_on_filled": mae,
              "status": ("heldout_photo_projection_screen_not_photoreal_qualification"
                         if reference is not None else "upright_free_view_visual_screen_not_qualified"),
              "limitations": ("coarse mesh, nearest-source seams, no physical body; "
                              "single off-trajectory view, no route qualification" if upright_eye_heading is not None
                              else "coarse mesh, nearest-source seams, no off-trajectory or body qualification")}
    if upright_eye_heading is not None:
        result["upright_eye_world_m"] = eye.tolist()
        result["heading_rad"] = float(heading)
    (output / "metrics.json").write_text(json.dumps(result, indent=2) + "\n")
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--scene", required=True)
    parser.add_argument("--withheld-index", type=int, default=24)
    parser.add_argument("--max-source", type=int, default=12)
    parser.add_argument("--min-source-displacement-m", type=float, default=.15)
    parser.add_argument("--upright-xy", type=float, nargs=2)
    parser.add_argument("--heading-deg", type=float, default=0.)
    parser.add_argument("--include-sampled-vga", action="store_true")
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    upright = None
    if args.upright_xy is not None:
        floor_z = float(np.load(ASSETS / args.scene / "nav_candidate.npz")["floor_z_m"])
        upright = (np.array([args.upright_xy[0], args.upright_xy[1], floor_z + .42]),
                   np.deg2rad(args.heading_deg))
    print(json.dumps(evaluate(args.scene, args.withheld_index, args.output, args.max_source,
                              args.min_source_displacement_m, upright, args.include_sampled_vga), indent=2))
