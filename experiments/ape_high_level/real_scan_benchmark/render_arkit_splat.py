"""Inspect a gated SceneSplat ARKit room from an Apple-posed camera sample.

Splat RGB/depth is an appearance observation; collision and range truth must be
derived independently from the measured Apple mesh. A successful render does
not qualify ape-height novel views or closed navigation.
"""

import argparse
import json
from pathlib import Path

import numpy as np
import open3d as o3d
from scipy.spatial.transform import Rotation

from prepare_arkit_corpus import ASSETS


def render(scene_id: str, splat_path: Path, frame_index: int, output: Path) -> None:
    scene_dir = ASSETS / scene_id
    record = json.loads((scene_dir / "scene.json").read_text())
    frame = record["selected_frames"][frame_index]
    frame_dir = scene_dir / "frames" / f"{frame_index:02d}"
    width, height, fx, fy, cx, cy = np.loadtxt(frame_dir / "intrinsics_vga.pincam")
    if (width, height) != (640, 480):
        raise ValueError("Expected calibrated 640x480 VGA intrinsics")
    poses = np.loadtxt(scene_dir / "camera.traj")
    pose = poses[np.argmin(np.abs(poses[:, 0] - frame["timestamp_s"]))]
    extrinsic = np.eye(4)
    extrinsic[:3, :3] = Rotation.from_rotvec(pose[1:4]).as_matrix()
    extrinsic[:3, 3] = pose[4:7]

    splats = o3d.t.io.read_point_cloud(str(splat_path))
    for attribute in ("positions", "f_dc", "opacity", "scale", "rot"):
        if attribute not in splats.point:
            raise ValueError(f"Missing Gaussian-splat attribute: {attribute}")
    # The common 3DGS PLY stores opacity logits. Open3D's reader linearizes
    # log-scales, but leaves opacity as stored; the renderer expects [0, 1].
    logits = splats.point.opacity.numpy()
    alpha = 1 / (1 + np.exp(-np.clip(logits, -30, 30)))
    splats.point.opacity = o3d.core.Tensor(alpha.astype(np.float32))

    renderer = o3d.visualization.rendering.OffscreenRenderer(640, 480)
    renderer.scene.set_background([0, 0, 0, 1])
    material = o3d.visualization.rendering.MaterialRecord()
    material.shader = "gaussianSplat"
    material.gaussian_splat_sh_degree = 2
    renderer.scene.add_geometry("real_capture_splat", splats, material)
    intrinsics = o3d.camera.PinholeCameraIntrinsic(640, 480, fx, fy, cx, cy)
    renderer.setup_camera(intrinsics, extrinsic)
    output.parent.mkdir(parents=True, exist_ok=True)
    o3d.io.write_image(str(output), renderer.render_to_image())
    depth = np.asarray(renderer.render_to_depth_image(z_in_view_space=True))
    np.save(output.with_suffix(".depth.npy"), depth)
    output.with_suffix(".json").write_text(json.dumps({
        "scene_id": scene_id, "visit_id": record["visit_id"],
        "splat_path": str(splat_path), "frame_index": frame_index,
        "pose_timestamp_s": float(pose[0]),
        "rgb_pose_timestamp_s": frame["timestamp_s"],
        "camera_height_world_z_m": float((-extrinsic[:3, :3].T @ extrinsic[:3, 3])[2]),
        "render_status": "first_view_alignment_and_ape_height_unverified",
        "depth_status": "splat_render_depth_not_collision_truth",
    }, indent=2) + "\n")
    print(output)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--scene", required=True)
    parser.add_argument("--splat", required=True, type=Path)
    parser.add_argument("--frame", type=int, default=0)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    render(args.scene, args.splat, args.frame, args.output)


if __name__ == "__main__":
    main()
