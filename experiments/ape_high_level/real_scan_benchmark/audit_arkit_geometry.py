"""Check measured RGB-D/pose alignment against each captured reconstruction.

This is an asset-quality screen, not a navigation or robot collision certificate.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import trimesh
from PIL import Image
from scipy.spatial import cKDTree
from scipy.spatial.transform import Rotation


ROOT = Path(__file__).resolve().parents[3]
ASSETS = ROOT / "assets/real_indoor_scenes/arkit_v1"


def audit(scene_dir: Path) -> dict:
    record = json.loads((scene_dir / "scene.json").read_text())
    mesh = trimesh.load(scene_dir / "mesh.ply", force="mesh", process=False)
    vertices = np.asarray(mesh.vertices)
    nearest_vertex = cKDTree(vertices)
    poses = np.loadtxt(scene_dir / "camera.traj")
    errors = []
    valid_depth_fraction = []
    camera_heights = []
    for frame in record["selected_frames"]:
        folder = scene_dir / "frames" / f"{frame['index']:02d}"
        depth = np.asarray(Image.open(folder / "depth.png"), dtype=np.float32) / 1000.0
        h, w = depth.shape
        width, height, fx, fy, cx, cy = np.loadtxt(folder / "intrinsics.pincam")
        if (width, height) != (w, h):
            raise ValueError(f"{scene_dir.name}: depth/intrinsics dimensions differ")
        valid_depth_fraction.append(float(np.mean(depth > 0)))
        y, x = np.mgrid[0:h:12, 0:w:12]
        z = depth[y, x]
        keep = z > 0
        point_cam = np.column_stack(((x[keep] - cx) * z[keep] / fx,
                                     (y[keep] - cy) * z[keep] / fy, z[keep]))
        pose = poses[np.argmin(np.abs(poses[:, 0] - frame["timestamp_s"]))]
        rotation_world_to_phone = Rotation.from_rotvec(pose[1:4]).as_matrix()
        point_world = (point_cam - pose[4:7]) @ rotation_world_to_phone
        errors.extend(nearest_vertex.query(point_world)[0])
        camera_heights.append(float((-rotation_world_to_phone.T @ pose[4:7])[2]))
    errors = np.asarray(errors)
    # Estimate horizontal support from a uniform face sample. It is deliberately
    # only a screening signal: holes and 50 cm body clearance need navmesh checks.
    faces = np.asarray(mesh.faces)
    ids = np.linspace(0, len(faces) - 1, min(100_000, len(faces)), dtype=np.int64)
    triangles = vertices[faces[ids]]
    normals_twice_area = np.cross(triangles[:, 1] - triangles[:, 0],
                                  triangles[:, 2] - triangles[:, 0])
    areas = np.linalg.norm(normals_twice_area, axis=1) / 2
    horizontal = (np.abs(normals_twice_area[:, 2]) >=
                  0.9 * np.maximum(np.linalg.norm(normals_twice_area, axis=1), 1e-9))
    below_camera = triangles[:, :, 2].mean(axis=1) < np.median(camera_heights)
    approx_horizontal_area_m2 = float(np.sum(areas[horizontal & below_camera]) * len(faces) / len(ids))
    return {
        "video_id": scene_dir.name,
        "visit_id": record["visit_id"], "fold": record["fold"],
        "mesh_vertices": len(vertices), "mesh_faces": len(faces),
        "mesh_bounds_m": np.asarray(mesh.bounds).round(3).tolist(),
        "camera_height_z_m": [round(float(min(camera_heights)), 3),
                              round(float(max(camera_heights)), 3)],
        "mean_valid_recorded_depth_fraction": round(float(np.mean(valid_depth_fraction)), 4),
        "depth_to_mesh_nearest_vertex_median_m": round(float(np.median(errors)), 4),
        "depth_to_mesh_nearest_vertex_p90_m": round(float(np.percentile(errors, 90)), 4),
        "depth_to_mesh_within_10cm_fraction": round(float(np.mean(errors < 0.1)), 4),
        "approx_horizontal_below_camera_area_m2": round(approx_horizontal_area_m2, 2),
        "navigation_status": "not_certified",
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--limit", type=int)
    args = parser.parse_args()
    manifest = json.loads((ASSETS / "manifest.json").read_text())
    scenes = manifest["scenes"][:args.limit]
    results = []
    for scene in scenes:
        result = audit(ASSETS / scene["video_id"])
        results.append(result)
        print(f"{result['video_id']} visit={result['visit_id']} "
              f"depth/mesh<10cm={result['depth_to_mesh_within_10cm_fraction']:.1%} "
              f"horizontal_area≈{result['approx_horizontal_below_camera_area_m2']:.1f}m²",
              flush=True)
    summary = {
        "schema": "ape.real-capture-geometry-audit.v1",
        "audit_type": "sampled_RGBD_pose_to_mesh_and_horizontal_surface_screen",
        "not_qualified": "robot-height navmesh, collision, photoreal novel views, dynamic occupants",
        "scene_count": len(results), "unique_visit_count": len({r["visit_id"] for r in results}),
        "scenes": results,
    }
    (ASSETS / "geometry_audit.json").write_text(json.dumps(summary, indent=2) + "\n")


if __name__ == "__main__":
    main()
