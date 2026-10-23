"""Screen captured meshes for provisional 50 cm proxy routes.

This deliberately retains unknown/fragmented cells as blocked. The screen is
not a physical collision certificate or a substitute for a selected body.
"""

from __future__ import annotations

import argparse
import gc
import json
from pathlib import Path

import cv2
import numpy as np
import trimesh
from scipy.spatial.transform import Rotation


ROOT = Path(__file__).resolve().parents[3]
ASSETS = ROOT / "assets/real_indoor_scenes/arkit_v1"
RESOLUTION_M = 0.05
PROXY_RADIUS_M = 0.17
PROXY_HEIGHT_M = 0.55


def camera_positions(scene_dir: Path) -> np.ndarray:
    poses = np.loadtxt(scene_dir / "camera.traj")
    return np.asarray([-Rotation.from_rotvec(row[1:4]).as_matrix().T @ row[4:7]
                       for row in poses])


def screen(scene_dir: Path) -> dict:
    mesh = trimesh.load(scene_dir / "mesh.ply", force="mesh", process=False)
    vertices = np.asarray(mesh.vertices)
    faces = np.asarray(mesh.faces)
    triangles = vertices[faces]
    z = triangles[:, :, 2]
    cross = np.cross(triangles[:, 1] - triangles[:, 0],
                     triangles[:, 2] - triangles[:, 0])
    twice_area = np.linalg.norm(cross, axis=1)
    horizontal = np.abs(cross[:, 2]) >= 0.9 * twice_area
    center_z = z.mean(axis=1)
    camera_xyz = camera_positions(scene_dir)
    lower = center_z[horizontal & (center_z < np.median(camera_xyz[:, 2]) - 0.25)]
    weights = twice_area[horizontal & (center_z < np.median(camera_xyz[:, 2]) - 0.25)] / 2
    if len(lower) == 0:
        raise ValueError(f"{scene_dir.name}: no horizontal surface below camera")
    bins = np.arange(np.floor(lower.min() / 0.05) * 0.05,
                     np.ceil(lower.max() / 0.05) * 0.05 + 0.051, 0.05)
    hist, edges = np.histogram(lower, bins=bins, weights=weights)
    mode = int(np.argmax(hist))
    floor_z = float((edges[mode] + edges[mode + 1]) / 2)
    xy_min = vertices[:, :2].min(axis=0)
    xy_max = vertices[:, :2].max(axis=0)
    width, height = (np.ceil((xy_max - xy_min) / RESOLUTION_M).astype(int) + 2).tolist()
    if width * height > 2_000_000:
        raise ValueError(f"{scene_dir.name}: unreasonable raster extent")
    pixels = np.rint((triangles[:, :, :2] - xy_min) / RESOLUTION_M).astype(np.int32)
    floor_faces = horizontal & (np.abs(center_z - floor_z) < 0.08)
    obstacle_faces = ((z.min(axis=1) < floor_z + PROXY_HEIGHT_M) &
                      (z.max(axis=1) > floor_z + 0.05) & ~floor_faces)
    floor_mask = np.zeros((height, width), np.uint8)
    obstacle_mask = np.zeros_like(floor_mask)
    for triangle in pixels[floor_faces]:
        cv2.fillConvexPoly(floor_mask, triangle, 255)
    for triangle in pixels[obstacle_faces]:
        cv2.fillConvexPoly(obstacle_mask, triangle, 255)
    radius_pixels = int(np.ceil(PROXY_RADIUS_M / RESOLUTION_M))
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE,
                                       (radius_pixels * 2 + 1, radius_pixels * 2 + 1))
    free = cv2.erode(floor_mask, kernel) & ~cv2.dilate(obstacle_mask, kernel)
    components, component_labels, stats, _ = cv2.connectedComponentsWithStats(free)
    areas = sorted((int(stats[i, cv2.CC_STAT_AREA]) for i in range(1, components)), reverse=True)
    main_component = np.uint8(component_labels == np.argmax(stats[1:, cv2.CC_STAT_AREA]) + 1) if areas else np.zeros_like(free)
    # Store a provisional grid so the next closed-loop renderer can use exactly
    # the same floor/unknown/obstacle decision as the visual audit.
    np.savez_compressed(scene_dir / "nav_candidate.npz", free=free > 0,
                        main_component=main_component > 0,
                        floor=floor_mask > 0, obstacles=obstacle_mask > 0,
                        floor_z_m=floor_z, xy_min_m=xy_min,
                        resolution_m=RESOLUTION_M, proxy_radius_m=PROXY_RADIUS_M,
                        proxy_height_m=PROXY_HEIGHT_M)
    preview = np.dstack((obstacle_mask, floor_mask // 3, free))
    cv2.imwrite(str(scene_dir / "nav_candidate.png"), cv2.resize(preview, None,
                 fx=3, fy=3, interpolation=cv2.INTER_NEAREST))
    del mesh, triangles, pixels
    gc.collect()
    return {
        "video_id": scene_dir.name, "floor_z_m": round(floor_z, 3),
        "floor_cells": int(np.count_nonzero(floor_mask)),
        "blocked_cells": int(np.count_nonzero(obstacle_mask)),
        "free_cells": int(np.count_nonzero(free)),
        "largest_component_area_m2": round((areas[0] if areas else 0) * RESOLUTION_M ** 2, 3),
        "components_above_1m2": sum(a * RESOLUTION_M ** 2 >= 1 for a in areas),
        "status": "provisional_proxy_screen_only",
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--limit", type=int)
    args = parser.parse_args()
    scenes = json.loads((ASSETS / "manifest.json").read_text())["scenes"][:args.limit]
    results = []
    for scene in scenes:
        try:
            result = screen(ASSETS / scene["video_id"])
        except Exception as exc:
            result = {"video_id": scene["video_id"], "status": "failed", "reason": str(exc)}
        results.append(result)
        print(f"{result['video_id']} {result['status']} "
              f"largest={result.get('largest_component_area_m2', 0):.2f}m²", flush=True)
    (ASSETS / "nav_screen.json").write_text(json.dumps({
        "schema": "ape.real-capture-nav-screen.v1", "proxy_radius_m": PROXY_RADIUS_M,
        "proxy_height_m": PROXY_HEIGHT_M, "resolution_m": RESOLUTION_M,
        "warning": "Rasterized mesh screen only; not a collision or selected-body certificate.",
        "scenes": results}, indent=2) + "\n")


if __name__ == "__main__":
    main()
