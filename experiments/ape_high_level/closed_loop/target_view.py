"""Actor-only RGB-D target surface and short-horizon view evidence.

Depth crop points are a visible-surface hypothesis, not object segmentation.
Unknown voxels never become known-free or a motion-clearance certificate.
"""

from __future__ import annotations

from collections import Counter
import math

import numpy as np

from depth_voxel import VOXEL_M, DepthVoxelMap


def crop_surface_points(depth: np.ndarray, box: list[float],
                        camera_pose: tuple[np.ndarray, np.ndarray],
                        fovy_deg: float,
                        center_world: np.ndarray) -> np.ndarray:
    height, width = depth.shape
    origin, rotation = camera_pose
    box = np.asarray(box, dtype=float)
    center = np.asarray(center_world, dtype=float)
    focal = height / (2 * math.tan(math.radians(fovy_deg) / 2))
    center_u = int(np.clip(round((box[0] + box[2]) / 2), 0, width - 1))
    center_v = int(np.clip(round((box[1] + box[3]) / 2), 0, height - 1))
    center_depth = float(np.median(depth[max(0, center_v - 2):center_v + 3,
                                         max(0, center_u - 2):center_u + 3]))
    points = []
    for vf in np.linspace(.2, .8, 5):
        for uf in np.linspace(.2, .8, 5):
            u = float(box[0] + uf * (box[2] - box[0]))
            v = float(box[1] + vf * (box[3] - box[1]))
            x = int(np.clip(round(u), 0, width - 1))
            y = int(np.clip(round(v), 0, height - 1))
            distance = float(np.median(depth[max(0, y - 1):y + 2,
                                            max(0, x - 1):x + 2]))
            if not (.08 < distance < 1.4) or abs(distance - center_depth) > .06:
                continue
            ray = np.asarray([(u - width / 2) / focal,
                              -(v - height / 2) / focal, -1.])
            point = origin + rotation @ (distance * ray)
            if np.linalg.norm(point - center) <= .09:
                points.append(point)
    return np.asarray(points, dtype=float).reshape(-1, 3)


def in_frustum(origin: np.ndarray, rotation: np.ndarray,
               point_world: np.ndarray, fovy_deg: float,
               image_shape: tuple[int, int]) -> bool:
    height, width = image_shape
    point = rotation.T @ (point_world - origin)
    if point[2] >= -.08:
        return False
    focal = height / (2 * math.tan(math.radians(fovy_deg) / 2))
    u = width / 2 + focal * point[0] / -point[2]
    v = height / 2 - focal * point[1] / -point[2]
    return bool(0 <= u < width and 0 <= v < height)


def ray_status(voxels: DepthVoxelMap, origin: np.ndarray,
               target: np.ndarray, time_s: float) -> str:
    segment = target - origin
    length = float(np.linalg.norm(segment))
    if length < .12:
        return "unknown"
    states = Counter(voxels.state(origin + distance / length * segment, time_s)
                     for distance in np.arange(.08, max(.08, length - .06),
                                               VOXEL_M))
    return ("occupied" if states["occupied"] else
            "unknown" if states["unknown"] else "free")


def view_ray_evidence(voxels: DepthVoxelMap,
                      camera_pose: tuple[np.ndarray, np.ndarray],
                      surface: np.ndarray, fovy_deg: float,
                      image_shape: tuple[int, int], time_s: float) -> dict:
    origin, rotation = camera_pose
    states = Counter(ray_status(voxels, origin, point, time_s)
                     for point in surface
                     if in_frustum(origin, rotation, point, fovy_deg,
                                   image_shape))
    total = sum(states.values())
    return {"in_frustum_rays": total,
            "free_rays": states["free"],
            "occupied_rays": states["occupied"],
            "unknown_rays": states["unknown"],
            "ray_supported": bool(total and states["free"] >= 3 and
                                  states["free"] / total >= .25)}
