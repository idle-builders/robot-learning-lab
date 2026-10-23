"""Appearance and pinhole geometry baseline, with no simulator object input."""

from __future__ import annotations

from dataclasses import asdict, dataclass
import math

import cv2
import numpy as np

from scene import FOV_Y_DEG, IMAGE_HEIGHT, IMAGE_WIDTH


@dataclass(frozen=True)
class Detection:
    name: str
    xy_m: tuple[float, float]
    uncertainty_m: float
    pixels: int
    bbox_xywh: tuple[int, int, int, int]
    time_s: float

    def record(self):
        return asdict(self)


def focal_px() -> float:
    return IMAGE_HEIGHT / (2 * math.tan(math.radians(FOV_Y_DEG) / 2))


def frame_valid(rgb: np.ndarray) -> bool:
    """Conservative image-health gate; a blank frame cannot guide an action."""
    if rgb.shape != (IMAGE_HEIGHT, IMAGE_WIDTH, 3):
        raise ValueError("unexpected ape-eye image size")
    return bool(float(np.mean(rgb)) > 3.0 and float(np.std(rgb)) > 2.0)


def _mask(hsv: np.ndarray, name: str) -> np.ndarray:
    limits = {
        "red cube": [(0, 90, 45, 12, 255, 255), (170, 90, 45, 179, 255, 255)],
        "green cube": [(35, 65, 35, 90, 255, 255)],
        "blue cube": [(95, 150, 75, 140, 255, 255)],
        "yellow destination": [(16, 65, 45, 39, 255, 255)],
    }
    mask = np.zeros(hsv.shape[:2], dtype=np.uint8)
    for h0, s0, v0, h1, s1, v1 in limits[name]:
        mask |= cv2.inRange(hsv, (h0, s0, v0), (h1, s1, v1))
    return cv2.morphologyEx(mask, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))


def _ground_xy(u: float, v: float, height_m: float, camera_xyz: np.ndarray,
               camera_rotation: np.ndarray) -> tuple[float, float] | None:
    f = focal_px()
    # Local camera +X right, +Y up, -Z forward; image +V down.
    ray_local = np.array([(u - (IMAGE_WIDTH - 1) / 2) / f,
                          - (v - (IMAGE_HEIGHT - 1) / 2) / f, -1.0])
    ray = camera_rotation @ ray_local
    if ray[2] >= -1e-5:
        return None
    distance = (height_m - camera_xyz[2]) / ray[2]
    if distance <= 0:
        return None
    xy = camera_xyz[:2] + distance * ray[:2]
    return float(xy[0]), float(xy[1])


def detect(rgb: np.ndarray, camera_pose: tuple[np.ndarray, np.ndarray],
           time_s: float) -> list[Detection]:
    if rgb.shape != (IMAGE_HEIGHT, IMAGE_WIDTH, 3):
        raise ValueError("unexpected ape-eye image size")
    camera_xyz, camera_rotation = camera_pose
    hsv = cv2.cvtColor(rgb, cv2.COLOR_RGB2HSV)
    found = []
    for name in ("red cube", "green cube", "blue cube", "yellow destination"):
        count, labels, stats, centroids = cv2.connectedComponentsWithStats(_mask(hsv, name))
        for index in range(1, count):
            x, y, w, h, area = map(int, stats[index])
            if area < (45 if name != "yellow destination" else 70):
                continue
            # A clipped blob has a biased centroid and cannot support the
            # small position uncertainty used for a complete visible object.
            if x == 0 or y == 0 or x + w >= IMAGE_WIDTH or y + h >= IMAGE_HEIGHT:
                continue
            u, v = centroids[index]
            # Cube centroid is approximated at its known 10 mm centre height.
            # The flat destination is at z=1 mm.
            xy = _ground_xy(u, v, 0.01 if name != "yellow destination" else 0.001,
                            camera_xyz, camera_rotation)
            if xy is None:
                continue
            range_m = float(np.linalg.norm(np.array(xy) - camera_xyz[:2]))
            # A deliberately conservative pixel/shape uncertainty baseline.
            pixel_uncertainty = 2.0 + 60.0 / math.sqrt(area)
            uncertainty_m = max(0.012, range_m * pixel_uncertainty / focal_px())
            found.append(Detection(name, xy, uncertainty_m, area, (x, y, w, h), time_s))
    return found
