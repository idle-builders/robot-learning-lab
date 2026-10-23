"""Explicit pinhole observation metadata; not physical lens calibration."""

import math
import numpy as np


def pinhole(width, height, vertical_fov_deg):
    if type(width) is not int or type(height) is not int or min(width, height) <= 0:
        raise ValueError("image dimensions must be positive integers")
    if not math.isfinite(vertical_fov_deg) or not 0 < vertical_fov_deg < 180:
        raise ValueError("vertical FOV must be finite and between 0 and 180 degrees")
    focal = height / (2 * math.tan(math.radians(vertical_fov_deg) / 2))
    return {
        "width": width,
        "height": height,
        "fx": focal,
        "fy": focal,
        "cx": (width - 1) / 2,
        "cy": (height - 1) / 2,
        "vertical_fov_deg": float(vertical_fov_deg),
        "horizontal_fov_deg": math.degrees(2 * math.atan(width / (2 * focal))),
        "model": "ideal_pinhole",
        "pixel_coordinates": "integer pixel centres; u right, v down",
        "optical_axes": "right, down, forward",
        "physical_calibration": "not_run",
        "limits": "no distortion, sensor-mode crop calibration, exposure, blur or autofocus simulation",
    }


def project_optical(points_m, intrinsics):
    """Project optical right/down/forward points; points behind camera return NaN."""
    points = np.asarray(points_m, dtype=float)
    if points.ndim != 2 or points.shape[1] != 3 or not np.isfinite(points).all():
        raise ValueError("finite Nx3 optical points required")
    front = points[:, 2] > 0
    pixels = np.full((len(points), 2), np.nan)
    pixels[front, 0] = (
        intrinsics["fx"] * points[front, 0] / points[front, 2] + intrinsics["cx"]
    )
    pixels[front, 1] = (
        intrinsics["fy"] * points[front, 1] / points[front, 2] + intrinsics["cy"]
    )
    inside = (
        front
        & (pixels[:, 0] >= -0.5)
        & (pixels[:, 0] < intrinsics["width"] - 0.5)
        & (pixels[:, 1] >= -0.5)
        & (pixels[:, 1] < intrinsics["height"] - 0.5)
    )
    return pixels, inside
