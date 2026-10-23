"""Project calibrated head depth into a low-obstacle bearing fan.

The output is positive evidence of a surface, not a scan of free space:
unobserved bins stay NaN.  Depth is Euclidean distance from the camera, as
returned by Isaac's ``distance_to_camera`` annotator.  The eye-only rig has a
fixed 45 cm camera height and a fixed upward pitch; an articulated ape would
need measured head pose and sensor extrinsics instead.
"""

from __future__ import annotations

import math

import numpy as np


def low_surface_fan(depth_m: np.ndarray, angles_deg: np.ndarray, *,
                    horizontal_fov_deg: float, camera_yaw_deg: float,
                    body_yaw_deg: float, camera_height_m: float = .45,
                    camera_pitch_deg: float = math.degrees(math.atan2(.25, 1.5)),
                    full_image_wh: tuple[int, int] = (640, 480),
                    pixel_stride: int = 3,
                    height_band_m: tuple[float, float] = (.06, .30),
                    min_depth_m: float = .25,
                    max_depth_m: float = 2.0,
                    min_support_pixels: int = 3) -> dict:
    """Return body-relative hit ranges and pixel support for low surfaces.

    No valid return is unknown, never a measured-clear ray.  Height filtering
    removes the floor but also leaves low objects outside the camera view
    unknown.  Callers must keep that distinction when merging with range.
    """
    depth = np.asarray(depth_m, dtype=np.float32)
    angles = np.asarray(angles_deg, dtype=np.float32)
    if depth.ndim != 2 or angles.ndim != 1 or len(angles) < 2:
        raise ValueError("Expected 2D depth and a nontrivial 1D angle fan")
    width, height = full_image_wh
    if (depth.shape[1] != math.ceil(width / pixel_stride) or
            depth.shape[0] != math.ceil(height / pixel_stride)):
        raise ValueError("Depth shape does not match full image and stride")
    interval = float(angles[1] - angles[0])
    if not np.allclose(np.diff(angles), interval, atol=1e-4):
        raise ValueError("Expected a uniformly spaced range fan")
    focal_px = width / (2.0 * math.tan(math.radians(horizontal_fov_deg / 2.0)))
    columns = (np.arange(depth.shape[1]) * pixel_stride + .5 - width / 2) / focal_px
    rows = -(np.arange(depth.shape[0]) * pixel_stride + .5 - height / 2) / focal_px
    u, v = np.meshgrid(columns, rows)
    pitch = math.radians(camera_pitch_deg)
    # Forward/up components in the horizontal camera plane; the positive
    # lateral axis points toward positive bearing, matching the PhysX fan.
    forward = math.cos(pitch) - v * math.sin(pitch)
    vertical = math.sin(pitch) + v * math.cos(pitch)
    normalizer = np.sqrt(1.0 + u * u + v * v)
    height_above_floor = camera_height_m + depth * vertical / normalizer
    horizontal_range = depth * np.hypot(u, forward) / normalizer
    bearing = np.degrees(np.arctan2(u, forward)) + camera_yaw_deg - body_yaw_deg
    bearing = (bearing + 180.0) % 360.0 - 180.0
    bin_index = np.rint((bearing - float(angles[0])) / interval).astype(np.int32)
    supported = (np.isfinite(depth) & (depth >= min_depth_m) &
                 (depth <= max_depth_m) &
                 (height_above_floor >= height_band_m[0]) &
                 (height_above_floor <= height_band_m[1]) &
                 (forward > 0.0) &
                 (bin_index >= 0) & (bin_index < len(angles)))
    # Each bin is a small angular cell. Reject points outside its half-width
    # instead of rounding an edge observation into a neighboring direction.
    in_bounds = np.clip(bin_index, 0, len(angles) - 1)
    supported &= np.abs(bearing - angles[in_bounds]) <= abs(interval) / 2 + 1e-4
    ranges = np.full(len(angles), np.inf, dtype=np.float32)
    counts = np.zeros(len(angles), dtype=np.int32)
    np.minimum.at(ranges, bin_index[supported], horizontal_range[supported])
    np.add.at(counts, bin_index[supported], 1)
    ranges[counts < min_support_pixels] = np.nan
    # These are hit positions only.  Passing them to a planar ray mapper as
    # if every intervening low-height cell were free would be incorrect:
    # camera rays tilt through the 3D volume.
    return {"hit_ranges_m": ranges, "support_pixels": counts,
            "candidate_pixels": int(supported.sum()),
            "hit_bins": int(np.isfinite(ranges).sum()),
            "min_horizontal_hit_m": (float(np.nanmin(ranges))
                if np.isfinite(ranges).any() else None)}
