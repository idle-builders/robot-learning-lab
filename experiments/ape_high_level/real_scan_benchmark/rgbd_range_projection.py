"""Conservative two-height route evidence from one forward RGB-D frame.

This is a camera-only candidate for comparing observability with a separate
horizontal range fan. Unknown pixels and sectors remain unknown; a ray that is
occluded at a different height never becomes invented planar free space.
"""

from __future__ import annotations

import math

import numpy as np


def projected_depth_fan(depth_m: np.ndarray, angles_deg: np.ndarray, *,
                        vfov_deg: float = 75., eye_height_m: float = .42,
                        camera_pitch_down_deg: float = 0.,
                        planes_m: tuple[float, float] = (.1, .5),
                        max_range_m: float = 2.) -> np.ndarray:
    height, width = depth_m.shape
    focal = height / (2 * math.tan(math.radians(vfov_deg) / 2))
    pitch = math.radians(camera_pitch_down_deg)
    result = np.full((len(planes_m), len(angles_deg)), np.nan, dtype=np.float32)
    # Sampling every 2 cm is finer than the 8 cm route grid. A sector is
    # measured-clear only if every projected sample to max range is visible.
    distances = np.arange(.12, max_range_m + .001, .02)
    for index, angle_deg in enumerate(angles_deg):
        angle = math.radians(float(angle_deg))
        cosine = math.cos(angle)
        if cosine <= 0:
            continue
        for plane_index, plane_height in enumerate(planes_m):
            reading = math.nan
            near_gap = False
            saw_valid_sample = False
            for radial_m in distances:
                vertical_m = plane_height - eye_height_m
                z = radial_m * cosine * math.cos(pitch) - vertical_m * math.sin(pitch)
                if z <= 0:
                    near_gap = True
                    continue
                u = int(round(width / 2 - focal * radial_m * math.sin(angle) / z))
                camera_up_m = radial_m * cosine * math.sin(pitch) + \
                              vertical_m * math.cos(pitch)
                v = int(round(height / 2 - focal * camera_up_m / z))
                if not (0 <= u < width and 0 <= v < height):
                    if not saw_valid_sample:
                        near_gap = True
                        continue
                    reading = math.nan
                    break
                seen_z = float(depth_m[v, u])
                if not math.isfinite(seen_z):
                    reading = math.nan
                    break
                saw_valid_sample = True
                # The half-pixel depth tolerance stops quantization of the
                # projected row from turning the floor into a false hit.
                tolerance_z = max(.025, radial_m / focal)
                if seen_z + tolerance_z < z:
                    hit_height = eye_height_m + vertical_m * seen_z / z
                    if not near_gap and abs(hit_height - plane_height) <= .08:
                        reading = radial_m * seen_z / z
                    else:
                        reading = math.nan
                    break
                if not near_gap:
                    reading = math.inf
            result[plane_index, index] = reading
    return result


def dual_view_fan(upright_depth_m: np.ndarray, downward_depth_m: np.ndarray,
                  angles_deg: np.ndarray, *, downward_pitch_deg: float = 45.) -> np.ndarray:
    upright = projected_depth_fan(upright_depth_m, angles_deg)
    downward = projected_depth_fan(
        downward_depth_m, angles_deg, camera_pitch_down_deg=downward_pitch_deg)
    # A finite obstacle in either view dominates a clear measurement; one
    # fully observed view can supply a sector unknown to the other.
    with np.errstate(invalid="ignore"):
        return np.fmin(upright, downward)
