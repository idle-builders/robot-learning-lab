"""Convert a nominal pitched head-camera depth pixel to the body frame.

Body axes follow the candidate contract: x forward, y left, z up. The real
capture renderer puts image-right along body-left. The returned point is a
visible surface, not automatically the center of a person or object.
"""

from __future__ import annotations

import math

import numpy as np


def optical_pixel_to_body(u_px: float, v_px: float, optical_depth_m: float,
                          frame_position_m: tuple[float, float, float], *,
                          pitch_up_deg: float, vfov_deg: float,
                          width_px: int = 640, height_px: int = 360
                          ) -> tuple[float, float, float]:
    if not optical_depth_m > 0 or not math.isfinite(optical_depth_m):
        raise ValueError("A finite positive optical depth is required")
    focal = height_px / (2 * math.tan(math.radians(vfov_deg) / 2))
    cam_left = (u_px - width_px / 2) * optical_depth_m / focal
    cam_up = (height_px / 2 - v_px) * optical_depth_m / focal
    pitch = math.radians(pitch_up_deg)
    mount_forward, mount_left, mount_up = frame_position_m
    return (mount_forward + optical_depth_m * math.cos(pitch) -
            cam_up * math.sin(pitch),
            mount_left + cam_left,
            mount_up + optical_depth_m * math.sin(pitch) +
            cam_up * math.cos(pitch))


def body_point_to_optical(point_m: tuple[float, float, float],
                          frame_position_m: tuple[float, float, float], *,
                          pitch_up_deg: float, vfov_deg: float,
                          width_px: int = 640, height_px: int = 360
                          ) -> tuple[float, float, float]:
    focal = height_px / (2 * math.tan(math.radians(vfov_deg) / 2))
    forward = point_m[0] - frame_position_m[0]
    left = point_m[1] - frame_position_m[1]
    up = point_m[2] - frame_position_m[2]
    pitch = math.radians(pitch_up_deg)
    z = forward * math.cos(pitch) + up * math.sin(pitch)
    if z <= 0:
        raise ValueError("Point lies behind the camera")
    cam_up = -forward * math.sin(pitch) + up * math.cos(pitch)
    return (width_px / 2 + focal * left / z,
            height_px / 2 - focal * cam_up / z, z)


def observed_free_body_fan(
        views: tuple[tuple[np.ndarray, tuple[float, float, float], float], ...],
        angles_deg: np.ndarray, *, vfov_deg: float,
        planes_m: tuple[float, float] = (.1, .5),
        proxy_radius_m: float = .17, max_range_m: float = 2.,
        sample_m: float = .04) -> np.ndarray:
    """Conservative body-origin free beams from positioned depth cameras.

    A beam is clear only when at least one camera sees *every* sampled point
    from the current occupied proxy radius to max range. A missing or occluded
    point leaves the sector unknown; this adapter invents no obstacle hit.
    The proxy's existing occupied footprint is assumed collision-free at the
    captured state, not inferred from the camera. Clear-to-max is encoded as
    max+0.1 m so both the route map and owner actor can distinguish it from
    missing NaN; it is a sentinel, not a measured obstacle distance.
    """
    result = np.full((len(planes_m), len(angles_deg)), np.nan, np.float32)
    samples = np.arange(proxy_radius_m, max_range_m + .001, sample_m)
    for plane_index, height_m in enumerate(planes_m):
        for angle_index, angle_deg in enumerate(angles_deg):
            angle = math.radians(float(angle_deg))
            clear = True
            for distance in samples:
                # Route fan angles are positive toward body-right, whereas
                # this body frame has positive y toward body-left.
                point = (float(distance * math.cos(angle)),
                         float(-distance * math.sin(angle)), height_m)
                seen_free = False
                for depth, frame, pitch in views:
                    image_height, image_width = depth.shape
                    try:
                        u, v, optical_z = body_point_to_optical(
                            point, frame, pitch_up_deg=pitch,
                            vfov_deg=vfov_deg,
                            width_px=image_width, height_px=image_height)
                    except ValueError:
                        continue
                    ui, vi = round(u), round(v)
                    if not (0 <= ui < image_width and 0 <= vi < image_height):
                        continue
                    observed_z = float(depth[vi, ui])
                    focal = image_height / (2 * math.tan(math.radians(vfov_deg) / 2))
                    tolerance_z = max(.025, optical_z / focal)
                    if math.isfinite(observed_z) and \
                            observed_z + tolerance_z >= optical_z:
                        seen_free = True
                        break
                if not seen_free:
                    clear = False
                    break
            if clear:
                result[plane_index, angle_index] = max_range_m + .1
    return result


def observed_free_prefix_body_fan(
        views: tuple[tuple[np.ndarray, tuple[float, float, float], float], ...],
        angles_deg: np.ndarray, *, vfov_deg: float,
        planes_m: tuple[float, float] = (.1, .5),
        proxy_radius_m: float = .17, max_range_m: float = 2.,
        sample_m: float = .04) -> tuple[np.ndarray, np.ndarray]:
    """Certified free prefix of each body-origin ray, with first-loss reason.

    A positive result is *free space only* through the returned radius. Its
    endpoint is not an obstacle observation. Zero means even the first point
    outside the occupied proxy disk was unobserved. Reasons are ``clear``,
    ``outside_view``, ``missing_depth``, or ``nearer_surface``.
    """
    prefix = np.zeros((len(planes_m), len(angles_deg)), np.float32)
    reason = np.full(prefix.shape, "outside_view", dtype=object)
    samples = np.arange(proxy_radius_m, max_range_m + .001, sample_m)
    for plane_index, height_m in enumerate(planes_m):
        for angle_index, angle_deg in enumerate(angles_deg):
            angle = math.radians(float(angle_deg))
            last_clear = 0.
            for distance in samples:
                point = (float(distance * math.cos(angle)),
                         float(-distance * math.sin(angle)), height_m)
                seen_free = False
                in_view = False
                valid_depth = False
                for depth, frame, pitch in views:
                    image_height, image_width = depth.shape
                    try:
                        u, v, optical_z = body_point_to_optical(
                            point, frame, pitch_up_deg=pitch,
                            vfov_deg=vfov_deg, width_px=image_width,
                            height_px=image_height)
                    except ValueError:
                        continue
                    ui, vi = round(u), round(v)
                    if not (0 <= ui < image_width and 0 <= vi < image_height):
                        continue
                    in_view = True
                    observed_z = float(depth[vi, ui])
                    if not math.isfinite(observed_z):
                        continue
                    valid_depth = True
                    focal = image_height / (2 * math.tan(math.radians(vfov_deg) / 2))
                    tolerance_z = max(.025, optical_z / focal)
                    if observed_z + tolerance_z >= optical_z:
                        seen_free = True
                        break
                if not seen_free:
                    reason[plane_index, angle_index] = (
                        "nearer_surface" if valid_depth else
                        "missing_depth" if in_view else "outside_view")
                    break
                last_clear = float(distance)
            else:
                last_clear = max_range_m
                reason[plane_index, angle_index] = "clear"
            prefix[plane_index, angle_index] = last_clear
    return prefix, reason
