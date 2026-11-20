"""Illustrative offset low-plane scanner packet for captured owner rooms.

The nominal timing/range error is inspired by a 10 Hz 2D scanner, but no real
hardware packet, body occlusion or material response is reproduced here.
"""

from __future__ import annotations

import hashlib
import math

import numpy as np


MOUNT_FORWARD_M = .08
MOUNT_HEIGHT_M = .1
FRONT_HALF_ARC_DEG = 60.
RANGE_HORIZON_M = 2.
BLIND_M = .05
ILLUSTRATIVE_ERROR_M = .03
ILLUSTRATIVE_DROPOUT = .05
SCAN_AGE_S = .05


def sample_front_low_scan(scene, body_xz: np.ndarray, heading_deg: float,
                          owner_xz: tuple[float, float],
                          angles_deg: np.ndarray, *, scene_id: str,
                          frame_index: int) -> dict:
    yaw = math.radians(heading_deg)
    forward = np.array([math.sin(yaw), math.cos(yaw)])
    sensor_xz = np.asarray(body_xz, dtype=float) + MOUNT_FORWARD_M * forward
    eye = np.array([sensor_xz[0], scene.floor_y + MOUNT_HEIGHT_M, sensor_xz[1]])
    true_ranges = scene.range_fan(
        eye, heading_deg, owner_xz=owner_xz,
        angles_deg=angles_deg, heights_m=(MOUNT_HEIGHT_M,))[0]
    seed_material = f"{scene_id}:{frame_index}:low-scan-v1".encode()
    seed = int.from_bytes(hashlib.sha256(seed_material).digest()[:8], "little")
    rng = np.random.default_rng(seed)
    ranges = np.asarray(true_ranges, dtype=np.float32).copy()
    within_arc = np.abs(np.asarray(angles_deg)) <= FRONT_HALF_ARC_DEG
    finite = np.isfinite(ranges)
    ranges[finite] += rng.uniform(
        -ILLUSTRATIVE_ERROR_M, ILLUSTRATIVE_ERROR_M, size=int(finite.sum()))
    ranges[(~within_arc) | (ranges < BLIND_M) |
           (rng.random(len(ranges)) < ILLUSTRATIVE_DROPOUT)] = np.nan
    ranges[ranges > RANGE_HORIZON_M] = np.inf
    return {"ranges_m": ranges, "sensor_xz_m": sensor_xz,
            "mount_height_m": MOUNT_HEIGHT_M,
            "mean_scan_age_s": SCAN_AGE_S,
            "scan_rate_hz": 10.,
            "valid_front_beams": int(np.count_nonzero(~np.isnan(ranges))),
            "finite_hits_within_horizon": int(np.count_nonzero(
                np.isfinite(ranges) & (ranges <= RANGE_HORIZON_M)))}
