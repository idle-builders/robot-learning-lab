"""Spatially supported before/after depth changes for owner presentation.

This is a candidate actor observation transform, not a person detector. It
requires connected depth changes before projecting them with calibrated
camera pose and filtering to the current range-track neighborhood.
"""

from __future__ import annotations

import math

import cv2
import numpy as np


def changed_points(before, after, rotation, camera_xyz_est, track_xy):
    height, width = after.shape
    valid = np.isfinite(after) & (after >= .15) & (after <= 2.8)
    closer = (before - after >= .05) | (~np.isfinite(before))
    raw = (valid & closer).astype(np.uint8)
    opened = cv2.morphologyEx(raw, cv2.MORPH_OPEN,
                              np.ones((3, 3), dtype=np.uint8))
    count, labels, stats, _ = cv2.connectedComponentsWithStats(opened, 8)
    keep_ids = [i for i in range(1, count) if stats[i, cv2.CC_STAT_AREA] >= 50]
    mask = np.isin(labels, keep_ids)
    u, v = np.meshgrid(np.arange(width), np.arange(height))
    focal = height / (2 * math.tan(math.radians(90.) / 2))
    camera_points = np.column_stack(((u[mask] - width/2) * after[mask] / focal,
                                     (height/2 - v[mask]) * after[mask] / focal,
                                     -after[mask]))
    xyz = np.asarray(camera_xyz_est) + camera_points @ np.asarray(rotation).T
    nearby = np.linalg.norm(xyz[:, :2] - track_xy, axis=1) <= .95
    above_floor = (xyz[:, 2] >= .24) & (xyz[:, 2] <= 1.55)
    accepted = nearby & above_floor
    return mask, xyz[accepted], (u[mask][accepted], v[mask][accepted])
