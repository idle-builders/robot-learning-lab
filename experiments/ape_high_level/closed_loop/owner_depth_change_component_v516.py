"""Connected depth-change cloud and conservative lower-tail surface bound."""

from __future__ import annotations

import math

import cv2
import numpy as np


def packet_health(before, after):
    h, w = after.shape
    center = (slice(h//6, 5*h//6), slice(w//4, 3*w//4))
    fractions = []
    for packet in (before, after):
        patch = packet[center]
        fractions.append(float(np.mean(np.isfinite(patch) &
                                       (patch >= .15) & (patch <= 2.8))))
    return min(fractions) >= .5, fractions


def changed_points(before, after, rotation, camera_xyz_est, track_xy):
    height, width = after.shape
    valid = np.isfinite(after) & (after >= .15) & (after <= 2.8)
    closer = (before - after >= .05) | (~np.isfinite(before))
    raw = (valid & closer).astype(np.uint8)
    count, labels, stats, _ = cv2.connectedComponentsWithStats(raw, 8)
    supported = [i for i in range(1, count) if stats[i, cv2.CC_STAT_AREA] >= 50]
    mask = np.isin(labels, supported)
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


def choose_standoff(grid, changed_xyz_est, estimate,
                    motion_bound_m=.6, body_radius_m=.17, required_gap_m=.05,
                    pose_uncertainty_m=.03, depth_uncertainty_m=.03,
                    development_reserve_m=.10):
    bound = (motion_bound_m + body_radius_m + required_gap_m +
             pose_uncertainty_m + depth_uncertainty_m)
    rows = []
    for distance in np.arange(0., 1.001, .1):
        target = np.asarray(estimate[:2]) + [0., -distance]
        observed_distances = np.linalg.norm(changed_xyz_est[:, :2] - target,
                                             axis=1)
        # Discard only the lowest 1% of a spatially supported change cloud;
        # preserve the remaining near edge rather than averaging the person.
        near_surface = float(np.quantile(observed_distances, .01))
        reserve = near_surface - bound
        cells = []
        for amount in np.linspace(0., distance, int(round(distance/.05)) + 1):
            xy = np.asarray(estimate[:2]) + [0., -amount]
            cell = grid.cell(xy)
            cells.append({"xy_m": xy.tolist(),
                          "free": bool(grid.free[cell]),
                          "occupied": bool(grid.occupied[cell])})
        known = all(row["free"] and not row["occupied"] for row in cells)
        row = {"distance_m": round(float(distance), 3),
               "target_xy_est_m": target.tolist(),
               "lower_1pct_changed_surface_m": near_surface,
               "development_bound_m": bound,
               "reserve_m": reserve, "known_free_path": known,
               "cells": cells}
        rows.append(row)
        if reserve >= development_reserve_m and known:
            return row, rows
    return None, rows
