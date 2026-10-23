"""Reopen stale moving-object trail cells only after fresh range clearance.

This changes an accumulated obstacle hit into observed free space only when
two adjacent finite beams reach beyond that cell with the body/grid margin.
Missing beams and cells outside confirmed moving-track history stay unknown or
occupied. The current tracked object footprint remains occupied.
"""

from __future__ import annotations

import numpy as np


def fresh_clear_mask(grid, ranges, pose_xy, max_range_m, inflation_m):
    readings = np.asarray([np.nan if value is None else value
                           for value in ranges], dtype=float)
    dx, dy = grid.xx - pose_xy[0], grid.yy - pose_xy[1]
    distance = np.hypot(dx, dy)
    beam = ((np.arctan2(dy, dx) + np.pi) / (2. * np.pi) * len(readings)) % len(readings)
    lower = np.floor(beam).astype(int)
    upper = (lower + 1) % len(readings)
    supporting_limit = np.minimum(readings[lower], readings[upper]) - inflation_m
    return (np.isfinite(supporting_limit) &
            (distance < supporting_limit) &
            (distance < max_range_m - inflation_m))


def reopen_observed_trail(grid, ranges, pose_xy, capture_time_s,
                          multi_history, max_range_m, inflation_m):
    latest = multi_history[-1]
    assert abs(latest["capture_time_s"] - capture_time_s) < .0001
    fresh_clear = fresh_clear_mask(grid, ranges, pose_xy,
                                  max_range_m, inflation_m)
    stale_trail = np.zeros_like(grid.occupied)
    current_footprints = np.zeros_like(grid.occupied)
    eligible_ids = []
    for track in latest["tracks"]:
        velocity = track["velocity_xy_m_s"]
        if (track["status"] != "confirmed_now" or track["hits"] < 3 or
                velocity is None or np.linalg.norm(velocity) < .1 or
                track["radius_m"] < .18):
            continue
        track_id = track["id"]
        radius = track["radius_m"] + inflation_m
        center = np.asarray(track["center_xy_m"])
        current_footprints |= np.hypot(grid.xx - center[0],
                                       grid.yy - center[1]) < radius
        prior_centers = []
        for snapshot in multi_history[:-1]:
            if snapshot["capture_time_s"] > capture_time_s - .1001:
                continue
            for previous in snapshot["tracks"]:
                if (previous["id"] == track_id and
                        previous["status"] == "confirmed_now" and
                        abs(previous["last_seen_s"] -
                            snapshot["capture_time_s"]) < .0001):
                    prior_centers.append(previous["center_xy_m"])
        if not prior_centers:
            continue
        eligible_ids.append(track_id)
        for old_center in prior_centers:
            stale_trail |= np.hypot(grid.xx - old_center[0],
                                    grid.yy - old_center[1]) < radius
    reclaimed = grid.occupied & stale_trail & fresh_clear & ~current_footprints
    grid.occupied[reclaimed] = False
    grid.free[reclaimed] = True
    return {"capture_time_s": round(float(capture_time_s), 4),
            "eligible_track_ids": eligible_ids,
            "fresh_clear_cell_count": int(np.count_nonzero(fresh_clear)),
            "stale_trail_cell_count": int(np.count_nonzero(stale_trail)),
            "reclaimed_cell_count": int(np.count_nonzero(reclaimed)),
            "reclaimed_indices_yx": np.argwhere(reclaimed).tolist()}
