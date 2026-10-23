"""One-step range-cleared detour when target-directed routing is blocked.

The planner sees only an actor range scan, estimated pose, latest confirmed
target coordinate and recent estimated positions. It never reads scene truth.
"""

from __future__ import annotations

import math

import numpy as np

from footprint_route import (MOVING_MARGIN_M, STATIC_MARGIN_M,
                             footprint_distance, observed_free, scan_points)


def detour_step(ranges: list[float | None], pose: np.ndarray,
                target_xy: np.ndarray, recent_xy: list[np.ndarray],
                track: dict | None) -> tuple[float, float, float] | None:
    """Choose a safe 8 cm first step, allowing temporary movement away."""
    points = scan_points(ranges, pose)
    if len(points) < 10:
        return None
    desired = math.atan2(target_xy[1] - pose[1], target_xy[0] - pose[0])
    moving = (track is not None and track.get("status") == "tracked" and
              track.get("center_xy_m") is not None and
              track.get("velocity_xy_m_s") is not None)
    center = np.asarray(track["center_xy_m"]) if moving else None
    velocity = np.asarray(track["velocity_xy_m_s"]) if moving else None
    choices = []
    for offset in np.linspace(-math.pi, math.pi, 49):
        heading = desired + float(offset)
        delta = .08 * np.array([math.cos(heading), math.sin(heading)])
        endpoint = pose[:2] + delta
        remaining = float(np.linalg.norm(target_xy - endpoint))
        if remaining < .25 or not observed_free(endpoint, pose, ranges):
            continue
        clearance = min(footprint_distance(
            points, pose[:2] + fraction * delta, pose[2])
            for fraction in (.25, .5, .75, 1.))
        if clearance < STATIC_MARGIN_M:
            continue
        if moving:
            future = np.asarray([center + velocity * t for t in (.2, .4)])
            moving_clearance = min(footprint_distance(
                future, pose[:2] + fraction * delta, pose[2])
                for fraction in (.5, 1.))
            if moving_clearance < MOVING_MARGIN_M:
                continue
        revisit = max((math.exp(-float(np.linalg.norm(endpoint - previous))**2 /
                                 (.08**2)) for previous in recent_xy[-10:]),
                      default=0.)
        score = (remaining + .18 * revisit + .012 * abs(float(offset)) -
                 .015 * min(clearance, .20))
        choices.append((score, abs(float(offset)), heading, float(offset),
                        clearance))
    if not choices:
        return None
    _, _, heading, offset, clearance = min(choices)
    return heading, offset, clearance
