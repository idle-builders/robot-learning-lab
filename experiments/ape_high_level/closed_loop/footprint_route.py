"""Short receding-horizon route from a range scan and a nominal Cove footprint.

The route planner sees only the time-stamped actor scan, pose estimate, object
location, and optional range-motion track. It does not access simulator geometry.
"""

from __future__ import annotations

import heapq
import math

import numpy as np


GRID_M = .08
HALF_LENGTH_M = .11
HALF_WIDTH_M = .21
STATIC_MARGIN_M = .055
MOVING_MARGIN_M = .10
MAX_CELLS = 12


def scan_points(ranges: list[float | None], pose: np.ndarray) -> np.ndarray:
    angles = np.linspace(-math.pi, math.pi, len(ranges), endpoint=False)
    return np.asarray([pose[:2] + distance * np.array([
        math.cos(pose[2] + angle), math.sin(pose[2] + angle)])
        for angle, distance in zip(angles, ranges)
        if distance is not None and distance < 1.3], dtype=float).reshape(-1, 2)


def footprint_distance(points: np.ndarray, center: np.ndarray, yaw: float) -> float:
    """Nearest point to a conservative rectangle around nominal collision geoms."""
    if len(points) == 0:
        return 10.
    delta = points - center
    c, s = math.cos(yaw), math.sin(yaw)
    longitudinal = np.abs(c * delta[:, 0] + s * delta[:, 1])
    lateral = np.abs(-s * delta[:, 0] + c * delta[:, 1])
    dx = np.maximum(longitudinal - HALF_LENGTH_M, 0.)
    dy = np.maximum(lateral - HALF_WIDTH_M, 0.)
    return float(np.min(np.hypot(dx, dy)))


def observed_free(center: np.ndarray, pose: np.ndarray,
                  ranges: list[float | None]) -> bool:
    """The center must lie in current ray-cleared space, not behind a return."""
    delta = center - pose[:2]
    radius = float(np.linalg.norm(delta))
    if radius < .01:
        return True
    if radius > .92:
        return False
    bearing = math.atan2(delta[1], delta[0]) - pose[2]
    beam = int(round(((bearing + math.pi) % (2 * math.pi)) /
                     (2 * math.pi) * len(ranges))) % len(ranges)
    distance = ranges[beam]
    return distance is None or radius <= distance - .04


def turn_is_clear(ranges: list[float | None], pose: np.ndarray,
                  new_yaw: float, track: dict | None) -> bool:
    """Check the footprint sweep of an in-place camera/body turn from a scan."""
    points = scan_points(ranges, pose)
    if len(points) < 10:
        return False
    delta = math.atan2(math.sin(new_yaw - pose[2]),
                       math.cos(new_yaw - pose[2]))
    moving = (track is not None and track.get("status") == "tracked" and
              track.get("center_xy_m") is not None and
              track.get("velocity_xy_m_s") is not None)
    for fraction in (0., .25, .5, .75, 1.):
        yaw = pose[2] + fraction * delta
        if footprint_distance(points, pose[:2], yaw) < STATIC_MARGIN_M:
            return False
        if moving:
            surface = (np.asarray(track["center_xy_m"]) +
                       .4 * fraction * np.asarray(track["velocity_xy_m_s"]))
            if footprint_distance(surface.reshape(1, 2), pose[:2], yaw) < MOVING_MARGIN_M:
                return False
    return True


def route_step(ranges: list[float | None], pose: np.ndarray,
               target_xy: np.ndarray, track: dict | None,
               *, margin_m: float = STATIC_MARGIN_M) -> tuple[float, float, float] | None:
    """A* to the target annulus; issue just its first 8 cm step, then reobserve."""
    points = scan_points(ranges, pose)
    if len(points) < 10:
        return None
    moving = (track is not None and track.get("status") == "tracked" and
              track.get("center_xy_m") is not None and
              track.get("velocity_xy_m_s") is not None)
    center = np.asarray(track["center_xy_m"], dtype=float) if moving else None
    velocity = np.asarray(track["velocity_xy_m_s"], dtype=float) if moving else None
    # The scan itself already contains the current moving surface. The track's
    # centroid is a surface observation, not a simulator object center.
    initial_distance = float(np.linalg.norm(target_xy - pose[:2]))
    desired = math.atan2(target_xy[1] - pose[1], target_xy[0] - pose[0])
    neighbors = [(dx, dy) for dx in (-1, 0, 1) for dy in (-1, 0, 1)
                 if dx or dy]
    queue = [(max(initial_distance - .34, 0.), 0., 0, (0, 0))]
    costs = {(0, 0): 0.}
    first_steps: dict[tuple[int, int], tuple[int, int]] = {}
    best = None
    frontier: tuple[float, tuple[int, int]] | None = None
    serial = 0
    expanded = 0
    while queue and expanded < 1000:
        _, cost, _, node = heapq.heappop(queue)
        if cost > costs[node] + 1e-8:
            continue
        expanded += 1
        xy = pose[:2] + GRID_M * np.asarray(node)
        remaining = float(np.linalg.norm(target_xy - xy))
        if (node != (0, 0) and max(abs(node[0]), abs(node[1])) <= 3 and
                remaining < initial_distance - .025):
            # A range scan cannot certify free space behind the first hit.
            # Advance only to a visible frontier, then take another scan.
            candidate = (remaining + .12 * cost, node)
            if frontier is None or candidate < frontier:
                frontier = candidate
        if node != (0, 0) and .27 <= remaining <= .36:
            best = node
            break
        if cost > 1.5:
            continue
        for dx, dy in neighbors:
            following = (node[0] + dx, node[1] + dy)
            if max(abs(following[0]), abs(following[1])) > MAX_CELLS:
                continue
            next_xy = pose[:2] + GRID_M * np.asarray(following)
            if np.linalg.norm(next_xy - target_xy) < .25:
                continue
            if not observed_free(next_xy, pose, ranges):
                continue
            midpoint = (xy + next_xy) / 2
            clear = min(footprint_distance(points, point, pose[2])
                        for point in (midpoint, next_xy))
            if clear < margin_m:
                continue
            if moving:
                arrival_s = min(2.0, (cost + GRID_M * math.hypot(dx, dy)) / .20)
                # Inflate a moving point-track to reflect its visible surface,
                # motion estimate, and the 0.4 s action observation interval.
                moving_points = np.asarray([center + velocity * t
                                            for t in (arrival_s * .5, arrival_s)])
                if min(footprint_distance(moving_points, point, pose[2])
                       for point in (midpoint, next_xy)) < MOVING_MARGIN_M:
                    continue
            step_cost = GRID_M * math.hypot(dx, dy)
            step_cost += .004 / max(clear, .025)
            new_cost = cost + step_cost
            if new_cost >= costs.get(following, math.inf):
                continue
            costs[following] = new_cost
            first_steps[following] = following if node == (0, 0) else first_steps[node]
            serial += 1
            heuristic = max(float(np.linalg.norm(target_xy - next_xy)) - .34, 0.)
            heapq.heappush(queue, (new_cost + heuristic, new_cost, serial, following))
    if best is None and frontier is not None:
        best = frontier[1]
    if best is None:
        return None
    first = first_steps[best]
    heading = math.atan2(first[1], first[0])
    step = GRID_M * np.asarray(first)
    chosen_xy = pose[:2] + step
    clearance = min(footprint_distance(points, pose[:2] + fraction * step,
                                       pose[2]) for fraction in (.5, 1.))
    offset = math.atan2(math.sin(heading - desired), math.cos(heading - desired))
    return heading, offset, clearance
