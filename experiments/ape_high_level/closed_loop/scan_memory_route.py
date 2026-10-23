"""Short-lived ray-cleared space and footprint-aware route for Cove-50 POC.

Only time-stamped actor scans and estimated poses enter this map. The current
scan supplies obstacle surfaces; recent scans supply evidence that a corridor
was actually observed free. No simulator geometry or target truth is used.
"""

from __future__ import annotations

from collections import deque
import heapq
import math

import numpy as np

from footprint_route import (MOVING_MARGIN_M, STATIC_MARGIN_M,
                             footprint_distance, scan_points)


GRID_M = .08
MAX_CELLS = 12
MAX_AGE_S = 2.0


class ScanMemoryRoute:
    def __init__(self) -> None:
        self.scans: deque[tuple[float, np.ndarray, list[float | None]]] = deque(
            maxlen=12)
        self.visited: deque[np.ndarray] = deque(maxlen=16)

    def observe(self, ranges: list[float | None], pose: np.ndarray,
                time_s: float) -> None:
        self.scans.append((time_s, pose.copy(), list(ranges)))
        while self.scans and time_s - self.scans[0][0] > MAX_AGE_S:
            self.scans.popleft()

    def visit(self, pose: np.ndarray) -> None:
        self.visited.append(pose[:2].copy())

    def ray_cleared(self, xy: np.ndarray) -> bool:
        """Require an actual recently observed ray to pass through this cell."""
        for _, pose, ranges in reversed(self.scans):
            delta = xy - pose[:2]
            radius = float(np.linalg.norm(delta))
            if radius < .04:
                return True
            if radius > 1.15:
                continue
            bearing = math.atan2(delta[1], delta[0]) - pose[2]
            index = int(round(((bearing + math.pi) % (2 * math.pi)) /
                              (2 * math.pi) * len(ranges))) % len(ranges)
            distance = ranges[index]
            if distance is None:
                # A lone missing return may be dropout. Adjacent missing
                # returns are treated as a no-hit corridor only to 1.15 m.
                if (ranges[(index - 1) % len(ranges)] is None and
                        ranges[(index + 1) % len(ranges)] is None):
                    return True
                continue
            if radius <= distance - .055:
                return True
        return False

    def step(self, ranges: list[float | None], pose: np.ndarray,
             target_xy: np.ndarray, track: dict | None
             ) -> tuple[float, float, float] | None:
        """A* to a target annulus or observed frontier; return one 8 cm step."""
        if not self.scans:
            return None
        points = scan_points(ranges, pose)
        if len(points) < 10:
            return None
        moving = (track is not None and track.get("status") == "tracked" and
                  track.get("center_xy_m") is not None and
                  track.get("velocity_xy_m_s") is not None)
        center = np.asarray(track["center_xy_m"]) if moving else None
        velocity = np.asarray(track["velocity_xy_m_s"]) if moving else None
        initial_distance = float(np.linalg.norm(target_xy - pose[:2]))
        desired = math.atan2(target_xy[1] - pose[1], target_xy[0] - pose[0])
        previous_distinct = next((
            old for old in reversed(self.visited)
            if float(np.linalg.norm(old - pose[:2])) > .05), None)

        def safe(xy: np.ndarray, cost: float) -> tuple[bool, float]:
            if not self.ray_cleared(xy):
                return False, 0.
            clearance = footprint_distance(points, xy, pose[2])
            if clearance < STATIC_MARGIN_M:
                return False, clearance
            if moving:
                arrival_s = min(2., cost / .20)
                future = np.asarray([center + velocity * t
                                     for t in (arrival_s * .5, arrival_s)])
                if footprint_distance(future, xy, pose[2]) < MOVING_MARGIN_M:
                    return False, clearance
            return True, clearance

        queue = [(max(initial_distance - .34, 0.), 0., 0, (0, 0))]
        costs = {(0, 0): 0.}
        first_steps: dict[tuple[int, int], tuple[int, int]] = {}
        frontier: tuple[float, tuple[int, int]] | None = None
        goal = None
        serial = 0
        expanded = 0
        while queue and expanded < 1200:
            _, cost, _, node = heapq.heappop(queue)
            if cost > costs[node] + 1e-8:
                continue
            expanded += 1
            xy = pose[:2] + GRID_M * np.asarray(node)
            remaining = float(np.linalg.norm(target_xy - xy))
            if node != (0, 0):
                first = first_steps[node]
                first_xy = pose[:2] + GRID_M * np.asarray(first)
                revisit = max((math.exp(-float(np.linalg.norm(first_xy - old))**2 /
                                         (.08**2)) for old in self.visited),
                              default=0.)
                # A useful frontier may initially be farther from the
                # target. The remembered route avoids choosing the same
                # failed first step after every scan.
                if max(abs(node[0]), abs(node[1])) <= 4:
                    candidate = (remaining + .10 * cost + .20 * revisit, node)
                    if frontier is None or candidate < frontier:
                        frontier = candidate
                if .27 <= remaining <= .36:
                    goal = node
                    break
            if cost > 1.6:
                continue
            for dx, dy in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                following = (node[0] + dx, node[1] + dy)
                if (node == (0, 0) and previous_distinct is not None and
                        np.linalg.norm(pose[:2] + GRID_M * np.asarray(following) -
                                       previous_distinct) < .055):
                    continue
                if max(abs(following[0]), abs(following[1])) > MAX_CELLS:
                    continue
                next_xy = pose[:2] + GRID_M * np.asarray(following)
                if np.linalg.norm(next_xy - target_xy) < .25:
                    continue
                samples = [xy + fraction * (next_xy - xy)
                           for fraction in (.25, .5, .75, 1.)]
                checks = [safe(sample, cost + .08 * fraction)
                          for fraction, sample in zip((.25, .5, .75, 1.),
                                                       samples)]
                if not all(ok for ok, _ in checks):
                    continue
                clear = min(clearance for _, clearance in checks)
                bearing = math.atan2(target_xy[1] - next_xy[1],
                                     target_xy[0] - next_xy[0])
                view_angle = abs(math.atan2(
                    math.sin(bearing - pose[2]), math.cos(bearing - pose[2])))
                view_cost = .015 * max(view_angle - .60, 0.)
                visit_cost = max((.06 * math.exp(
                    -float(np.linalg.norm(next_xy - old))**2 / (.08**2))
                    for old in self.visited), default=0.)
                next_cost = (cost + GRID_M + .003 / max(clear, .02) +
                             view_cost + visit_cost)
                if next_cost >= costs.get(following, math.inf):
                    continue
                costs[following] = next_cost
                first_steps[following] = (following if node == (0, 0)
                                          else first_steps[node])
                serial += 1
                heuristic = max(float(np.linalg.norm(target_xy - next_xy)) -
                                .34, 0.)
                heapq.heappush(queue, (next_cost + heuristic,
                                       next_cost, serial, following))
        chosen = goal if goal is not None else frontier[1] if frontier else None
        if chosen is None:
            return None
        first = first_steps[chosen]
        step = GRID_M * np.asarray(first)
        heading = math.atan2(step[1], step[0])
        clearance = min(footprint_distance(
            points, pose[:2] + fraction * step, pose[2])
            for fraction in (.25, .5, .75, 1.))
        offset = math.atan2(math.sin(heading - desired),
                            math.cos(heading - desired))
        return heading, offset, clearance
