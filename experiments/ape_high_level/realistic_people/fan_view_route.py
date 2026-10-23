"""Short eye-proxy viewpoint route from observed forward range sectors.

This planner has no simulator geometry or person labels. It does not represent
the ape body: its configurable circular radius is a temporary kinematic
eye-proxy clearance for testing map/pose/action wiring, not a Cove-50 safety
envelope.
"""

from __future__ import annotations

import heapq
import math

import numpy as np


GRID_M = .08
MAP_AGE_S = 4.0
HIT_AGE_S = 2.0
PROXY_RADIUS_M = .12
MAX_RAY_M = 2.0
MAX_ROUTE_M = 1.5
REJECTED_STEP_AGE_S = 20.0
STANDOFF_MIN_M = 1.20
STANDOFF_MAX_M = 1.35


class FanViewRoute:
    def __init__(self, *, proxy_radius_m: float = PROXY_RADIUS_M,
                 planning_margin_m: float = 0.,
                 required_free_planes: tuple[int, ...] = (),
                 required_free_age_s: float = MAP_AGE_S,
                 standoff_min_m: float = STANDOFF_MIN_M,
                 standoff_max_m: float = STANDOFF_MAX_M,
                 max_route_m: float = MAX_ROUTE_M,
                 max_ray_m: float = MAX_RAY_M) -> None:
        self.proxy_radius_m = proxy_radius_m
        self.plan_clearance_m = proxy_radius_m + planning_margin_m
        self.required_free_planes = required_free_planes
        self.required_free_age_s = required_free_age_s
        if not 0 < standoff_min_m <= standoff_max_m or max_route_m <= 0 or max_ray_m <= 0:
            raise ValueError("Invalid route distance envelope")
        self.standoff_min_m = standoff_min_m
        self.standoff_max_m = standoff_max_m
        self.max_route_m = max_route_m
        self.max_ray_m = max_ray_m
        self.cells: dict[tuple[int, int], tuple[float, float]] = {}
        self.plane_free: list[dict[tuple[int, int], float]] = [{}, {}]
        self.hit_points: dict[tuple[int, int], tuple[np.ndarray, float]] = {}
        # A commanded step that did not occur is independent motion evidence.
        # Keep its endpoint blocked even if a range ray later marks it free.
        self.rejected_cells: dict[tuple[int, int], float] = {}
        self.goal: tuple[int, int] | None = None
        self.last_target: np.ndarray | None = None
        self.arrived = False
        self.outside_count = 0

    @staticmethod
    def cell(xy: np.ndarray) -> tuple[int, int]:
        return tuple(np.rint(np.asarray(xy) / GRID_M).astype(int))

    @staticmethod
    def xy(cell: tuple[int, int]) -> np.ndarray:
        return GRID_M * np.asarray(cell, dtype=float)

    def observe(self, distances: np.ndarray, angles_deg: np.ndarray,
                pose_xy: np.ndarray, heading_deg: float, time_s: float) -> dict:
        """`NaN` means missing sector; `inf` means measured clear to max range."""
        distances = np.asarray(distances, dtype=float)
        if distances.ndim != 2 or distances.shape[1] != len(angles_deg):
            raise ValueError("Expected height-by-angle range fan")
        if distances.shape[0] != 2:
            raise ValueError("Exactly two low/mid height plane slots are required")
        hit_count = 0
        free_count = 0
        origin_cell = self.cell(pose_xy)
        _, origin_hit_s = self.cells.get(origin_cell, (-math.inf, -math.inf))
        self.cells[origin_cell] = (time_s, origin_hit_s)
        for plane_free in self.plane_free:
            plane_free[origin_cell] = time_s
        for plane_index, plane in enumerate(distances):
            for angle_deg, reading in zip(angles_deg, plane):
                if np.isnan(reading):
                    continue
                hit = np.isfinite(reading) and reading <= self.max_ray_m
                length = min(float(reading), self.max_ray_m) if np.isfinite(reading) else self.max_ray_m
                bearing = math.radians(heading_deg + float(angle_deg))
                direction = np.array([math.sin(bearing), math.cos(bearing)])
                for distance in np.arange(.06, max(.06, length - .08), .04):
                    cell = self.cell(pose_xy + distance * direction)
                    free_s, hit_s = self.cells.get(cell, (-math.inf, -math.inf))
                    self.cells[cell] = (time_s, hit_s)
                    self.plane_free[plane_index][cell] = time_s
                    free_count += 1
                if hit:
                    hit_xy = pose_xy + length * direction
                    cell = self.cell(hit_xy)
                    free_s, hit_s = self.cells.get(cell, (-math.inf, -math.inf))
                    self.cells[cell] = (free_s, time_s)
                    self.hit_points[cell] = (hit_xy, time_s)
                    hit_count += 1
        self.cells = {cell: times for cell, times in self.cells.items()
                      if time_s - max(times) <= MAP_AGE_S}
        self.hit_points = {cell: record for cell, record in self.hit_points.items()
                           if cell in self.cells and time_s - record[1] <= HIT_AGE_S}
        self.plane_free = [{cell: seen_s for cell, seen_s in plane.items()
                            if time_s - seen_s <= MAP_AGE_S}
                           for plane in self.plane_free]
        self.rejected_cells = {
            cell: seen_s for cell, seen_s in self.rejected_cells.items()
            if time_s - seen_s <= REJECTED_STEP_AGE_S}
        return {"known_cells": len(self.cells), "free_ray_samples": free_count,
                "hit_beams": hit_count,
                "rejected_motion_cells": len(self.rejected_cells)}

    def observe_rejected_step(self, pose_xy: np.ndarray, heading_deg: float,
                              commanded_step_m: float, time_s: float) -> dict:
        """Invalidate the attempted endpoint without inventing obstacle shape."""
        if not commanded_step_m > 0:
            raise ValueError("Rejected step must have a positive commanded distance")
        direction = np.array([math.sin(math.radians(heading_deg)),
                              math.cos(math.radians(heading_deg))])
        start = self.cell(pose_xy)
        endpoint = np.asarray(pose_xy, dtype=float) + commanded_step_m * direction
        cell = self.cell(endpoint)
        if cell == start:
            cell = self.cell(np.asarray(pose_xy, dtype=float) +
                             max(commanded_step_m, GRID_M * 1.5) * direction)
        self.rejected_cells[cell] = time_s
        self.goal = None
        return {"rejected_endpoint_xy_m": endpoint.tolist(),
                "rejected_cell_xy_m": self.xy(cell).tolist(),
                "motion_feedback_time_s": time_s}

    def observe_free_prefix(self, prefix_m: np.ndarray, angles_deg: np.ndarray,
                            pose_xy: np.ndarray, heading_deg: float,
                            time_s: float) -> dict:
        """Add camera-certified free ray prefixes without inventing hits.

        A finite positive prefix certifies only the interior segment. The
        endpoint may be an occlusion, missing depth or the edge of the image;
        none of these is an obstacle at the body-origin ray endpoint.
        """
        prefixes = np.asarray(prefix_m, dtype=float)
        if prefixes.shape != (2, len(angles_deg)):
            raise ValueError("Expected two free-prefix planes by route angle")
        if np.any((prefixes < 0.) | (np.isfinite(prefixes) &
                                    (prefixes > self.max_ray_m))):
            raise ValueError("Free prefixes must lie between zero and the route horizon")
        free_samples = 0
        beams = 0
        for plane_index, plane in enumerate(prefixes):
            for angle_deg, length in zip(angles_deg, plane):
                if not np.isfinite(length) or length <= .17:
                    continue
                beams += 1
                bearing = math.radians(heading_deg + float(angle_deg))
                direction = np.array([math.sin(bearing), math.cos(bearing)])
                for distance in np.arange(.06, max(.06, length - .04), .04):
                    cell = self.cell(pose_xy + distance * direction)
                    free_s, hit_s = self.cells.get(cell, (-math.inf, -math.inf))
                    self.cells[cell] = (time_s, hit_s)
                    self.plane_free[plane_index][cell] = time_s
                    free_samples += 1
        return {"free_only_prefix_beams": beams,
                "free_only_ray_samples": free_samples}

    def observe_offset_low_scan(self, distances_m: np.ndarray,
                                angles_deg: np.ndarray,
                                sensor_xy_m: np.ndarray, heading_deg: float,
                                time_s: float, *, error_bound_m: float = .03) -> dict:
        """Insert a low-plane scan from its offset origin, conservatively.

        A finite return may be nearer than reported by ``error_bound_m``.
        Unknown samples remain absent; a positive infinity clears only the
        finite route horizon. This is mapping evidence, not a fast stop rule.
        """
        distances = np.asarray(distances_m, dtype=float)
        if distances.shape != np.asarray(angles_deg).shape:
            raise ValueError("Offset scan must match route angles")
        sensor = np.asarray(sensor_xy_m, dtype=float)
        if sensor.shape != (2,) or error_bound_m < 0:
            raise ValueError("Expected planar offset sensor and nonnegative error bound")
        free_samples = 0
        hits = 0
        known_beams = 0
        for angle_deg, reading in zip(angles_deg, distances):
            if np.isnan(reading) or reading <= 0:
                continue
            known_beams += 1
            hit = np.isfinite(reading) and reading <= self.max_ray_m
            length = (max(.05, float(reading) - error_bound_m)
                      if hit else self.max_ray_m)
            bearing = math.radians(heading_deg + float(angle_deg))
            direction = np.array([math.sin(bearing), math.cos(bearing)])
            for distance in np.arange(.05, max(.05, length - .08), .04):
                cell = self.cell(sensor + distance * direction)
                free_s, hit_s = self.cells.get(cell, (-math.inf, -math.inf))
                self.cells[cell] = (time_s, hit_s)
                self.plane_free[0][cell] = time_s
                free_samples += 1
            if hit:
                hit_xy = sensor + length * direction
                cell = self.cell(hit_xy)
                free_s, hit_s = self.cells.get(cell, (-math.inf, -math.inf))
                self.cells[cell] = (free_s, time_s)
                self.hit_points[cell] = (hit_xy, time_s)
                hits += 1
        return {"offset_low_scan_known_beams": known_beams,
                "offset_low_scan_free_samples": free_samples,
                "offset_low_scan_hits": hits}

    def _known_free(self, cell: tuple[int, int], time_s: float) -> bool:
        if self.rejected_cells.get(cell, -math.inf) >= time_s - REJECTED_STEP_AGE_S:
            return False
        free_s, hit_s = self.cells.get(cell, (-math.inf, -math.inf))
        return (free_s >= time_s - MAP_AGE_S and free_s > hit_s and
                all(self.plane_free[index].get(cell, -math.inf) >=
                    time_s - self.required_free_age_s
                    for index in self.required_free_planes))

    def observe_hit_only(self, hit_ranges_m: np.ndarray, angles_deg: np.ndarray,
                         pose_xy: np.ndarray, heading_deg: float,
                         time_s: float) -> int:
        """Add RGB-D low-surface hits without inventing planar free rays."""
        hits = np.asarray(hit_ranges_m, dtype=float)
        if hits.shape != np.asarray(angles_deg).shape:
            raise ValueError("Hit-only fan must match route angles")
        count = 0
        for angle_deg, distance in zip(angles_deg, hits):
            if not np.isfinite(distance) or not .03 < distance <= self.max_ray_m:
                continue
            bearing = math.radians(heading_deg + float(angle_deg))
            point = np.asarray(pose_xy, dtype=float) + float(distance) * np.array(
                [math.sin(bearing), math.cos(bearing)])
            cell = self.cell(point)
            free_s, _ = self.cells.get(cell, (-math.inf, -math.inf))
            self.cells[cell] = (free_s, time_s)
            self.hit_points[cell] = (point, time_s)
            count += 1
        return count

    def _occupied(self, time_s: float) -> np.ndarray:
        return np.asarray([self.hit_points[cell][0] for cell, (free_s, hit_s) in
                           self.cells.items() if hit_s >= free_s and
                           cell in self.hit_points and
                           hit_s >= time_s - HIT_AGE_S],
                          dtype=float).reshape(-1, 2)

    def _clear_segment(self, start_xy: np.ndarray, end_xy: np.ndarray,
                       time_s: float, occupied: np.ndarray) -> bool:
        length = float(np.linalg.norm(end_xy - start_xy))
        for fraction in np.linspace(0., 1., max(2, math.ceil(length / .04) + 1)):
            point = start_xy + fraction * (end_xy - start_xy)
            if not self._known_free(self.cell(point), time_s):
                return False
            if (len(occupied) and
                    np.min(np.linalg.norm(occupied - point, axis=1)) < self.plan_clearance_m):
                return False
        return True

    def plan(self, distances: np.ndarray, angles_deg: np.ndarray,
             pose_xy: np.ndarray, heading_deg: float, target_xy: np.ndarray,
             time_s: float, *, hit_only_ranges_m: np.ndarray | None = None,
             free_only_prefix_m: np.ndarray | None = None) -> dict:
        pose_xy = np.asarray(pose_xy, dtype=float)
        target_xy = np.asarray(target_xy, dtype=float)
        evidence = self.observe(distances, angles_deg, pose_xy, heading_deg, time_s)
        evidence.update(self.observe_free_prefix(
            free_only_prefix_m, angles_deg, pose_xy, heading_deg, time_s)
            if free_only_prefix_m is not None else
            {"free_only_prefix_beams": 0, "free_only_ray_samples": 0})
        evidence["rgbd_low_hit_beams"] = (self.observe_hit_only(
            hit_only_ranges_m, angles_deg, pose_xy, heading_deg, time_s)
            if hit_only_ranges_m is not None else 0)
        target_distance = float(np.linalg.norm(target_xy - pose_xy))
        if self.last_target is not None and np.linalg.norm(target_xy - self.last_target) > .20:
            self.goal = None
            self.arrived = False
            self.outside_count = 0
        self.last_target = target_xy.copy()
        if target_distance <= self.standoff_max_m:
            self.arrived = True
            self.outside_count = 0
            return {"intent": "hold_viewpoint", "reason": "within_standoff",
                    "target_distance_m": target_distance, **evidence}
        if self.arrived:
            self.outside_count = (self.outside_count + 1 if
                                  target_distance > self.standoff_max_m + .10 else 0)
            if self.outside_count < 2:
                return {"intent": "hold_viewpoint",
                        "reason": "standoff_hysteresis_reobserve",
                        "target_distance_m": target_distance, **evidence}
            self.arrived = False
        occupied = self._occupied(time_s)
        start = self.cell(pose_xy)
        queue = [(0., start)]
        costs = {start: 0.}
        parent: dict[tuple[int, int], tuple[int, int]] = {}
        while queue and len(costs) < 1800:
            cost, cell = heapq.heappop(queue)
            if cost > costs[cell] + 1e-8 or cost > self.max_route_m:
                continue
            for dx, dy in ((1, 0), (-1, 0), (0, 1), (0, -1),
                           (1, 1), (1, -1), (-1, 1), (-1, -1)):
                neighbor = (cell[0] + dx, cell[1] + dy)
                point = self.xy(neighbor)
                if np.linalg.norm(point - pose_xy) > self.max_route_m:
                    continue
                if not self._known_free(neighbor, time_s):
                    continue
                midpoint = (point + self.xy(cell)) / 2
                if not self._known_free(self.cell(midpoint), time_s):
                    continue
                if len(occupied) and min(
                        np.min(np.linalg.norm(occupied - point, axis=1)),
                           np.min(np.linalg.norm(occupied - midpoint, axis=1))) < self.plan_clearance_m:
                    continue
                step = float(np.linalg.norm(point - self.xy(cell)))
                new_cost = cost + step
                if new_cost < costs.get(neighbor, math.inf):
                    costs[neighbor] = new_cost
                    parent[neighbor] = cell
                    heapq.heappush(queue, (new_cost, neighbor))
        candidates = []
        preferred_standoff_m = (self.standoff_min_m + self.standoff_max_m) / 2.
        for cell, cost in costs.items():
            if cell == start:
                continue
            distance = float(np.linalg.norm(self.xy(cell) - target_xy))
            if self.standoff_min_m <= distance <= self.standoff_max_m:
                candidates.append((cost + 2.0 * abs(distance - preferred_standoff_m), cell))
        if not candidates:
            return {"intent": "hold_reobserve", "reason": "no_known_free_standoff",
                    "target_distance_m": target_distance,
                    "reachable_cells": len(costs), **evidence}
        best_score, best_goal = min(candidates)
        old_goal_score = next((score for score, cell in candidates
                               if cell == self.goal), math.inf)
        goal = (self.goal if old_goal_score <= best_score + .18 else best_goal)
        self.goal = goal
        path = []
        cursor = goal
        while cursor != start:
            path.append(cursor)
            cursor = parent[cursor]
        path.reverse()
        next_xy = self.xy(path[0])
        for waypoint in path[1:]:
            candidate_xy = self.xy(waypoint)
            if np.linalg.norm(candidate_xy - pose_xy) > .32:
                break
            if not self._clear_segment(pose_xy, candidate_xy, time_s, occupied):
                break
            next_xy = candidate_xy
        delta = next_xy - pose_xy
        desired_deg = math.degrees(math.atan2(delta[0], delta[1]))
        turn_deg = (desired_deg - heading_deg + 180) % 360 - 180
        common = {"target_xy_m": target_xy.tolist(),
                  "target_distance_m": target_distance,
                  "goal_xy_m": self.xy(goal).tolist(),
                  "next_waypoint_xy_m": next_xy.tolist(),
                  "path_cells": len(path), "reachable_cells": len(costs),
                  "occupied_cells": len(occupied), **evidence}
        if abs(turn_deg) > 5:
            return {**common, "intent": "turn_to_waypoint",
                    "yaw_delta_deg": float(np.clip(turn_deg, -10, 10))}
        step_m = min(.08, float(np.linalg.norm(delta)))
        next_pose = pose_xy + step_m * np.array([
            math.sin(math.radians(heading_deg)),
            math.cos(math.radians(heading_deg))])
        if not self._clear_segment(pose_xy, next_pose, time_s, occupied):
            return {**common, "intent": "hold_reobserve",
                    "reason": "next_step_not_known_clear"}
        return {**common, "intent": "step_to_waypoint", "forward_step_m": step_m}


class LowRangeClosingGuard:
    """Compare low hit clouds at the same new pose to remove ego range change.

    This warns about a closing surface within a short eye-proxy horizon. It
    does not track objects or certify a physical stopping distance.
    """

    def __init__(self, *, warning_clearance_m: float = .16,
                 minimum_closure_m: float = .012,
                 minimum_nearest_m: float | None = None) -> None:
        self.previous_hits: np.ndarray | None = None
        self.previous_time_s: float | None = None
        self.closing_streak = 0
        self.warning_clearance_m = warning_clearance_m
        self.minimum_closure_m = minimum_closure_m
        self.minimum_nearest_m = minimum_nearest_m

    def observe(self, distances: np.ndarray, angles_deg: np.ndarray,
                pose_xy: np.ndarray, heading_deg: float, time_s: float) -> dict:
        pose = np.asarray(pose_xy, dtype=float)
        readings = np.asarray(distances, dtype=float)
        angles = np.asarray(angles_deg, dtype=float)
        valid = np.isfinite(readings) & (readings > .03) & (readings < .8)
        bearing = np.radians(heading_deg + angles[valid])
        hits = pose + readings[valid, None] * np.column_stack(
            (np.sin(bearing), np.cos(bearing)))
        current_nearest = (float(np.min(np.linalg.norm(hits - pose, axis=1)))
                           if len(hits) else None)
        previous_nearest = (float(np.min(np.linalg.norm(self.previous_hits - pose,
                                                     axis=1)))
                            if self.previous_hits is not None and
                            len(self.previous_hits) else None)
        dt = (time_s - self.previous_time_s
              if self.previous_time_s is not None else None)
        residual = (previous_nearest - current_nearest
                    if previous_nearest is not None and
                    current_nearest is not None else None)
        valid_comparison = (len(hits) >= 5 and self.previous_hits is not None and
                            len(self.previous_hits) >= 5 and dt is not None and
                            .05 <= dt <= .40)
        closing = bool(valid_comparison and residual is not None and
                       residual >= self.minimum_closure_m)
        self.closing_streak = self.closing_streak + 1 if closing else 0
        closing_speed = (residual / dt if closing and dt is not None else 0.)
        # 0.8 s is a short planning horizon. The caller chooses a warning
        # distance above its temporary eye-proxy route clearance.
        projected_nearest = (current_nearest - .8 * closing_speed
                             if current_nearest is not None else None)
        predicted_threat = bool(self.closing_streak >= 2 and
                                projected_nearest is not None and
                                projected_nearest < self.warning_clearance_m)
        near_threat = bool(self.minimum_nearest_m is not None and
                           current_nearest is not None and
                           current_nearest < self.minimum_nearest_m)
        threatened = predicted_threat or near_threat
        self.previous_hits = hits
        self.previous_time_s = time_s
        return {"low_hit_count": len(hits),
                "nearest_low_hit_m": current_nearest,
                "previous_cloud_at_current_pose_m": previous_nearest,
                "unexplained_closure_m": residual,
                "closing_streak": self.closing_streak,
                "estimated_closing_speed_m_s": closing_speed,
                "projected_nearest_0p8s_m": projected_nearest,
                "predicted_threat": predicted_threat,
                "near_threat": near_threat,
                "threatened": threatened}


class PathTimedOccupancyGuard:
    """Screen the next bounded action against observed low-surface motion.

    The actor sees only ranges and its estimated pose. Adjacent beams form
    surface patches; three consistent world-frame edge displacements are needed
    before projecting them. A prediction can veto a step or request a known-
    free reverse, but never certifies that an unobserved volume is safe.
    """

    def __init__(self, *, clearance_m: float, horizon_s: float = .8) -> None:
        self.clearance_m = clearance_m
        self.horizon_s = horizon_s
        self.previous: list[dict] = []
        self.active: list[dict] = []
        self.last_observation: dict = {}

    @staticmethod
    def _patches(distances: np.ndarray, angles_deg: np.ndarray,
                 pose_xy: np.ndarray, heading_deg: float) -> list[np.ndarray]:
        readings = np.asarray(distances, dtype=float)
        angles = np.asarray(angles_deg, dtype=float)
        if readings.shape != angles.shape:
            raise ValueError("Low ranges and angles must have matching shapes")
        patches: list[np.ndarray] = []
        current: list[np.ndarray] = []
        previous_point = None
        for reading, angle in zip(readings, angles):
            if np.isfinite(reading) and .03 < reading < 1.0:
                bearing = math.radians(heading_deg + float(angle))
                point = pose_xy + float(reading) * np.array([
                    math.sin(bearing), math.cos(bearing)])
                if (previous_point is not None and
                        np.linalg.norm(point - previous_point) > .13):
                    if len(current) >= 3:
                        patches.append(np.asarray(current))
                    current = []
                current.append(point)
                previous_point = point
            else:
                if len(current) >= 3:
                    patches.append(np.asarray(current))
                current = []
                previous_point = None
        if len(current) >= 3:
            patches.append(np.asarray(current))
        return patches

    def observe(self, distances: np.ndarray, angles_deg: np.ndarray,
                pose_xy: np.ndarray, heading_deg: float, time_s: float) -> dict:
        pose_xy = np.asarray(pose_xy, dtype=float)
        patches = self._patches(distances, angles_deg, pose_xy, heading_deg)
        current = [{"points": points, "center": points.mean(axis=0),
                    "edges": np.stack((points.min(axis=0), points.max(axis=0))),
                    "time_s": time_s,
                    "edge_delta": np.zeros((2, 2)),
                    "edge_streak": np.zeros((2, 2), dtype=int),
                    "velocity": np.zeros(2)}
                   for points in patches]
        available = set(range(len(self.previous)))
        for patch in current:
            matches = sorted((float(np.linalg.norm(patch["center"] -
                                  self.previous[i]["center"])), i)
                             for i in available)
            if not matches or matches[0][0] > .18:
                continue
            _, index = matches[0]
            available.remove(index)
            previous = self.previous[index]
            dt = time_s - previous["time_s"]
            if not .05 <= dt <= .40:
                continue
            # A changing visible fraction can move a patch centroid even when
            # the surface is stationary. Require a persistent displacement of
            # one observed edge on each predicted axis instead.
            delta = patch["edges"] - previous["edges"]
            patch["edge_delta"] = delta
            for edge in range(2):
                for axis in range(2):
                    movement = delta[edge, axis]
                    if not .006 <= abs(movement) <= .12:
                        continue
                    old = previous["edge_delta"][edge, axis]
                    aligned = abs(old) >= .006 and movement * old > 0
                    patch["edge_streak"][edge, axis] = (
                        previous["edge_streak"][edge, axis] + 1
                        if aligned else 1)
            velocity = np.zeros(2)
            for axis in range(2):
                edge = int(np.argmax(patch["edge_streak"][:, axis]))
                if patch["edge_streak"][edge, axis] >= 3:
                    other = 1 - edge
                    other_now = delta[other, axis]
                    other_before = previous["edge_delta"][other, axis]
                    # A stable opposite edge means the apparent moving edge
                    # is changing visibility/width, not translating the
                    # whole patch. Opposed edges expanding likewise cannot
                    # support a rigid-motion prediction.
                    if (abs(other_now) < max(.004, .5 * abs(delta[edge, axis]))
                            and abs(other_before) < max(
                                .004, .5 * abs(previous["edge_delta"][edge, axis]))):
                        continue
                    if other_now * delta[edge, axis] < -.000036:
                        continue
                    component = delta[edge, axis] / dt
                    if abs(component) >= .06:
                        velocity[axis] = component
            speed = float(np.linalg.norm(velocity))
            if speed:
                patch["velocity"] = velocity * min(1., .30 / speed)
        self.previous = current
        self.active = [patch for patch in current
                       if np.linalg.norm(patch["velocity"]) >= .06]
        self.last_observation = {"patch_count": len(current),
            "moving_patch_count": len(self.active),
            "moving_patch_centers_xy_m": [p["center"].tolist() for p in self.active],
            "moving_patch_velocity_xy_m_s": [p["velocity"].tolist()
                                               for p in self.active]}
        return self.last_observation

    def assess(self, pose_xy: np.ndarray, heading_deg: float,
               proposed_step_m: float, source_age_s: float) -> dict:
        pose_xy = np.asarray(pose_xy, dtype=float)
        heading = math.radians(heading_deg)
        end_xy = pose_xy + proposed_step_m * np.array([
            math.sin(heading), math.cos(heading)])
        path_min = math.inf
        hold_min = math.inf
        for patch in self.active:
            points = patch["points"]
            velocity = patch["velocity"]
            for future_s in np.linspace(0., self.horizon_s, 9):
                shifted = points + velocity * (max(0., source_age_s) + future_s)
                # The bounded step is issued once, then the pose is held while
                # the obstacle continues moving through the horizon.
                fraction = min(1., future_s / .25)
                body_xy = pose_xy + fraction * (end_xy - pose_xy)
                path_min = min(path_min, float(np.min(
                    np.linalg.norm(shifted - body_xy, axis=1))))
                hold_min = min(hold_min, float(np.min(
                    np.linalg.norm(shifted - pose_xy, axis=1))))
        hold_threat = hold_min < self.clearance_m
        path_threat = path_min < self.clearance_m
        return {**self.last_observation,
            "proposed_step_m": proposed_step_m,
            "source_age_s": source_age_s,
            "horizon_s": self.horizon_s,
            "predicted_path_min_m": path_min if math.isfinite(path_min) else None,
            "predicted_hold_min_m": hold_min if math.isfinite(hold_min) else None,
            "path_threat": path_threat,
            "hold_threat": hold_threat,
            "action": ("retreat" if hold_threat else
                       "hold_reobserve" if path_threat else "allow")}
