"""Actor-only local map and persistent viewing-goal route for Cove-50.

Range rays mark free, occupied, and unknown grid cells. A fresh target location
defines an annulus of possible viewing poses, not a direct motion command.
The planner chooses a reachable pose, remembers it through replans, and issues
only one 8 cm step. The caller owns target freshness and checks each step with
the independent range guard and a subsequent camera observation.
"""

from __future__ import annotations

import heapq
import math
from typing import Callable

import numpy as np

from footprint_route import (MOVING_MARGIN_M, STATIC_MARGIN_M, scan_points)
from goal_directed_frontier import choose_frontier


GRID_M = .04
MAX_MAP_AGE_S = 6.0
MAX_ROUTE_M = 1.6
CLOSENESS_WEIGHT = 2.0
GOAL_REACHED_M = .035


def footprint_distance(points: np.ndarray, center: np.ndarray,
                       yaw: float) -> float:
    """Actor-side neutral-shell scan-plane proxy, calibrated offline on Cove-50.

    The arm and foot bands project farther sideways than the torso, but do not
    fill the front corners of one bounding rectangle. This is fixed body
    metadata, not access to current simulator contacts or obstacle geometry.
    """
    if len(points) == 0:
        return 10.
    delta = points - center
    c, s = math.cos(yaw), math.sin(yaw)
    local_x = c * delta[:, 0] + s * delta[:, 1]
    local_y = -s * delta[:, 0] + c * delta[:, 1]
    parts = ((0., 0., .055, .105),
             (0., .125, .045, .055), (0., -.125, .045, .055),
             (.01, .063, .065, .045), (.01, -.063, .065, .045))
    return min(float(np.min(np.hypot(np.maximum(np.abs(local_x - cx) - hx, 0.),
                                      np.maximum(np.abs(local_y - cy) - hy, 0.))))
               for cx, cy, hx, hy in parts)


class GoalMapRoute:
    def __init__(self) -> None:
        # Evidence is (last free-ray time, last hit time). An unseen cell is
        # unknown and cannot be traversed.
        self.cells: dict[tuple[int, int], tuple[float, float]] = {}
        self.goal: tuple[int, int] | None = None
        self.goal_kind: str | None = None
        self.goal_set_s = -math.inf
        self.failed: list[tuple[float, tuple[int, int]]] = []
        self.frontier_visits: list[tuple[float, tuple[int, int]]] = []
        self.last_target: np.ndarray | None = None
        self.last_debug: dict = {}
        self.last_reachable_costs: dict[tuple[int, int], float] = {}
        self.last_parent: dict[tuple[int, int], tuple[int, int]] = {}

    @staticmethod
    def cell(xy: np.ndarray) -> tuple[int, int]:
        return int(round(xy[0] / GRID_M)), int(round(xy[1] / GRID_M))

    @staticmethod
    def xy(cell: tuple[int, int]) -> np.ndarray:
        return GRID_M * np.asarray(cell, dtype=float)

    def observe(self, ranges: list[float | None], pose: np.ndarray,
                time_s: float) -> None:
        n = len(ranges)
        for index, reading in enumerate(ranges):
            if reading is None:
                if (ranges[(index - 1) % n] is not None or
                        ranges[(index + 1) % n] is not None):
                    continue
                length, hit = 1.12, False
            else:
                length, hit = min(float(reading), 1.25), reading < 1.25
            angle = pose[2] - math.pi + 2 * math.pi * index / n
            direction = np.array([math.cos(angle), math.sin(angle)])
            for distance in np.arange(.04, max(.04, length - .07), .04):
                cell = self.cell(pose[:2] + distance * direction)
                free_s, hit_s = self.cells.get(cell, (-math.inf, -math.inf))
                self.cells[cell] = (time_s, hit_s)
            if hit:
                cell = self.cell(pose[:2] + length * direction)
                free_s, hit_s = self.cells.get(cell, (-math.inf, -math.inf))
                self.cells[cell] = (free_s, time_s)
        self.cells = {cell: evidence for cell, evidence in self.cells.items()
                      if time_s - max(evidence) <= MAX_MAP_AGE_S}
        self.failed = [(stamp, cell) for stamp, cell in self.failed
                       if time_s - stamp <= 3.0]
        self.frontier_visits = [(stamp, cell)
                                for stamp, cell in self.frontier_visits
                                if time_s - stamp <= 12.0]

    def known_free(self, cell: tuple[int, int], time_s: float) -> bool:
        free_s, hit_s = self.cells.get(cell, (-math.inf, -math.inf))
        return free_s >= time_s - MAX_MAP_AGE_S and free_s > hit_s

    def occupied_points(self, time_s: float) -> np.ndarray:
        return np.asarray([self.xy(cell) for cell, (free_s, hit_s) in
                           self.cells.items() if hit_s >= free_s and
                           time_s - hit_s <= MAX_MAP_AGE_S],
                          dtype=float).reshape(-1, 2)

    @staticmethod
    def sightline_gap(points: np.ndarray, viewpoint: np.ndarray,
                      target: np.ndarray) -> float:
        segment = target - viewpoint
        length2 = float(segment @ segment)
        if length2 < .04 or len(points) == 0:
            return .2
        along = (points - viewpoint) @ segment / length2
        interior = (along > .16) & (along < .88)
        if not np.any(interior):
            return .2
        projection = viewpoint + along[interior, None] * segment
        return float(np.min(np.linalg.norm(points[interior] - projection,
                                           axis=1)))

    def plan(self, ranges: list[float | None], pose: np.ndarray,
             target_xy: np.ndarray, track: dict | None, time_s: float,
             *, searching: bool,
             extra_obstacles_xy: np.ndarray | None = None,
             allow_confirmed_frontier: bool = False,
             true_frontier: bool = False,
             viewpoint_evidence: Callable[[np.ndarray], dict] | None = None,
             local_progressive_view: bool = False,
             bounded_detour_view: bool = False,
             goal_directed_frontier: bool = False,
             edge_clearance: Callable[[np.ndarray, np.ndarray, float], float] | None = None,
             edge_clearance_margin_m: float = .065,
             local_all_point_edges: bool = False,
             point_all_graph_edges: bool = False,
             annulus_outer_m: float = .40,
             ) -> tuple[float, float, float, dict] | None:
        """Return a first step toward a persistent reachable viewing pose."""
        self.observe(ranges, pose, time_s)
        if self.last_target is not None and np.linalg.norm(
                target_xy - self.last_target) > .10:
            self.goal = None
            self.goal_kind = None
            self.failed.clear()
            self.frontier_visits.clear()
        self.last_target = target_xy.copy()
        if (self.goal is not None and self.goal_kind == "annulus" and
                np.linalg.norm(self.xy(self.goal) - target_xy) > annulus_outer_m):
            self.goal = None
            self.goal_kind = None
        current_points = scan_points(ranges, pose)
        if len(current_points) < 10:
            self.last_debug = {"reason": "range_coverage", "scan_points": len(current_points)}
            return None
        mapped_hits = self.occupied_points(time_s)
        extra = (np.asarray(extra_obstacles_xy, dtype=float).reshape(-1, 2)
                 if extra_obstacles_xy is not None else np.zeros((0, 2)))
        obstacles = np.concatenate((mapped_hits, current_points, extra))
        origin = self.cell(pose[:2])
        moving = (track is not None and track.get("status") == "tracked" and
                  track.get("center_xy_m") is not None and
                  track.get("velocity_xy_m_s") is not None)
        center = np.asarray(track["center_xy_m"]) if moving else None
        velocity = np.asarray(track["velocity_xy_m_s"]) if moving else None

        queue = [(0., 0, origin)]
        costs = {origin: 0.}
        parent: dict[tuple[int, int], tuple[int, int]] = {}
        serial = 0
        while queue and len(costs) < 1500:
            cost, _, node = heapq.heappop(queue)
            if cost > costs[node] + 1e-8 or cost > MAX_ROUTE_M:
                continue
            node_xy = pose[:2] if node == origin else self.xy(node)
            for dx, dy in ((1, 0), (-1, 0), (0, 1), (0, -1),
                           (1, 1), (1, -1), (-1, 1), (-1, -1)):
                nxt = node[0] + dx, node[1] + dy
                nxt_xy = self.xy(nxt)
                if np.linalg.norm(nxt_xy - pose[:2]) > 1.12:
                    continue
                if not self.known_free(nxt, time_s):
                    continue
                step = nxt_xy - node_xy
                samples = [node_xy + fraction * step for fraction in (.5, 1.)]
                flat_clear = min(footprint_distance(
                    obstacles, sample, pose[2]) for sample in samples)
                local_point_edge = (
                    edge_clearance is not None and
                    (point_all_graph_edges or node == origin or
                     (np.linalg.norm(node_xy - pose[:2]) <= .32 and
                      np.linalg.norm(nxt_xy - pose[:2]) <= .32 and
                      (local_all_point_edges or
                       flat_clear < STATIC_MARGIN_M))))
                clear = (edge_clearance(node_xy, nxt_xy, pose[2])
                         if local_point_edge else flat_clear)
                if clear < (edge_clearance_margin_m if local_point_edge
                            else STATIC_MARGIN_M):
                    continue
                new_cost = cost + float(np.linalg.norm(step)) + .002 / max(clear, .03)
                if moving:
                    arrival_s = min(2., new_cost / .20)
                    future = np.asarray([center + velocity * t
                                         for t in (arrival_s * .5, arrival_s)])
                    if min(footprint_distance(future, sample, pose[2])
                           for sample in samples) < MOVING_MARGIN_M:
                        continue
                if new_cost >= costs.get(nxt, math.inf):
                    continue
                costs[nxt] = new_cost
                parent[nxt] = node
                serial += 1
                heapq.heappush(queue, (new_cost, serial, nxt))

        self.last_reachable_costs = costs.copy()
        self.last_parent = parent.copy()
        self.last_debug = {"reachable_cells": len(costs),
                           "known_cells": len(self.cells),
                           "mapped_hits": len(mapped_hits),
                           "fresh_upper_obstacles": len(extra)}
        if len(costs) <= 1:
            self.last_debug["reason"] = "no_safe_neighbor"
            return None
        if self.goal is not None and self.goal not in costs:
            if self.goal_kind == "goal_frontier":
                self.frontier_visits.append((time_s, self.goal))
            self.failed.append((time_s, self.goal))
            self.goal = None
            self.goal_kind = None
        if (goal_directed_frontier and self.goal_kind == "goal_frontier" and
                any(.27 <= np.linalg.norm(self.xy(node) - target_xy) <= annulus_outer_m
                    for node in costs)):
            self.goal = None
            self.goal_kind = None
        if (self.goal is not None and self.goal_kind in
                ("view_ray", "view_ray_local", "view_ray_detour") and
                viewpoint_evidence is not None and not
                viewpoint_evidence(self.xy(self.goal))["ray_supported"]):
            self.failed.append((time_s, self.goal))
            self.goal = None
            self.goal_kind = None
        if (self.goal is not None and not searching and
                not allow_confirmed_frontier and self.goal_kind != "annulus"):
            # A frontier is an information-gathering goal, never an approach.
            self.goal = None
            self.goal_kind = None
        if self.goal is not None and np.linalg.norm(
                self.xy(self.goal) - pose[:2]) < GOAL_REACHED_M:
            # A reached search viewpoint without a fresh target is not useful.
            if searching or np.linalg.norm(target_xy - pose[:2]) > .34:
                if ((true_frontier and self.goal_kind == "frontier") or
                        self.goal_kind in ("view_ray", "view_ray_detour",
                                           "goal_frontier")):
                    self.frontier_visits.append((time_s, self.goal))
                self.failed.append((time_s, self.goal))
                self.goal = None
                self.goal_kind = None
        if self.goal is None:
            failed_cells = {cell for _, cell in self.failed}
            annulus = []
            frontier = []
            view_ray = []
            local_view_ray = []
            detour_view_ray = []
            true_frontier_count = 0
            field_debug = {}
            current_gap = self.sightline_gap(obstacles, pose[:2], target_xy)
            initial_distance = float(np.linalg.norm(target_xy - pose[:2]))
            for node, cost in costs.items():
                if node == origin or node in failed_cells:
                    continue
                xy = self.xy(node)
                target_distance = float(np.linalg.norm(target_xy - xy))
                gap = self.sightline_gap(obstacles, xy, target_xy)
                if (local_progressive_view and viewpoint_evidence is not None and
                        parent.get(node) == origin and
                        initial_distance - target_distance >= .01):
                    evidence = viewpoint_evidence(xy)
                    if evidence["ray_supported"]:
                        score = (initial_distance - target_distance -
                                 .15 * cost)
                        local_view_ray.append((-score, node))
                if (not local_progressive_view and viewpoint_evidence is not None and
                        not (node[0] % 2 or node[1] % 2) and
                        cost <= .9 and .27 <= target_distance <= .70 and
                        not any(np.linalg.norm(xy - self.xy(visited)) < .12
                                for _, visited in self.frontier_visits)):
                    cursor = node
                    path_steps = 0
                    while cursor != origin and cursor in parent and path_steps <= 20:
                        cursor = parent[cursor]
                        path_steps += 1
                    if cursor == origin and path_steps <= 20:
                        evidence = viewpoint_evidence(xy)
                        if evidence["ray_supported"]:
                            score = (initial_distance - target_distance -
                                     .15 * cost)
                            view_ray.append((-score, node))
                if (bounded_detour_view and viewpoint_evidence is not None and
                        not (node[0] % 2 or node[1] % 2) and
                        cost <= .9 and .27 <= target_distance and
                        target_distance <= initial_distance - .01 and
                        not any(np.linalg.norm(xy - self.xy(visited)) < .12
                                for _, visited in self.frontier_visits)):
                    evidence = viewpoint_evidence(xy)
                    if evidence["ray_supported"]:
                        detour_view_ray.append((target_distance + .18 * cost,
                                                node))
                if .27 <= target_distance <= annulus_outer_m:
                    # A cheap outer pose may reobserve the target yet remain
                    # outside the task's stop band after localization error.
                    # Pay some extra route cost for a closer, still-clear
                    # viewing pose rather than cycling around outer cells.
                    annulus.append((cost + CLOSENESS_WEIGHT * target_distance +
                                    .4 * max(.09 - gap, 0.), node))
                if (cost >= .12 and cost <= (1.2 if true_frontier else .9) and
                        target_distance <= initial_distance + .30):
                    if true_frontier:
                        unknown = [(dx, dy) for dx, dy in
                                   ((1, 0), (-1, 0), (0, 1), (0, -1))
                                   if (node[0] + dx, node[1] + dy)
                                   not in self.cells]
                        if not unknown:
                            continue
                        true_frontier_count += 1
                        if any(np.linalg.norm(xy - self.xy(visited)) < .12
                               for _, visited in self.frontier_visits):
                            continue
                        facing = sum(dx * (target_xy[0] - xy[0]) +
                                     dy * (target_xy[1] - xy[1]) > 0
                                     for dx, dy in unknown)
                        score = (target_distance + .15 * cost -
                                 .5 * min(gap, .2) - .05 * facing +
                                 (.15 if facing == 0 else 0.))
                    else:
                        unexplored_penalty = (.16 if target_distance >=
                                              initial_distance - .025 and gap <=
                                              current_gap + .025 else 0.)
                        score = (target_distance + .2 * cost -
                                 .5 * min(gap, .2) + unexplored_penalty)
                    frontier.append((score, node))
            best = min(annulus, default=None)
            kind = "annulus"
            if best is None and bounded_detour_view:
                best = min(detour_view_ray, default=None)
                kind = "view_ray_detour"
                if best is None:
                    best = min(view_ray, default=None)
                    kind = "view_ray"
                if best is None and goal_directed_frontier:
                    directed_goal, field_debug = choose_frontier(
                        self.cells, costs, pose[:2], target_xy, time_s,
                        extra, self.frontier_visits)
                    if directed_goal is not None:
                        best = (field_debug["selected_goal_score_m"],
                                directed_goal)
                        kind = "goal_frontier"
                if (best is None and not goal_directed_frontier and
                        (searching or allow_confirmed_frontier)):
                    best = min(frontier, default=None)
                    kind = "frontier"
            elif best is None and local_progressive_view:
                best = min(local_view_ray, default=None)
                kind = "view_ray_local"
            elif best is None and viewpoint_evidence is not None:
                best = min(view_ray, default=None)
                kind = "view_ray"
            elif best is None and (searching or allow_confirmed_frontier):
                best = min(frontier, default=None)
                kind = "frontier"
            if best is None:
                self.last_debug.update({"reason": ("no_goal" if searching
                                                   else "no_reachable_annulus"),
                                        "annulus_candidates": len(annulus),
                                        "frontier_candidates": len(frontier),
                                        "view_ray_candidates": len(view_ray),
                                        "local_view_ray_candidates": len(local_view_ray),
                                        "detour_view_ray_candidates": len(detour_view_ray),
                                        "goal_directed_field": field_debug,
                                        "true_frontier_boundary_cells":
                                            true_frontier_count})
                return None
            self.goal = best[1]
            self.goal_kind = kind
            self.goal_set_s = time_s

        node = self.goal
        path = []
        while node != origin:
            if node not in parent:
                return None
            path.append(node)
            node = parent[node]
        path.reverse()
        next_node = next((cell for cell in path
                          if np.linalg.norm(self.xy(cell) - pose[:2]) >= .025),
                         None)
        if next_node is None:
            self.failed.append((time_s, self.goal))
            self.goal = None
            self.goal_kind = None
            self.last_debug["reason"] = "goal_within_short_step"
            return None
        step = self.xy(next_node) - pose[:2]
        length = float(np.linalg.norm(step))
        heading = math.atan2(step[1], step[0])
        desired = math.atan2(target_xy[1] - pose[1], target_xy[0] - pose[0])
        offset = math.atan2(math.sin(heading - desired),
                            math.cos(heading - desired))
        step_distance = min(.08, length)
        commanded_step = step_distance * np.array([math.cos(heading), math.sin(heading)])
        first_step_points = np.concatenate((current_points, extra))
        clear = (edge_clearance(pose[:2], pose[:2] + commanded_step, pose[2])
                 if edge_clearance is not None else
                 min(footprint_distance(first_step_points,
                                        pose[:2] + fraction * commanded_step,
                                        pose[2]) for fraction in (.25, .5, .75, 1.)))
        if clear < (edge_clearance_margin_m if edge_clearance is not None
                    else STATIC_MARGIN_M):
            self.last_debug["reason"] = "first_step_clearance"
            return None
        details = {"goal_xy_m": self.xy(self.goal).tolist(),
                   "goal_kind": self.goal_kind,
                   "view_ray_evidence": (
                       viewpoint_evidence(self.xy(self.goal))
                       if viewpoint_evidence is not None and
                       self.goal_kind in ("view_ray", "view_ray_local",
                                          "view_ray_detour") else None),
                   "goal_age_s": round(time_s - self.goal_set_s, 3),
                   "path_remaining_cells": len(path),
                   "skipped_short_waypoints": path.index(next_node),
                   "step_distance_m": round(step_distance, 4),
                   "known_cells": len(self.cells),
                   "reachable_cells": len(costs),
                   "fresh_upper_obstacles": len(extra),
                   "confirmed_frontier_allowed": allow_confirmed_frontier,
                   "true_frontier": true_frontier,
                   "recent_frontier_visits": len(self.frontier_visits),
                   "goal_directed_field": field_debug if 'field_debug' in locals() else None,
                   "searching": searching}
        return heading, offset, clear, details
