"""Actor-only optimistic cost-to-target for choosing a true map frontier.

Unknown cells participate in the *goal hypothesis* only. The caller's
known-free route and fresh swept-body check own every executed step.
"""

from __future__ import annotations

import heapq
import math

import numpy as np


GRID_M = .04
MAP_AGE_S = 6.
UNKNOWN_COST = 2.5
CORRIDOR_PAD_M = .8
MAX_ROUTE_COST_M = 1.0
MIN_FIELD_PROGRESS_M = .04
VISIT_RADIUS_M = .12
VISIT_AGE_S = 12.
CARDINAL = ((1, 0), (-1, 0), (0, 1), (0, -1))
NEIGHBORS = CARDINAL + ((1, 1), (1, -1), (-1, 1), (-1, -1))


def cell(xy: np.ndarray) -> tuple[int, int]:
    return int(round(float(xy[0]) / GRID_M)), int(round(float(xy[1]) / GRID_M))


def xy(node: tuple[int, int]) -> np.ndarray:
    return GRID_M * np.asarray(node, dtype=float)


def choose_frontier(
        cells: dict[tuple[int, int], tuple[float, float]],
        reachable_costs: dict[tuple[int, int], float],
        pose_xy: np.ndarray, target_xy: np.ndarray, time_s: float,
        upper_hits_xy: np.ndarray,
        recent_visits: list[tuple[float, tuple[int, int]]]
        ) -> tuple[tuple[int, int] | None, dict]:
    origin = cell(pose_xy)
    xmin = math.floor((min(pose_xy[0], target_xy[0]) - CORRIDOR_PAD_M) / GRID_M)
    xmax = math.ceil((max(pose_xy[0], target_xy[0]) + CORRIDOR_PAD_M) / GRID_M)
    ymin = math.floor((min(pose_xy[1], target_xy[1]) - CORRIDOR_PAD_M) / GRID_M)
    ymax = math.ceil((max(pose_xy[1], target_xy[1]) + CORRIDOR_PAD_M) / GRID_M)

    def in_bounds(node: tuple[int, int]) -> bool:
        return xmin <= node[0] <= xmax and ymin <= node[1] <= ymax

    known_free = {node for node, (free_s, hit_s) in cells.items()
                  if free_s >= time_s - MAP_AGE_S and free_s > hit_s}
    blocked = {node for node, (free_s, hit_s) in cells.items()
               if hit_s >= free_s and hit_s >= time_s - MAP_AGE_S}
    for point in np.asarray(upper_hits_xy, dtype=float).reshape(-1, 2):
        center = cell(point)
        for dx in (-1, 0, 1):
            for dy in (-1, 0, 1):
                blocked.add((center[0] + dx, center[1] + dy))

    field: dict[tuple[int, int], float] = {}
    queue: list[tuple[float, tuple[int, int]]] = []
    for ix in range(xmin, xmax + 1):
        for iy in range(ymin, ymax + 1):
            node = (ix, iy)
            radius = float(np.linalg.norm(xy(node) - target_xy))
            if .27 <= radius <= .38 and node not in blocked:
                field[node] = 0.
                heapq.heappush(queue, (0., node))
    seed_goals = len(queue)
    while queue:
        cost, node = heapq.heappop(queue)
        if cost > field[node] + 1e-9:
            continue
        for dx, dy in NEIGHBORS:
            neighbor = (node[0] + dx, node[1] + dy)
            if not in_bounds(neighbor) or neighbor in blocked:
                continue
            factor = 1. if neighbor in known_free else UNKNOWN_COST
            new_cost = cost + GRID_M * math.hypot(dx, dy) * factor
            if new_cost + 1e-9 < field.get(neighbor, math.inf):
                field[neighbor] = new_cost
                heapq.heappush(queue, (new_cost, neighbor))

    origin_cost = field.get(origin, math.inf)
    candidates = []
    for node, travel_cost in reachable_costs.items():
        if (node == origin or node not in known_free or
                travel_cost > MAX_ROUTE_COST_M or node not in field):
            continue
        unknown_neighbors = sum(
            (node[0] + dx, node[1] + dy) not in cells
            for dx, dy in CARDINAL)
        if unknown_neighbors == 0:
            continue
        if origin_cost - field[node] < MIN_FIELD_PROGRESS_M:
            continue
        if any(time_s - stamp <= VISIT_AGE_S and
               np.linalg.norm(xy(node) - xy(visited)) < VISIT_RADIUS_M
               for stamp, visited in recent_visits):
            continue
        score = travel_cost + field[node] - .01 * unknown_neighbors
        candidates.append((score, node, travel_cost, field[node],
                           unknown_neighbors))
    best = min(candidates, default=None)
    debug = {"field_origin_cost_m": (None if not math.isfinite(origin_cost)
                                     else round(origin_cost, 4)),
             "field_cells": len(field), "field_goals": seed_goals,
             "true_frontier_candidates": len(candidates),
             "selected_goal_score_m": (None if best is None
                                       else round(best[0], 4)),
             "selected_known_route_cost_m": (None if best is None
                                             else round(best[2], 4)),
             "selected_optimistic_remaining_m": (None if best is None
                                                 else round(best[3], 4)),
             "selected_unknown_neighbors": (None if best is None
                                            else best[4])}
    return (None if best is None else best[1]), debug
