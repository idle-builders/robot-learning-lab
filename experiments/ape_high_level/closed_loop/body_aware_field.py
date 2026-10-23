"""Actor-only target cost field with the selected Cove planar body envelope.

This is a route *hypothesis* over known and unknown cells. Only a caller's
known-free graph and independent swept-step guard may authorize motion.
"""

from __future__ import annotations

import heapq
import math

import numpy as np

from footprint_route import STATIC_MARGIN_M, scan_points
from goal_directed_frontier import (CORRIDOR_PAD_M, GRID_M, MAP_AGE_S,
                                    MAX_ROUTE_COST_M, MIN_FIELD_PROGRESS_M,
                                    NEIGHBORS, UNKNOWN_COST, cell, xy)
from goal_map_route import footprint_distance


def field_and_descent(
        cells: dict[tuple[int, int], tuple[float, float]],
        reachable_costs: dict[tuple[int, int], float],
        ranges: list[float | None], pose: np.ndarray,
        target_xy: np.ndarray, time_s: float,
        upper_hits_xy: np.ndarray, *, return_field: bool = False):
    origin = cell(pose[:2])
    xmin = math.floor((min(pose[0], target_xy[0]) - CORRIDOR_PAD_M) / GRID_M)
    xmax = math.ceil((max(pose[0], target_xy[0]) + CORRIDOR_PAD_M) / GRID_M)
    ymin = math.floor((min(pose[1], target_xy[1]) - CORRIDOR_PAD_M) / GRID_M)
    ymax = math.ceil((max(pose[1], target_xy[1]) + CORRIDOR_PAD_M) / GRID_M)

    def in_bounds(node: tuple[int, int]) -> bool:
        return xmin <= node[0] <= xmax and ymin <= node[1] <= ymax

    known_free = {node for node, (free_s, hit_s) in cells.items()
                  if free_s >= time_s - MAP_AGE_S and free_s > hit_s}
    mapped_hits = np.asarray([
        xy(node) for node, (free_s, hit_s) in cells.items()
        if hit_s >= free_s and hit_s >= time_s - MAP_AGE_S],
        dtype=float).reshape(-1, 2)
    upper = np.asarray(upper_hits_xy, dtype=float).reshape(-1, 2)
    obstacles = np.concatenate((mapped_hits, scan_points(ranges, pose), upper))
    blocked = set()
    for ix in range(xmin, xmax + 1):
        for iy in range(ymin, ymax + 1):
            node = (ix, iy)
            if node != origin and footprint_distance(
                    obstacles, xy(node), float(pose[2])) < STATIC_MARGIN_M:
                blocked.add(node)
    for point in upper:
        center = cell(point)
        for dx in (-1, 0, 1):
            for dy in (-1, 0, 1):
                blocked.add((center[0] + dx, center[1] + dy))
    blocked.discard(origin)
    known_free -= blocked

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
            candidate_cost = cost + GRID_M * math.hypot(dx, dy) * factor
            if candidate_cost + 1e-9 < field.get(neighbor, math.inf):
                field[neighbor] = candidate_cost
                heapq.heappush(queue, (candidate_cost, neighbor))

    origin_cost = field.get(origin, math.inf)
    candidates = []
    for node, travel_cost in reachable_costs.items():
        if (node == origin or node not in known_free or node not in field or
                travel_cost > MAX_ROUTE_COST_M):
            continue
        progress = origin_cost - field[node]
        if progress < MIN_FIELD_PROGRESS_M:
            continue
        candidates.append((travel_cost + field[node], node,
                           travel_cost, field[node], progress))
    best = min(candidates, default=None)
    debug = {"field_origin_cost_m": (None if not math.isfinite(origin_cost)
                                     else round(origin_cost, 4)),
             "field_cells": len(field), "field_goals": seed_goals,
             "body_blocked_cells": len(blocked),
             "known_free_cells": len(known_free),
             "reachable_descending_cells": len(candidates),
             "chosen_score_m": (None if best is None else round(best[0], 4)),
             "chosen_route_cost_m": (None if best is None else round(best[2], 4)),
             "chosen_remaining_cost_m": (None if best is None else round(best[3], 4)),
             "chosen_field_progress_m": (None if best is None else round(best[4], 4))}
    chosen = None if best is None else best[1]
    if return_field:
        return chosen, debug, field, known_free, blocked
    return chosen, debug
