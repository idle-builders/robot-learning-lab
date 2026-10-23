"""Select a mapped response route and finite waiting pose from actor packets.

The support claim is conditional on these declared motion, timing and body
assumptions. Scene truth and future prop scripts never enter this module.
"""

from __future__ import annotations

import heapq
import math
import time

import numpy as np


BODY_RADIUS_M = .17
REQUIRED_GAP_M = .05
POSE_TRACK_RESERVE_M = .10
RESPONSE_SPEED_M_S = .30
DISPATCH_DELAY_S = .12
WAIT_AFTER_ARRIVAL_S = .80
TURN_AFTER_S = .70
FORECAST_MAX_S = 4.0
ROUTE_SCALE = 1. + .30 / 5.89
PERSON_ASSOCIATION_M = .45
PERSON_MOTION_RESERVE_M = .10
UNKNOWN_MOTION_RESERVE_M = .60


def mapped_paths(grid, origin_xy):
    """One Dijkstra sweep, with the route planner's diagonal corner rule."""
    origin = grid.cell(origin_xy)
    if not grid.free[origin]:
        return {}
    queue = [(0., origin)]
    cost = {origin: 0.}
    parent = {}
    offsets = ((1, 0), (-1, 0), (0, 1), (0, -1),
               (1, 1), (1, -1), (-1, 1), (-1, -1))
    while queue:
        value, at = heapq.heappop(queue)
        if value > cost[at] + 1.e-8:
            continue
        for dy, dx in offsets:
            nxt = at[0] + dy, at[1] + dx
            if not (0 <= nxt[0] < len(grid.ys) and
                    0 <= nxt[1] < len(grid.xs) and grid.free[nxt]):
                continue
            if dx and dy and (not grid.free[at[0], nxt[1]] or
                              not grid.free[nxt[0], at[1]]):
                continue
            next_cost = (value + math.hypot(dx, dy) * .10 +
                         .02 * min(5, grid.visits[nxt]))
            if next_cost >= cost.get(nxt, math.inf):
                continue
            cost[nxt], parent[nxt] = next_cost, at
            heapq.heappush(queue, (next_cost, nxt))
    return {"origin": origin, "cost": cost, "parent": parent}


def path_to(grid, source, destination):
    target = grid.cell(destination)
    if target not in source["cost"]:
        return None
    cells = [target]
    while cells[-1] != source["origin"]:
        cells.append(source["parent"][cells[-1]])
    cells.reverse()
    return np.asarray([grid.xy(cell) for cell in cells])


def robot_trajectory(start, path, times):
    points = np.asarray([start] + [start + (point - path[0]) * ROUTE_SCALE
                                   for point in path[1:]])
    distance = np.r_[0., np.cumsum(np.linalg.norm(np.diff(points, axis=0), axis=1))]
    covered = np.minimum(distance[-1], RESPONSE_SPEED_M_S *
                         np.maximum(0., times - DISPATCH_DELAY_S))
    xy = np.column_stack((np.interp(covered, distance, points[:, 0]),
                          np.interp(covered, distance, points[:, 1])))
    return xy, float(distance[-1] / RESPONSE_SPEED_M_S + DISPATCH_DELAY_S)


def rotate(vector, degrees):
    angle = math.radians(degrees)
    return np.array([math.cos(angle) * vector[0] - math.sin(angle) * vector[1],
                     math.sin(angle) * vector[0] + math.cos(angle) * vector[1]])


def hazard_hypotheses(center, velocity, times):
    early = np.minimum(times, TURN_AFTER_S)
    late = np.maximum(0., times - TURN_AFTER_S)
    return {
        "straight": center + times[:, None] * velocity,
        "faster_straight": center + times[:, None] * (1.25 * velocity),
        "stops_after_0p7s": center + early[:, None] * velocity,
        "turn_left_45_after_0p7s": (center + early[:, None] * velocity +
                                      late[:, None] * rotate(velocity, 45)),
        "turn_right_45_after_0p7s": (center + early[:, None] * velocity +
                                       late[:, None] * rotate(velocity, -45)),
    }


def choose(grid, estimated_xy, track, velocity_xy_m_s, person_surface_xy_m,
           *, allow_best_effort=False, consider_goal_xy_m=None,
           require_initial_radial_away=False):
    begun = time.perf_counter()
    start = np.asarray(estimated_xy, dtype=float)
    center = np.asarray(track["center_xy_m"], dtype=float)
    velocity = np.asarray(velocity_xy_m_s, dtype=float)
    source = mapped_paths(grid, start)
    if not source:
        return None, {"status": "current_cell_unknown", "candidates": []}
    last_tracks = grid.multi_history[-1]["tracks"] if grid.multi_history else []
    other_tracks = []
    for other in last_tracks:
        xy = np.asarray(other["center_xy_m"])
        if np.linalg.norm(xy - center) < .30:
            continue
        person_match = (person_surface_xy_m is not None and
                        np.linalg.norm(xy - person_surface_xy_m) <=
                        PERSON_ASSOCIATION_M)
        envelope = (float(other["radius_m"]) + BODY_RADIUS_M + REQUIRED_GAP_M +
                    (PERSON_MOTION_RESERVE_M if person_match else
                     UNKNOWN_MOTION_RESERVE_M))
        other_tracks.append((xy, envelope,
                             "person" if person_match else "unknown"))
    targets = {}
    for radius in (.45, .65, .85):
        for angle in np.linspace(0., 2 * math.pi, 16, endpoint=False):
            goal = start + radius * np.array([math.cos(angle), math.sin(angle)])
            cell = grid.cell(goal)
            if cell not in targets:
                targets[cell] = grid.xy(cell)
    considered_cell = None
    if consider_goal_xy_m is not None:
        considered_cell = grid.cell(consider_goal_xy_m)
        targets[considered_cell] = grid.xy(considered_cell)
    rows = []
    for goal in targets.values():
        path = path_to(grid, source, goal)
        if path is None or len(path) < 2:
            continue
        # Every candidate is checked through travel and a short, renewable
        # held interval. The monitor must renew the forecast at 10 Hz.
        path_length = float(np.sum(np.linalg.norm(np.diff(path, axis=0), axis=1)))
        horizon = min(FORECAST_MAX_S, max(2.5, path_length * ROUTE_SCALE /
                                        RESPONSE_SPEED_M_S + DISPATCH_DELAY_S +
                                        WAIT_AFTER_ARRIVAL_S))
        times = np.arange(0., horizon + .025, .05)
        robot, arrival = robot_trajectory(start, path, times)
        gaps = {name: float(np.min(np.linalg.norm(robot - forecast, axis=1)) -
                            BODY_RADIUS_M - float(track["radius_m"]) -
                            POSE_TRACK_RESERVE_M)
                for name, forecast in hazard_hypotheses(center, velocity, times).items()}
        other_gap = min((float(np.min(np.linalg.norm(robot - xy, axis=1)) - envelope)
                         for xy, envelope, _ in other_tracks), default=math.inf)
        minimum = min(min(gaps.values()), other_gap)
        initial_step = path[1] - start
        radial_away = float(np.dot(
            initial_step, start - center) / max(
                np.linalg.norm(start - center), 1.e-6))
        rows.append({"goal_xy_m": goal.tolist(),
                     "route_cells": len(path),
                     "estimated_route_length_m": round(path_length, 4),
                     "arrival_s": round(arrival, 4),
                     "forecast_horizon_s": round(horizon, 4),
                     "hypothesis_min_gap_lower_m": {name: round(gap, 4)
                                                     for name, gap in gaps.items()},
                     "other_track_min_reserve_m": (None if math.isinf(other_gap)
                                                   else round(other_gap, 4)),
                     "minimum_predicted_gap_lower_m": round(minimum, 4),
                     "initial_step_radial_away_m": round(radial_away, 5),
                     "supported": minimum >= REQUIRED_GAP_M})
    ranked = sorted(rows, key=lambda row: (-row["minimum_predicted_gap_lower_m"],
                                            row["estimated_route_length_m"]))
    supported = [row for row in ranked if row["supported"]]
    choice = supported[0] if supported else None
    considered = next((row for row in rows if
                       considered_cell is not None and
                       grid.cell(row["goal_xy_m"]) == considered_cell), None)
    held_times = np.arange(0., FORECAST_MAX_S + .025, .05)
    held = np.broadcast_to(start, (len(held_times), 2))
    current_hold_lower = min(
        float(np.min(np.linalg.norm(held - forecast, axis=1)) -
              BODY_RADIUS_M - float(track["radius_m"]) - POSE_TRACK_RESERVE_M)
        for forecast in hazard_hypotheses(center, velocity, held_times).values())
    if choice is None and allow_best_effort:
        improved = [row for row in ranked
                    if row["minimum_predicted_gap_lower_m"] >=
                    current_hold_lower + .15 and
                    (row["other_track_min_reserve_m"] is None or
                     row["other_track_min_reserve_m"] >= REQUIRED_GAP_M) and
                    (not require_initial_radial_away or
                     row["initial_step_radial_away_m"] >= 0.)]
        choice = improved[0] if improved else None
    status = ("supported" if choice is not None and choice["supported"] else
              "unqualified_best_effort" if choice is not None else
              "no_supported_response")
    result = {"status": status,
              "chosen_goal_xy_m": None if choice is None else choice["goal_xy_m"],
              "chosen_minimum_gap_lower_m": (None if choice is None else
                                             choice["minimum_predicted_gap_lower_m"]),
              "candidate_count": len(rows),
              "supported_candidate_count": len(supported),
              "current_hold_predicted_gap_lower_m": round(current_hold_lower, 4),
              "top_candidates": ranked[:8],
              "considered_goal_evidence": considered,
              "motion_hypotheses": ["straight", "1.25x straight", "stop after 0.7s",
                                    "±45° turn after 0.7s"],
              "other_actor_tracks": [{"center_xy_m": xy.tolist(),
                                      "envelope_m": envelope, "role": role}
                                     for xy, envelope, role in other_tracks],
              "response_speed_m_s": RESPONSE_SPEED_M_S,
              "dispatch_delay_s": DISPATCH_DELAY_S,
              "renewal_interval_s": .1,
              "require_initial_radial_away": require_initial_radial_away,
              "selection_wall_s": time.perf_counter() - begun}
    return (None if choice is None else
            np.asarray(choice["goal_xy_m"], dtype=float)), result
