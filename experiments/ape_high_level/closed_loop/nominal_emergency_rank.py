"""Rank currently free emergency routes under one observed straight track.

This deliberately ignores future turns and slow/late execution. It is only a
candidate-ordering experiment after the broader held-pose screen has already
reported that no route is supported, never a safety certificate.
"""

from __future__ import annotations

import numpy as np


def score(candidate: dict, track: dict, decision_time_s: float,
          hold_until_s: float, pose_error_m: float) -> dict:
    path = np.asarray(candidate["future_hold_screen"]["path_xy_m"], dtype=float)
    distances = np.r_[0., np.cumsum(np.linalg.norm(np.diff(path, axis=0), axis=1))]
    center = np.asarray(track["center_xy_m"], dtype=float)
    velocity = np.asarray(track["velocity_xy_m_s"], dtype=float)
    arrival_s = float(distances[-1] / .235)
    route_worst, hold_worst = float("inf"), float("inf")
    for elapsed in np.arange(0., hold_until_s - decision_time_s + .0001, .1):
        along = min(.235 * elapsed, distances[-1])
        ape = np.asarray([np.interp(along, distances, path[:, axis])
                          for axis in (0, 1)])
        age = decision_time_s + elapsed - track["last_seen_s"]
        prop = center + velocity * age
        gap = (float(np.linalg.norm(ape - prop)) - .17 -
               track["radius_m"] - pose_error_m - .03 - .02 * age)
        if elapsed < arrival_s:
            route_worst = min(route_worst, gap)
        else:
            hold_worst = min(hold_worst, gap)
    if not np.isfinite(hold_worst):
        hold_worst = route_worst
    return {"nominal_route_min_gap_m": round(route_worst, 5),
            "nominal_hold_min_gap_m": round(hold_worst, 5),
            "nominal_joint_min_gap_m": round(min(route_worst, hold_worst), 5),
            "nominal_arrival_s": round(arrival_s, 5)}


def rank(candidates: list[dict], track: dict, decision_time_s: float,
         hold_until_s: float, pose_error_m: float) -> list[dict]:
    available = [candidate for candidate in candidates
                 if candidate["plan_status"] == "known_goal" and
                 candidate["has_request"] and
                 "future_hold_screen" in candidate]
    scored = [{**candidate,
               "nominal_emergency_score": score(candidate, track,
                                                  decision_time_s,
                                                  hold_until_s,
                                                  pose_error_m)}
              for candidate in available]
    return sorted(scored, key=lambda row: (
        row["nominal_emergency_score"]["nominal_joint_min_gap_m"],
        row["nominal_emergency_score"]["nominal_hold_min_gap_m"],
        -row["nominal_emergency_score"]["nominal_arrival_s"]),
        reverse=True)
