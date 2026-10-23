"""Actor-only moving-track screen for a provisional escape path and later hold.

The hypotheses are explicit, finite and deliberately testable. A pass means
supported within this declared model, never a physical collision guarantee.
"""

from __future__ import annotations

import numpy as np


BODY_RADIUS_M = .17
REQUIRED_GAP_M = .05
OBJECT_CENTER_ERROR_AT_CAPTURE_M = .03
OBJECT_ERROR_GROWTH_M_S = .02
SPEEDS_M_S = (.145, .235)
DISPATCH_DELAYS_S = (0., .2)
TURN_AFTER_S = (.6, 1.2)
TURN_ANGLES_RAD = (-np.pi / 2., np.pi / 2.)
STEP_S = .1


def along_path(points: np.ndarray, distance_m: float) -> np.ndarray:
    segments = np.diff(points, axis=0)
    lengths = np.linalg.norm(segments, axis=1)
    left = distance_m
    for start, segment, length in zip(points[:-1], segments, lengths):
        if length <= 1.e-8:
            continue
        if left <= length:
            return start + segment * (left / length)
        left -= length
    return points[-1].copy()


def screen(*, path_xy_m: list[list[float]], track_center_xy_m: list[float],
           track_velocity_xy_m_s: list[float], track_radius_m: float,
           track_capture_time_s: float, decision_time_s: float,
           hold_until_s: float, pose_error_m: float) -> dict:
    """Project each path/hold against straight and ±90° later-turn hypotheses."""
    path = np.asarray(path_xy_m, dtype=float)
    assert path.ndim == 2 and path.shape[1] == 2 and len(path) >= 2
    center = np.asarray(track_center_xy_m, dtype=float)
    velocity = np.asarray(track_velocity_xy_m_s, dtype=float)
    assert np.linalg.norm(velocity) > 0.
    assert hold_until_s > decision_time_s >= track_capture_time_s
    track_age_s = decision_time_s - track_capture_time_s
    remaining_s = hold_until_s - decision_time_s
    path_length = float(np.sum(np.linalg.norm(np.diff(path, axis=0), axis=1)))
    slow_arrival_s = max(DISPATCH_DELAYS_S) + path_length / min(SPEEDS_M_S)
    modeled_hold_s = remaining_s - slow_arrival_s
    models = [("straight", None, 0.)] + [
        (f"turn_{int(np.rad2deg(angle)):+d}_after_{when:.1f}s", when, angle)
        for when in TURN_AFTER_S for angle in TURN_ANGLES_RAD]
    worst = {"gap_m": float("inf"), "model": None, "time_from_decision_s": None,
             "speed_m_s": None, "dispatch_delay_s": None}
    for speed in SPEEDS_M_S:
        for delay in DISPATCH_DELAYS_S:
            for name, turn_after, angle in models:
                rotation = np.array([[np.cos(angle), -np.sin(angle)],
                                     [np.sin(angle), np.cos(angle)]])
                turned_velocity = rotation @ velocity
                for elapsed in np.arange(0., remaining_s + STEP_S / 2., STEP_S):
                    observed_elapsed = track_age_s + elapsed
                    if turn_after is None or observed_elapsed <= turn_after:
                        object_xy = center + velocity * observed_elapsed
                    else:
                        object_xy = (center + velocity * turn_after +
                                     turned_velocity * (observed_elapsed - turn_after))
                    traveled = max(0., elapsed - delay) * speed
                    ape_xy = along_path(path, traveled)
                    gap = (float(np.linalg.norm(ape_xy - object_xy)) -
                           BODY_RADIUS_M - track_radius_m - pose_error_m -
                           OBJECT_CENTER_ERROR_AT_CAPTURE_M -
                           OBJECT_ERROR_GROWTH_M_S * observed_elapsed)
                    if gap < worst["gap_m"]:
                        worst = {"gap_m": gap, "model": name,
                                 "time_from_decision_s": float(elapsed),
                                 "speed_m_s": speed, "dispatch_delay_s": delay}
    return {"supported_under_models": bool(worst["gap_m"] >= REQUIRED_GAP_M and
                                             modeled_hold_s >= .4),
            "worst_modeled_gap_m": round(float(worst["gap_m"]), 5),
            "worst_hypothesis": worst["model"],
            "worst_time_from_decision_s": round(worst["time_from_decision_s"], 4),
            "worst_speed_m_s": worst["speed_m_s"],
            "worst_dispatch_delay_s": worst["dispatch_delay_s"],
            "path_length_m": round(path_length, 5),
            "slow_arrival_from_decision_s": round(float(slow_arrival_s), 5),
            "modeled_hold_duration_s": round(float(modeled_hold_s), 5),
            "track_age_at_decision_s": round(track_age_s, 5),
            "pose_error_assumption_m": pose_error_m,
            "path_xy_m": path.round(5).tolist()}
