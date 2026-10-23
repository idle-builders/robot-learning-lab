"""Actor-only route-plus-hold choices from a confirmed predeparture range track."""

from __future__ import annotations

import numpy as np

import held_pose_screen


def screen(grid, estimate_xy_m, time_s: float, source: str,
           hold_until_s: float) -> dict:
    tracks = [track for track in grid.multi_history[-1]["tracks"]
              if track["status"] == "confirmed_now" and
              track["hits"] >= 3 and track["radius_m"] >= .18 and
              track["velocity_xy_m_s"] is not None and
              np.linalg.norm(track["velocity_xy_m_s"]) >= .1]
    alternatives = []
    for distance in (.30, .45, .60):
        for direction in range(16):
            angle = 2 * np.pi * direction / 16
            option = np.asarray(estimate_xy_m) + distance * np.array(
                [np.cos(angle), np.sin(angle)])
            request, state, path = grid.plan(estimate_xy_m, option)
            if state != "known_goal" or request is None:
                continue
            path_xy = [list(estimate_xy_m)] + [
                grid.xy(cell).tolist() for cell in path[1:]]
            screens = [{"track_id": track["id"],
                        "screen": held_pose_screen.screen(
                            path_xy_m=path_xy,
                            track_center_xy_m=track["center_xy_m"],
                            track_velocity_xy_m_s=track["velocity_xy_m_s"],
                            track_radius_m=track["radius_m"],
                            track_capture_time_s=track["last_seen_s"],
                            decision_time_s=time_s,
                            hold_until_s=hold_until_s,
                            pose_error_m=(.05 if source == "range" else .12))}
                       for track in tracks]
            alternatives.append({
                "goal_xy_m": option.round(5).tolist(),
                "distance_m": distance,
                "direction_index": direction,
                "path_cells": len(path),
                "early_future_hold_by_track": screens,
                "early_supported_under_models": bool(tracks) and all(
                    item["screen"]["supported_under_models"]
                    for item in screens)})
    supported = [candidate for candidate in alternatives
                 if candidate["early_supported_under_models"]]
    selected = (max(supported, key=lambda candidate: (
        min(row["screen"]["worst_modeled_gap_m"] for row in candidate[
            "early_future_hold_by_track"]),
        -candidate["early_future_hold_by_track"][0]["screen"][
            "path_length_m"])) if supported else None)
    return {"actor_tracks": tracks,
            "actor_track_ids": [track["id"] for track in tracks],
            "known_free_candidate_count": len(alternatives),
            "supported_candidate_count": len(supported),
            "alternatives": alternatives,
            "selected": selected}
