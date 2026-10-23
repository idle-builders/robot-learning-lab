"""Choose an owner-show station with a separately reachable map-backed refuge.

This pre-show geometry gate uses only the actor's observed grid and owner range
track. The refuge is a reachable *option*, not a future moving-object safety
certificate; a fresh track and independent monitor still govern departure.
"""

from __future__ import annotations

import numpy as np


MEETING_OFFSETS_XY_M = ((.3, -.9), (.2, -.9), (.4, -.9),
                        (.3, -1.), (.2, -1.), (.4, -1.))
REFUGE_OFFSETS_XY_M = ((-.2, -.2), (-.3, -.1), (0., -.3),
                       (.2, -.2), (-.2, 0.))
MIN_OWNER_RESERVE_M = .25
BODY_RADIUS_M = .17
PROVISIONAL_GAP_M = .05
OBJECT_ALLOWANCE_M = .03
OWNER_DISPLACEMENT_ASSUMPTION_M = .60


def _known_free_segment(grid, origin: np.ndarray, goal: np.ndarray) -> bool:
    length = float(np.linalg.norm(goal - origin))
    for fraction in np.linspace(0., 1., max(2, int(np.ceil(length / .04)) + 1)):
        cell = grid.cell(origin + fraction * (goal - origin))
        if not grid.free[cell] or grid.occupied[cell]:
            return False
    return True


def choose(grid, track: dict, estimate: np.ndarray) -> tuple[dict | None, list[dict]]:
    origin = np.asarray(estimate[:2], dtype=float)
    envelope = (track["radius_m"] + BODY_RADIUS_M + PROVISIONAL_GAP_M +
                OBJECT_ALLOWANCE_M + OWNER_DISPLACEMENT_ASSUMPTION_M)
    rows = []
    for offset in MEETING_OFFSETS_XY_M:
        station = origin + offset
        reserve = float(np.linalg.norm(station - track["center_xy_m"]) - envelope)
        route_known = _known_free_segment(grid, origin, station)
        refuges = []
        if route_known and reserve >= MIN_OWNER_RESERVE_M:
            for refuge_offset in REFUGE_OFFSETS_XY_M:
                refuge = station + refuge_offset
                refuge_reserve = float(np.linalg.norm(refuge - track["center_xy_m"]) - envelope)
                refuges.append({
                    "target_xy_est_m": refuge.round(5).tolist(),
                    "offset_xy_m": refuge_offset,
                    "known_free_path": _known_free_segment(grid, station, refuge),
                    "owner_track_reserve_m": refuge_reserve,
                })
        supported_refuge = next((row for row in refuges if row["known_free_path"] and
                                 row["owner_track_reserve_m"] >= MIN_OWNER_RESERVE_M), None)
        row = {"distance_m": float(np.linalg.norm(offset)),
               "offset_xy_m": offset,
               "target_xy_est_m": station.round(5).tolist(),
               "owner_track_id": track["id"],
               "owner_track_reserve_m": reserve,
               "known_free_path": route_known,
               "refuge_candidates": refuges,
               "selected_refuge": supported_refuge}
        rows.append(row)
        if supported_refuge is not None:
            return row, rows
    return None, rows
