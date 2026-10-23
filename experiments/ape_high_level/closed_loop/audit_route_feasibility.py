"""Evaluator-only fixed-yaw route existence screen for v14 layouts."""

from __future__ import annotations

import argparse
from collections import deque
import json
from pathlib import Path

import numpy as np

from run import OUT, Room


def route(room: Room, *, margin: float, resolution: float = .04) -> dict:
    xs = np.arange(-.04, .88 + resolution / 2, resolution)
    ys = np.arange(-.76, .76 + resolution / 2, resolution)
    free = np.zeros((len(xs), len(ys)), dtype=bool)
    clearances = np.zeros_like(free, dtype=float)
    for ix, x in enumerate(xs):
        for iy, y in enumerate(ys):
            room.actual[:2] = [x, y]
            room.actual[2] = 0
            room._forward()
            clearance = room._current_obstacle_clearance()[0]
            clearances[ix, iy] = clearance
            free[ix, iy] = clearance >= margin
    start = (int(np.argmin(abs(xs))), int(np.argmin(abs(ys))))
    target = np.asarray(room.positions["TomatoSauce"])
    queue = deque([start])
    parent = {start: None}
    end = None
    while queue:
        cell = queue.popleft()
        xy = np.array([xs[cell[0]], ys[cell[1]]])
        if .25 <= np.linalg.norm(xy - target) <= .40 and free[cell]:
            end = cell
            break
        for dx, dy in ((-1, 0), (1, 0), (0, -1), (0, 1)):
            nxt = (cell[0] + dx, cell[1] + dy)
            if (0 <= nxt[0] < len(xs) and 0 <= nxt[1] < len(ys)
                    and free[nxt] and nxt not in parent):
                parent[nxt] = cell
                queue.append(nxt)
    path = []
    if end is not None:
        cell = end
        while cell is not None:
            path.append([round(float(xs[cell[0]]), 3),
                         round(float(ys[cell[1]]), 3)])
            cell = parent[cell]
        path.reverse()
    return {"route_exists": end is not None, "margin_m": margin,
            "fixed_yaw_rad": 0., "grid_resolution_m": resolution,
            "path_length_m": round((len(path) - 1) * resolution, 3) if path else None,
            "path_min_clearance_m": min((clearances[
                int(round((point[0] - xs[0]) / resolution)),
                int(round((point[1] - ys[0]) / resolution))] for point in path),
                default=None), "path_xy_m": path}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seeds", nargs="+", type=int, default=[35, 61, 62, 63, 64])
    parser.add_argument("--furnished", action="store_true",
                        help="Audit the furnished room placement")
    args = parser.parse_args()
    room = Room()
    rows = []
    try:
        for seed in args.seeds:
            for phase, y in (("blocking", .16), ("retreated", .40)):
                room.reset()
                room.layout(seed, absent=False, furnished=args.furnished)
                side = 1 if seed % 2 else -1
                room.obstacle_xy[3] = [.44, side * y]
                room.model.body_pos[room.model.body("loop_stool").id] = (
                    *room.obstacle_xy[3], .20)
                room._forward()
                rows.append({"seed": seed, "phase": phase,
                             "zero_margin": route(room, margin=0.),
                             "five_cm_margin": route(room, margin=.05)})
    finally:
        room.close()
    report = {"rows": rows,
              "note": "Evaluator-only fixed-yaw neutral-body grid screen; not an actor map or full gait proof."}
    OUT.mkdir(parents=True, exist_ok=True)
    name = ("furnished-route-feasibility-audit.json" if args.furnished
            else "route-feasibility-audit.json")
    (OUT / name).write_text(json.dumps(report, indent=2) + "\n")
    for row in rows:
        print(row["seed"], row["phase"],
              "zero", row["zero_margin"]["route_exists"],
              "5cm", row["five_cm_margin"]["route_exists"],
              "length", row["five_cm_margin"]["path_length_m"])


if __name__ == "__main__":
    main()
