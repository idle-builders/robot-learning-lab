"""V241 evaluator-only fixed-yaw route witness from V239 final side views."""

from __future__ import annotations

import heapq
import json
import math
from pathlib import Path

import numpy as np

from audit_heldout_rooms_v175 import GRID_M, NEIGHBORS
from build_scene_pairs_v133 import body_geoms, full_body_clearance, scene_geoms
from run import ROOT, sha
from run_moved_props_v134 import MovedPropRoom
from run_moving_stool_short_depth_v229 import stool_y


SOURCE = Path(__file__)
OUT = ROOT / "artifacts/ape-closed-loop-cove50-v241"
V203 = ROOT / "artifacts/ape-closed-loop-cove50-v203/fourth-split-geometry-audit.json"
V239 = ROOT / "artifacts/ape-closed-loop-cove50-v239/active-view-moving-hole-live-audit.json"
MARGIN_M = .09
MAX_CELLS = 3500


def screen(seed: int, start: np.ndarray, stamp: float, positions: dict
           ) -> dict:
    room = MovedPropRoom({"CreamCheese": tuple(positions["CreamCheese"]),
                           "Parmesan": tuple(positions["Parmesan"])})
    try:
        room.reset()
        room.layout(seed, absent=False, furnished=True)
        room.model.body_pos[room.model.body("loop_table").id] = (.40, .46, 0.)
        room.set_table_height(.37)
        room.model.body_pos[room.model.body("loop_stool").id] = (
            .60, stool_y(stamp), .20)
        room.actual[:] = start
        room.estimated[:] = start
        room.head = 0.  # alternate neutral head posture, evaluator witness only
        room._forward()
        target = np.asarray(room.positions["TomatoSauce"], dtype=float)
        obstacles = scene_geoms(room)
        bodies = body_geoms(room, obstacles)
        cache = {}

        def clearance(xy: np.ndarray) -> float:
            key = tuple(np.round(xy, 5))
            if key not in cache:
                room.actual[:2] = xy
                room._forward()
                cache[key] = float(full_body_clearance(room, obstacles, bodies)[0])
            return cache[key]

        def xy(cell: tuple[int, int]) -> np.ndarray:
            return start[:2] + GRID_M * np.asarray(cell, dtype=float)

        start_gap = clearance(start[:2])
        queue = [(0., 0, (0, 0))]
        cost = {(0, 0): 0.}
        parent = {}
        serial = 0
        end = None
        while queue and len(cost) < MAX_CELLS:
            priority, _, cell = heapq.heappop(queue)
            point = xy(cell)
            heuristic = max(float(np.linalg.norm(point - target)) - .40, 0.)
            if priority > cost[cell] + heuristic + 1e-9:
                continue
            distance = float(np.linalg.norm(point - target))
            if .27 <= distance <= .40 and clearance(point) >= MARGIN_M:
                end = cell
                break
            for dx, dy in NEIGHBORS:
                nxt = (cell[0] + dx, cell[1] + dy)
                future = xy(nxt)
                if not (-.32 <= future[0] <= 1.16 and
                        -.72 <= future[1] <= .72):
                    continue
                step = future - point
                if min(clearance(point + fraction * step)
                       for fraction in np.linspace(0., 1., 21)) < MARGIN_M:
                    continue
                new_cost = cost[cell] + float(np.linalg.norm(step))
                if new_cost >= cost.get(nxt, math.inf):
                    continue
                cost[nxt] = new_cost
                parent[nxt] = cell
                serial += 1
                h = max(float(np.linalg.norm(future - target)) - .40, 0.)
                heapq.heappush(queue, (new_cost + h, serial, nxt))
        nodes = []
        if end is not None:
            cell = end
            while cell != (0, 0):
                nodes.append(cell)
                cell = parent[cell]
            nodes.append((0, 0))
            nodes.reverse()
        points = [xy(cell) for cell in nodes]
        edge_gaps = [clearance(a + fraction * (b - a))
                     for a, b in zip(points, points[1:])
                     for fraction in np.linspace(0., 1., 21)]
        return {"seed": seed, "final_time_s": stamp,
                "start_pose_xyyaw": start.tolist(),
                "stool_y_m": round(stool_y(stamp), 4),
                "target_xy_m": target.tolist(),
                "start_full_body_clearance_m": round(start_gap, 4),
                "fixed_yaw_neutral_head_route_found": end is not None,
                "route_steps": max(len(nodes) - 1, 0),
                "route_min_clearance_m": None if not edge_gaps else
                    round(min(edge_gaps), 4),
                "route_end_target_gap_m": None if end is None else
                    round(float(np.linalg.norm(points[-1] - target)), 4),
                "route_xy_m": [point.tolist() for point in points],
                "expanded_cells": len(cost), "sampled_poses": len(cache)}
    finally:
        room.close()


def main() -> None:
    scenes = {r["seed"]: r for r in json.loads(V203.read_text())["rows"]}
    live = {r["seed"]: r for r in json.loads(V239.read_text())["rows"]}
    rows = []
    for seed in (111, 113):
        trace = json.loads((ROOT / f"artifacts/ape-closed-loop-cove50-v239/"
                           f"scene{seed}/seed{seed}/trace.json").read_text())
        start = np.asarray(trace["evaluator_final_actual_xyyaw"], dtype=float)
        stamp = float(trace["events"][-1]["time_s"])
        positions = scenes[seed]["audits"]["0.065"]["positions_xy_m"]
        row = screen(seed, start, stamp, positions)
        rows.append(row)
        print(seed, row["fixed_yaw_neutral_head_route_found"],
              row["route_steps"], row["route_min_clearance_m"], flush=True)
    OUT.mkdir(parents=True, exist_ok=True)
    result = {"scope": "V241 evaluator-only 4cm-grid/21-sample full selected-body geometric route search from V239 final actual poses, dynamic stool frozen at each stop time, fixed actual body yaw, neutral head, all scene props/furniture, 9cm margin and 27-40cm target ring. It tests existence under these posture/search bounds, never provides a route to the actor and does not model native motion/stop.",
              "source_sha256": sha(SOURCE), "v203_sha256": sha(V203),
              "v239_sha256": sha(V239), "rows": rows}
    (OUT / "final-view-feasibility-audit.json").write_text(
        json.dumps(result, indent=2) + "\n")
    (OUT / SOURCE.name).write_bytes(SOURCE.read_bytes())


if __name__ == "__main__":
    main()
