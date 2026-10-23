"""V175: evaluator-only feasibility strata for frozen held-out furnished rooms."""

from __future__ import annotations

import heapq
import json
import math
import time
from pathlib import Path

import mujoco
import numpy as np

from build_scene_pairs_v133 import body_geoms, full_body_clearance, scene_geoms
from run import ROOT, Room, sha

SOURCE = Path(__file__)
OUT = ROOT / "artifacts/ape-closed-loop-cove50-v175"
GRID_M = .04
MARGIN_M = .065
MAX_CELLS = 3500
NEIGHBORS = ((1, 0), (-1, 0), (0, 1), (0, -1),
             (1, 1), (1, -1), (-1, 1), (-1, -1))


def audit_scene(room: Room, seed: int, *,
                prop_overrides: dict[str, tuple[float, float]] | None = None,
                output_dir: Path = OUT,
                model_margin_m: float = MARGIN_M) -> dict:
    room.reset()
    room.layout(seed, absent=False, furnished=True)
    room.model.body_pos[room.model.body("loop_table").id] = (
        .40, (1 if seed % 2 else -1) * .46, 0.)
    room.set_table_height(.37)
    for name, position in (prop_overrides or {}).items():
        room.positions[name] = position
        room.model.body_pos[room.model.body(f"loop_{name}").id] = (*position, .052)
    room._forward()
    target = np.asarray(room.positions["TomatoSauce"], dtype=float)
    start = room.actual[:2].copy()
    obstacles = scene_geoms(room)
    bodies = body_geoms(room, obstacles)
    object_ids = [room.model.geom(f"loop_{name}_geom").id
                  for name in ("TomatoSauce", "CreamCheese", "Parmesan")]
    scene_gap = min(
        (float(mujoco.mj_geomDistance(room.model, room.data, obj, other, 10., None)),
         room.model.geom(obj).name, room.model.geom(other).name)
        for obj in object_ids for other in obstacles
        if obj != other and not (other in object_ids and other < obj))
    cache: dict[tuple[float, float], float] = {}
    def clearance(point: np.ndarray) -> float:
        key = tuple(np.round(point, 5))
        if key not in cache:
            room.actual[:2] = point
            room._forward()
            cache[key] = full_body_clearance(room, obstacles, bodies)[0]
        return cache[key]
    def xy(node: tuple[int, int]) -> np.ndarray:
        return start + GRID_M * np.asarray(node, dtype=float)
    start_gap = clearance(start)
    queue = [(0., 0, (0, 0))]
    cost = {(0, 0): 0.}
    parent = {}
    serial = 0
    end = None
    begun = time.perf_counter()
    while queue and len(cost) < MAX_CELLS:
        priority, _, node = heapq.heappop(queue)
        point = xy(node)
        if priority > cost[node] + max(float(np.linalg.norm(point - target)) - .40, 0.) + 1e-9:
            continue
        distance = float(np.linalg.norm(point - target))
        if .27 <= distance <= .40 and clearance(point) >= model_margin_m:
            end = node
            break
        for dx, dy in NEIGHBORS:
            nxt = (node[0] + dx, node[1] + dy)
            future = xy(nxt)
            if not (-.32 <= future[0] <= 1.16 and -.72 <= future[1] <= .72):
                continue
            step = future - point
            if min(clearance(point + fraction * step)
                   for fraction in np.linspace(0., 1., 21)) < model_margin_m:
                continue
            new_cost = cost[node] + float(np.linalg.norm(step))
            if new_cost >= cost.get(nxt, math.inf):
                continue
            cost[nxt] = new_cost
            parent[nxt] = node
            serial += 1
            heuristic = max(float(np.linalg.norm(future - target)) - .40, 0.)
            heapq.heappush(queue, (new_cost + heuristic, serial, nxt))
    path_nodes = []
    if end is not None:
        node = end
        while node != (0, 0):
            path_nodes.append(node)
            node = parent[node]
        path_nodes.append((0, 0))
        path_nodes.reverse()
    points = [xy(node) for node in path_nodes]
    path_gaps = [clearance(a + fraction * (b - a))
                 for a, b in zip(points, points[1:])
                 for fraction in np.linspace(0., 1., 21)]
    room.actual[:] = 0.
    room._forward()
    image_path = output_dir / f"seed{seed}-external.png"
    room.capture_external().save(image_path)
    return {"seed": seed, "positions_xy_m": {k: list(v) for k, v in room.positions.items()},
            "native_light_scale": (.65, .85, 1., 1.15)[seed % 4],
            "object_scene_gap_m": round(scene_gap[0], 5),
            "object_scene_limit_geoms": list(scene_gap[1:]),
            "start_body_clearance_m": round(start_gap, 5),
            "route_model_margin_m": model_margin_m,
            "fixed_yaw_route_found": end is not None,
            "route_min_body_clearance_m": None if not path_gaps else round(min(path_gaps), 5),
            "route_steps": max(len(path_nodes) - 1, 0),
            "route_end_target_gap_m": None if end is None else
                round(float(np.linalg.norm(points[-1] - target)), 5),
            "route_xy_m": [p.tolist() for p in points],
            "expanded_cells": len(cost), "sampled_poses": len(cache),
            "search_s": round(time.perf_counter() - begun, 2),
            "external_image_sha256": sha(image_path)}


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    room = Room()
    try:
        rows = [audit_scene(room, seed) for seed in (89, 90, 91, 92)]
    finally:
        room.close()
    result = {"scope": "Evaluator-only frozen native furnished layouts seeds89-92, origin neutral selected body, fixed yaw 4cm-grid 21-sample-edge path to 27-40cm target ring at 6.5cm full-body clearance. No actor input or controller tuning.",
              "source_sha256": sha(SOURCE),
              "selected_design_sha256": sha(ROOT / "robot/ape-current-design.json"),
              "rows": rows}
    (OUT / "heldout-room-validity.json").write_text(json.dumps(result, indent=2) + "\n")
    (OUT / SOURCE.name).write_bytes(SOURCE.read_bytes())
    for row in rows:
        print(row["seed"], "scene gap", row["object_scene_gap_m"],
              "route", row["fixed_yaw_route_found"],
              "route clearance", row["route_min_body_clearance_m"], flush=True)


if __name__ == "__main__":
    main()
