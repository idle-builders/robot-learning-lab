"""V125: exact selected-body full-scene path search after bounded yaw turns."""

from __future__ import annotations

import heapq
import json
import math
from pathlib import Path
import time

import numpy as np

from audit_fresh_recovery_approach_v110 import accepted_source
from run import ROOT, Room, sha

SOURCE = Path(__file__)
OUT = ROOT / "artifacts/ape-closed-loop-cove50-v125"
GRID = .04
MARGIN = .065
OFFSETS = (-1.2, -.6, .6, 1.2)
NEIGHBORS = ((1,0),(-1,0),(0,1),(0,-1),(1,1),(1,-1),(-1,1),(-1,-1))


def configure(room: Room, seed: int) -> tuple[np.ndarray, np.ndarray]:
    _, actual, paths, _, _ = accepted_source(seed)
    with np.load(paths[-1]) as z:
        qpos = z["actor_qpos"].copy()
    room.reset()
    room.layout(seed, absent=False, furnished=True)
    room.model.body_pos[room.model.body("loop_table").id] = (
        .40, (1 if seed % 2 else -1) * .46, 0.)
    room.set_table_height(.37)
    room.data.qpos[:] = qpos
    room.actual[:] = actual
    room.head = float(qpos[room.head_addr])
    room._forward()
    return actual.copy(), np.asarray(room.positions["TomatoSauce"][:2])


def search(room: Room, actual: np.ndarray, truth: np.ndarray, offset: float) -> dict:
    room.actual[:] = actual
    turn_dist = []
    for fraction in np.linspace(0., 1., 21):
        room.actual[2] = actual[2] + fraction * offset
        room._forward()
        turn_dist.append(float(room._current_obstacle_clearance()[0]))
    turn_min = min(turn_dist)
    if turn_min < MARGIN:
        return {"turn_min_m": round(turn_min,4), "turn_passes": False,
                "route_found": False, "verified_21_samples": False,
                "expanded_cells": 0, "sampled_model_positions": 0,
                "path_xy_m": []}
    room.actual[2] = actual[2] + offset
    start = actual[:2].copy()
    cache: dict[tuple[float,float],float] = {}
    def xy(node: tuple[int,int]) -> np.ndarray:
        return start + GRID * np.asarray(node)
    def clearance(point: np.ndarray) -> float:
        key = tuple(np.round(point,5))
        if key not in cache:
            room.actual[:2] = point
            room._forward()
            cache[key] = float(room._current_obstacle_clearance()[0])
        return cache[key]
    cost = {(0,0): 0.}
    parent: dict[tuple[int,int],tuple[int,int]] = {}
    queue = [(0.,0,(0,0))]
    serial = 0
    end = None
    start_time = time.perf_counter()
    while queue and len(cost) < 3500:
        priority, _, node = heapq.heappop(queue)
        point = xy(node)
        expected = cost[node] + max(float(np.linalg.norm(point-truth))-.40,0.)
        if priority > expected + 1e-9:
            continue
        gap = float(np.linalg.norm(point-truth))
        if .27 <= gap <= .40 and clearance(point) >= MARGIN:
            end = node
            break
        for dx,dy in NEIGHBORS:
            nxt = (node[0]+dx,node[1]+dy)
            future = xy(nxt)
            if not (-.32 <= future[0] <= 1.16 and -.72 <= future[1] <= .72):
                continue
            edge = future-point
            if min(clearance(point+f*edge) for f in np.linspace(0.,1.,5)) < MARGIN:
                continue
            new_cost = cost[node] + float(np.linalg.norm(edge))
            if new_cost >= cost.get(nxt, math.inf):
                continue
            cost[nxt] = new_cost
            parent[nxt] = node
            serial += 1
            h = max(float(np.linalg.norm(future-truth))-.40,0.)
            heapq.heappush(queue,(new_cost+h,serial,nxt))
    path = []
    if end is not None:
        cursor = end
        while cursor != (0,0):
            path.append(xy(cursor))
            cursor = parent[cursor]
        path.append(start)
        path.reverse()
    verified_min = min((clearance(a+f*(b-a)) for a,b in zip(path,path[1:])
                        for f in np.linspace(0.,1.,21)),default=None)
    return {"turn_min_m": round(turn_min,4), "turn_passes": True,
            "route_found": end is not None,
            "verified_21_samples": verified_min is not None and verified_min >= MARGIN,
            "verified_route_min_m": None if verified_min is None else round(verified_min,4),
            "expanded_cells": len(cost), "sampled_model_positions": len(cache),
            "search_wall_time_s": round(time.perf_counter()-start_time,3),
            "path_steps": max(len(path)-1,0),
            "path_length_m": round(sum(float(np.linalg.norm(b-a))
                for a,b in zip(path,path[1:])),4),
            "path_xy_m": [p.tolist() for p in path]}


def main() -> None:
    OUT.mkdir(parents=True,exist_ok=True)
    room = Room()
    rows = []
    try:
        for seed in (85,88):
            actual,truth = configure(room,seed)
            for offset in OFFSETS:
                room.actual[:] = actual
                row = search(room,actual,truth,offset)
                rows.append({"seed":seed,"yaw_offset_rad":offset,**row})
                print(seed,offset,"turn",row["turn_min_m"],
                      "route",row["route_found"],
                      "verified",row["verified_21_samples"],
                      "cells",row["expanded_cells"],flush=True)
    finally:
        room.close()
    result = {"scope":"Evaluator-only 4cm grid full-scene 6.5cm selected-body fixed-yaw search after safe in-place turn; 5-sample optimistic edges, 21-sample path verification. No actor authority or native skill.",
              "source_sha256":sha(SOURCE),"run_source_sha256":sha(ROOT / "experiments/ape_high_level/closed_loop/run.py"),"rows":rows}
    (OUT / "alternate-yaw-route-audit.json").write_text(json.dumps(result,indent=2)+"\n")
    (OUT / "audit_alternate_yaw_route_v125_source.py").write_bytes(SOURCE.read_bytes())


if __name__ == "__main__":
    main()
