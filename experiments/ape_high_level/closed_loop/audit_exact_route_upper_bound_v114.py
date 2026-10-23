"""V114: evaluator-only selected-body A* route from recovered-view starts."""

from __future__ import annotations

import heapq
import json
import math
from pathlib import Path
import time

import numpy as np

from audit_fresh_recovery_approach_v110 import accepted_source
from run import ROOT, Room, sha

SOURCE=Path(__file__)
OUT=ROOT / "artifacts/ape-closed-loop-cove50-v114"
GRID_M=.04
MODEL_MARGIN_M=.05
MAX_CELLS=3500
NEIGHBORS=((1,0),(-1,0),(0,1),(0,-1),
           (1,1),(1,-1),(-1,1),(-1,-1))


def search(seed: int,room: Room) -> dict:
    _,actual,paths,_,source_audit=accepted_source(seed)
    with np.load(paths[-1]) as z:
        qpos=z["actor_qpos"].copy()
    room.reset()
    room.layout(seed,absent=False,furnished=True)
    room.model.body_pos[room.model.body("loop_table").id]=(
        .40,(1 if seed%2 else -1)*.46,0.)
    room.set_table_height(.37)
    room.data.qpos[:]=qpos
    room.actual[:]=actual
    room.head=float(qpos[room.head_addr])
    room._forward()
    truth=np.asarray(room.positions["TomatoSauce"][:2],dtype=float)
    start_xy=actual[:2].copy()
    cache={}
    def clearance(xy: np.ndarray) -> float:
        key=tuple(np.round(xy,5))
        if key not in cache:
            room.actual[:2]=xy
            room._forward()
            cache[key]=float(room._current_obstacle_clearance()[0])
        return cache[key]
    def xy(node: tuple[int,int]) -> np.ndarray:
        return start_xy+GRID_M*np.asarray(node,dtype=float)
    start_clear=clearance(start_xy)
    queue=[(0.,0,(0,0))]
    cost={(0,0):0.}
    parent={}
    serial=0
    end=None
    begun=time.perf_counter()
    while queue and len(cost)<MAX_CELLS:
        priority,_,node=heapq.heappop(queue)
        point=xy(node)
        expected=cost[node]+max(float(np.linalg.norm(point-truth))-.40,0.)
        if priority>expected+1e-9:
            continue
        gap=float(np.linalg.norm(point-truth))
        if .27<=gap<=.40 and clearance(point)>=MODEL_MARGIN_M:
            end=node
            break
        for dx,dy in NEIGHBORS:
            nxt=(node[0]+dx,node[1]+dy)
            future=xy(nxt)
            if not (-.32<=future[0]<=1.16 and -.72<=future[1]<=.72):
                continue
            step=future-point
            if min(clearance(point+fraction*step)
                   for fraction in np.linspace(0.,1.,21))<MODEL_MARGIN_M:
                continue
            new_cost=cost[node]+float(np.linalg.norm(step))
            if new_cost>=cost.get(nxt,math.inf):
                continue
            cost[nxt]=new_cost
            parent[nxt]=node
            serial+=1
            heuristic=max(float(np.linalg.norm(future-truth))-.40,0.)
            heapq.heappush(queue,(new_cost+heuristic,serial,nxt))
    elapsed=time.perf_counter()-begun
    path=[]
    if end is not None:
        node=end
        while node!=(0,0):
            path.append(node)
            node=parent[node]
        path.append((0,0))
        path.reverse()
    path_xy=[xy(node) for node in path]
    path_samples=[]
    for a,b in zip(path_xy,path_xy[1:]):
        path_samples.extend(clearance(a+fraction*(b-a))
                            for fraction in np.linspace(0.,1.,21))
    path_min=min(path_samples) if path_samples else None
    return {"seed":seed,"accepted_source_audit_sha256":sha(source_audit),
            "selected_actor_qpos_source_sha256":sha(paths[-1]),
            "evaluator_start_actual_xyyaw":actual.tolist(),
            "evaluator_true_target_xy_m":truth.tolist(),
            "fixed_body_yaw_rad":float(actual[2]),
            "start_model_clearance_m":round(start_clear,4),
            "route_found":end is not None,
            "expanded_cells":len(cost),
            "sampled_model_positions":len(cache),
            "search_wall_time_s":round(elapsed,3),
            "path_steps":max(len(path)-1,0),
            "path_length_m":round(sum(float(np.linalg.norm(b-a))
                for a,b in zip(path_xy,path_xy[1:])),4),
            "end_target_gap_m":None if end is None else round(float(
                np.linalg.norm(path_xy[-1]-truth)),4),
            "path_min_model_clearance_m":None if path_min is None else round(
                path_min,4),
            "path_xy_m":[p.tolist() for p in path_xy]}


def main() -> None:
    OUT.mkdir(parents=True,exist_ok=True)
    room=Room()
    try:
        rows=[search(seed,room) for seed in (85,88)]
    finally:
        room.close()
    gate=all(row["route_found"] and
             row["path_min_model_clearance_m"]>=MODEL_MARGIN_M
             for row in rows)
    result={"scope":"Evaluator-only exact selected-body fixed-current-yaw A* from V109 seed85/V105 seed88 actual starts to simulator target 27-40cm ring. 4cm graph, 21 samples per edge,5cm model clearance; no truth route or target enters actor motion. Kinematic fixed joints/yaw upper bound, not native gait proof.",
            "source_sha256":sha(SOURCE),
            "selected_design_sha256":sha(ROOT / "robot/ape-current-design.json"),
            "rows":rows,"two_scene_exact_route_exists":gate}
    (OUT / "exact-route-upper-bound-audit.json").write_text(
        json.dumps(result,indent=2)+"\n")
    (OUT / "audit_exact_route_upper_bound_v114_source.py").write_bytes(
        SOURCE.read_bytes())
    for row in rows:
        print(row["seed"],"route",row["route_found"],"steps",
              row["path_steps"],"min",row["path_min_model_clearance_m"],
              "seconds",row["search_wall_time_s"],flush=True)
    print("two-scene exact route upper bound:",gate,flush=True)


if __name__=="__main__":
    main()
