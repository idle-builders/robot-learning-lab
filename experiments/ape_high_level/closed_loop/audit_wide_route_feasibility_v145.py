"""V145: evaluator-only route width screen from seed85 start and alarm stop."""

from __future__ import annotations

import heapq
import json
import math
from pathlib import Path

import numpy as np

from build_scene_pairs_v133 import body_geoms,full_body_clearance,scene_geoms
from run import ROOT,sha
from run_moved_props_v134 import MovedPropRoom

SOURCE=Path(__file__)
OUT=ROOT / "artifacts/ape-closed-loop-cove50-v145"
GRID=.04
NEIGHBORS=((1,0),(-1,0),(0,1),(0,-1),(1,1),(1,-1),(-1,1),(-1,-1))


def state(name:str) -> tuple[np.ndarray,np.ndarray]:
    if name=="start":
        path=ROOT / "artifacts/ape-closed-loop-cove50-v134/seed85-passable-1/seed85-repair/trace.json"
        trace=json.loads(path.read_text())
        qpath=ROOT / "artifacts/ape-closed-loop-cove50-v139/seed85-passable-1/fresh-look-actor-input.npz"
    else:
        path=ROOT / "artifacts/ape-closed-loop-cove50-v143/seed85/trace.json"
        trace=json.loads(path.read_text())
        qpath=ROOT / "artifacts/ape-closed-loop-cove50-v143/seed85/frame-6.npz"
    with np.load(qpath) as z:qpos=z["actor_qpos"].copy()
    return np.asarray(trace["evaluator_final_actual_xyyaw"]),qpos


def search(room:MovedPropRoom,start:np.ndarray,target:np.ndarray,margin:float) -> dict:
    initial=start[:2].copy()
    cache={}
    def clearance(xy:np.ndarray) -> float:
        key=tuple(np.round(xy,5))
        if key not in cache:
            room.actual[:]=start
            room.actual[:2]=xy
            room._forward()
            cache[key]=float(room._current_obstacle_clearance()[0])
        return cache[key]
    def xy(cell):return initial+GRID*np.asarray(cell,dtype=float)
    start_clear=clearance(initial)
    if start_clear<margin:
        return {"margin_m":margin,"route_found":False,
                "reason":"starting_body_below_margin",
                "start_clearance_m":round(start_clear,4)}
    queue=[(0.,0,(0,0))]
    costs={(0,0):0.}
    parents={}
    serial=0
    goal=None
    while queue and len(costs)<3500:
        priority,_,node=heapq.heappop(queue)
        point=xy(node)
        expected=costs[node]+max(float(np.linalg.norm(point-target))-.40,0.)
        if priority>expected+1e-8:continue
        gap=float(np.linalg.norm(point-target))
        if .27<=gap<=.40 and clearance(point)>=margin:
            goal=node;break
        for dx,dy in NEIGHBORS:
            nxt=(node[0]+dx,node[1]+dy)
            future=xy(nxt)
            if not (-.32<=future[0]<=1.16 and -.72<=future[1]<=.72):continue
            step=future-point
            if min(clearance(point+fraction*step) for fraction in np.linspace(0.,1.,21))<margin:
                continue
            new=costs[node]+float(np.linalg.norm(step))
            if new>=costs.get(nxt,math.inf):continue
            costs[nxt]=new;parents[nxt]=node;serial+=1
            heuristic=max(float(np.linalg.norm(future-target))-.40,0.)
            heapq.heappush(queue,(new+heuristic,serial,nxt))
    path=[]
    if goal is not None:
        node=goal
        while node!=(0,0):path.append(node);node=parents[node]
        path.append((0,0));path.reverse()
    points=[xy(node) for node in path]
    model_min=None
    wall_min=None
    if points:
        obstacles=scene_geoms(room)
        robots=body_geoms(room,obstacles)
        model_min=math.inf;wall_min=math.inf
        for a,b in zip(points,points[1:]):
            for fraction in np.linspace(0.,1.,21):
                room.actual[:]=start
                room.actual[:2]=a+fraction*(b-a)
                room._forward()
                model_min=min(model_min,room._current_obstacle_clearance()[0])
                wall_min=min(wall_min,full_body_clearance(room,obstacles,robots)[0])
    return {"margin_m":margin,"route_found":goal is not None,
            "reason":None if goal is not None else "no_fixed_yaw_route",
            "start_clearance_m":round(start_clear,4),
            "expanded_cells":len(costs),"sampled_positions":len(cache),
            "path_steps":max(0,len(path)-1),
            "path_length_m":round(sum(float(np.linalg.norm(b-a))
                  for a,b in zip(points,points[1:])),4),
            "end_target_gap_m":None if goal is None else round(float(np.linalg.norm(points[-1]-target)),4),
            "path_min_all_object_m":None if model_min is None else round(model_min,4),
            "path_min_including_walls_m":None if wall_min is None else round(wall_min,4),
            "path_xy_m":[p.tolist() for p in points]}


def main() -> None:
    OUT.mkdir(parents=True,exist_ok=True)
    room=MovedPropRoom({"CreamCheese":(.4,-.72),"Parmesan":(.4,.6)})
    rows=[]
    try:
        room.reset();room.layout(85,absent=False,furnished=True)
        room.model.body_pos[room.model.body("loop_table").id]=(.4,.46,0.)
        room.set_table_height(.37)
        target=np.asarray(room.positions["TomatoSauce"][:2])
        for name,margins in (("start",(.065,.075,.09)),("alarm_stop",(.065,))):
            actual,qpos=state(name)
            room.data.qpos[:]=qpos
            room.actual[:]=actual
            room._forward()
            for margin in margins:
                row={"state":name,"actual_xyyaw":actual.tolist(),
                     **search(room,actual,target,margin)}
                rows.append(row)
                print(name,margin,"found",row["route_found"],
                      "steps",row.get("path_steps"),"min",row.get("path_min_including_walls_m"),flush=True)
    finally:
        room.close()
    result={"scope":"V145 evaluator-only 4cm fixed-yaw/fixed-joint route width audit in seed85 variant1 from actor start and alarm stop; all loop props/furniture in search, room walls rechecked on selected witness; no truth route enters actor.",
            "source_sha256":sha(SOURCE),"rows":rows}
    (OUT / "wide-route-feasibility.json").write_text(json.dumps(result,indent=2)+"\n")
    (OUT / SOURCE.name).write_bytes(SOURCE.read_bytes())


if __name__=="__main__":main()
