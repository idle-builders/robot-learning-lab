"""V133: geometry-validated furnished scene pairs for the closed-loop task."""

from __future__ import annotations

import itertools
import json
from pathlib import Path

import mujoco
import numpy as np

from audit_fresh_recovery_approach_v110 import accepted_source
from run import ROOT,Room,sha

SOURCE=Path(__file__)
OUT=ROOT / "artifacts/ape-closed-loop-cove50-v133"
OLD=ROOT / "artifacts/ape-closed-loop-cove50-v115/exact-margin-feasibility-audit.json"
GRID_X=(.40,.52,.64,.76,.88,1.00)
GRID_Y=(-.72,-.60,.60,.72)
PROP_CLEAR_M=.03
BODY_CLEAR_M=.065


def scene_geoms(room: Room) -> list[int]:
    return [gid for gid in range(room.model.ngeom)
            if ((room.model.geom(gid).name.startswith("loop_") and
                 room.model.geom(gid).name!="loop_rug_visual") or
                room.model.geom(gid).name.startswith("room_") and
                room.model.geom(gid).name.endswith("_wall"))]


def body_geoms(room: Room,obstacles: list[int]) -> list[int]:
    return [gid for gid in range(room.model.ngeom)
            if room.model.geom_bodyid[gid]!=0 and
               room.model.geom_contype[gid]>0 and gid not in obstacles]


def distance(room: Room,a: int,b: int) -> float:
    return float(mujoco.mj_geomDistance(room.model,room.data,a,b,10.,None))


def full_body_clearance(room: Room,obstacles: list[int],robots: list[int]) -> tuple[float,str,str]:
    value,robot,scene=min((distance(room,r,s),r,s)
                          for r in robots for s in obstacles)
    return value,room.model.geom(robot).name,room.model.geom(scene).name


def configure(room: Room,seed: int) -> tuple[np.ndarray,list[np.ndarray]]:
    _,actual,paths,_,_=accepted_source(seed)
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
    old=json.loads(OLD.read_text())
    row=next(row for row in old["rows"] if row["seed"]==seed)
    route=[np.asarray(p) for p in row["path_xy_m"]]
    return actual,route


def set_props(room: Room,cream: tuple[float,float],
              parmesan: tuple[float,float]) -> None:
    for name,xy in (("CreamCheese",cream),("Parmesan",parmesan)):
        room.positions[name]=xy
        room.model.body_pos[room.model.body(f"loop_{name}").id]=(*xy,.052)
    room._forward()


def prop_clearance(room: Room,obstacles: list[int]) -> tuple[float,str,str]:
    moved=[room.model.geom(f"loop_{name}_geom").id
           for name in ("CreamCheese","Parmesan")]
    result=[]
    for a in moved:
        for b in obstacles:
            if a==b or (a==moved[1] and b==moved[0]):
                continue
            result.append((distance(room,a,b),
                           room.model.geom(a).name,room.model.geom(b).name))
    return min(result)


def path_clearance(room: Room,route: list[np.ndarray],obstacles: list[int],
                   robots: list[int],samples: int) -> tuple[float,str,str]:
    closest=(float("inf"),"","")
    for a,b in zip(route,route[1:]):
        for fraction in np.linspace(0.,1.,samples):
            room.actual[:2]=a+fraction*(b-a)
            room._forward()
            gap=full_body_clearance(room,obstacles,robots)
            if gap[0]<closest[0]:closest=gap
            if closest[0]<BODY_CLEAR_M:return closest
    return closest


def main() -> None:
    OUT.mkdir(parents=True,exist_ok=True)
    room=Room()
    rows=[]
    try:
        obstacles=scene_geoms(room)
        robots=body_geoms(room,obstacles)
        candidates=list(itertools.product(GRID_X,GRID_Y))
        # Interleave distant positions before dense neighbor pairs, deterministically.
        pairs=[(cream,parmesan) for cream in candidates for parmesan in candidates
               if np.linalg.norm(np.asarray(cream)-parmesan)>=.22]
        for seed in (85,88):
            start,route=configure(room,seed)
            target=np.asarray(room.positions["TomatoSauce"])
            initial={name:list(room.positions[name])
                     for name in ("CreamCheese","Parmesan","TomatoSauce")}
            baseline=path_clearance(room,route,obstacles,robots,21)
            selected=[]
            inspected=0
            for cream,parmesan in pairs:
                if min(np.linalg.norm(np.asarray(cream)-target),
                       np.linalg.norm(np.asarray(parmesan)-target))>.55:
                    continue
                room.actual[:]=start
                set_props(room,cream,parmesan)
                scene_gap=prop_clearance(room,obstacles)
                inspected+=1
                if scene_gap[0]<PROP_CLEAR_M:
                    continue
                optimistic=path_clearance(room,route,obstacles,robots,5)
                if optimistic[0]<BODY_CLEAR_M:
                    continue
                exact=path_clearance(room,route,obstacles,robots,21)
                if exact[0]<BODY_CLEAR_M:
                    continue
                if any(
                    np.linalg.norm(np.asarray(cream)-
                                   np.asarray(prior["cream_cheese_xy_m"]))<.20 or
                    np.linalg.norm(np.asarray(parmesan)-
                                   np.asarray(prior["parmesan_xy_m"]))<.20 or
                    min(
                        np.linalg.norm(np.asarray(cream)-
                                       np.asarray(prior["cream_cheese_xy_m"]))+
                        np.linalg.norm(np.asarray(parmesan)-
                                       np.asarray(prior["parmesan_xy_m"])),
                        np.linalg.norm(np.asarray(cream)-
                                       np.asarray(prior["parmesan_xy_m"]))+
                        np.linalg.norm(np.asarray(parmesan)-
                                       np.asarray(prior["cream_cheese_xy_m"])))<.30
                    for prior in selected):
                    continue
                room.actual[:]=start
                room._forward()
                index=len(selected)+1
                image=OUT / f"seed{seed}-passable-{index}.png"
                room.capture_external().save(image)
                selected.append({"variant":f"seed{seed}-passable-{index}",
                                 "cream_cheese_xy_m":list(cream),
                                 "parmesan_xy_m":list(parmesan),
                                 "moved_prop_scene_gap_m":round(scene_gap[0],4),
                                 "moved_prop_limit_geoms":list(scene_gap[1:]),
                                 "witness_body_clearance_m":round(exact[0],4),
                                 "witness_limit_geoms":list(exact[1:]),
                                 "external_image_sha256":sha(image),
                                 "external_image_path":str(image)})
                print(seed,index,cream,parmesan,"scene",round(scene_gap[0],4),
                      "body",round(exact[0],4),flush=True)
                if len(selected)>=2:
                    break
            rows.append({"seed":seed,"baseline_props_xy_m":initial,
                         "baseline_old_route_full_body_min_m":round(baseline[0],4),
                         "baseline_limit_geoms":list(baseline[1:]),
                         "witness_route_xy_m":[p.tolist() for p in route],
                         "inspected_candidate_pairs":inspected,
                         "passable_variants":selected})
    finally:
        room.close()
    result={"scope":"Evaluator-only paired furnished scene generation for saved Cove-50 starts and old V115 XY witness routes. Floor props retained and placed from fixed candidate grid; 3cm moved-prop separation from all solid objects/walls; full selected-body 6.5cm clearance with 21 samples per 4cm edge including props, furniture and room walls. No truth passed to actor and no contact physics.",
            "source_sha256":sha(SOURCE),"selected_design_sha256":sha(ROOT / "robot/ape-current-design.json"),
            "legacy_route_audit_sha256":sha(OLD),"rows":rows,
            "two_passable_variants_each":all(len(row["passable_variants"])==2 for row in rows)}
    (OUT / "paired-scene-validity.json").write_text(json.dumps(result,indent=2)+"\n")
    (OUT / "build_scene_pairs_v133_source.py").write_bytes(SOURCE.read_bytes())
    print("paired validity",result["two_passable_variants_each"],flush=True)


if __name__=="__main__":
    main()
