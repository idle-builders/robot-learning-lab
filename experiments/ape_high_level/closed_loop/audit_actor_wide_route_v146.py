"""V146: actor-only 9cm graph target routes, evaluator full-path scoring afterward."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

import audit_moved_goal_reselection_v135 as screen
from build_scene_pairs_v133 import body_geoms,full_body_clearance,scene_geoms
from run import ROOT,Vision,sha
from run_moved_props_v134 import MovedPropRoom

SOURCE=Path(__file__)
OUT=ROOT / "artifacts/ape-closed-loop-cove50-v146"
MARGIN=.09
CASES=((85,"seed85-passable-1"),(88,"seed88-passable-1"))


def score_path(seed:int,variant:dict,row:dict) -> dict|None:
    selected=row.get("selected")
    if selected is None or selected["kind"]!="annulus":return None
    path=selected["path_cells"]
    branch=OUT / variant["variant"]
    with np.load(branch / "fresh-look-actor-input.npz") as z:qpos=z["actor_qpos"].copy()
    trace=json.loads((screen.V134 / variant["variant"] / f"seed{seed}-repair/trace.json").read_text())
    actual=np.asarray(trace["evaluator_final_actual_xyyaw"])
    room=MovedPropRoom({"CreamCheese":tuple(variant["cream_cheese_xy_m"]),
                        "Parmesan":tuple(variant["parmesan_xy_m"])})
    try:
        room.reset();room.layout(seed,absent=False,furnished=True)
        room.model.body_pos[room.model.body("loop_table").id]=(
            .40,(1 if seed%2 else -1)*.46,0.)
        room.set_table_height(.37)
        room.data.qpos[:]=qpos
        room.actual[:]=actual
        room._forward()
        obstacles=scene_geoms(room);robots=body_geoms(room,obstacles)
        points=[actual[:2].copy()]+[screen.GRID*np.asarray(cell)
                                     for cell in path[1:]]
        edge_rows=[]
        for index,(a,b) in enumerate(zip(points,points[1:]),1):
            closest=(float("inf"),"","")
            for fraction in np.linspace(0.,1.,21):
                room.actual[:]=actual
                room.actual[:2]=a+fraction*(b-a)
                room._forward()
                gap=full_body_clearance(room,obstacles,robots)
                if gap[0]<closest[0]:closest=gap
            edge_rows.append({"edge_index":index,
                              "min_full_scene_m":round(closest[0],4),
                              "limiting_geoms":list(closest[1:])})
        return {"actor_path_cells":path,
                "path_steps":len(edge_rows),
                "path_min_full_scene_m":min(e["min_full_scene_m"] for e in edge_rows),
                "edge_rows":edge_rows}
    finally:
        room.close()


def main() -> None:
    OUT.mkdir(parents=True,exist_ok=True)
    scenes=json.loads(screen.SCENES.read_text())
    exemplar=np.load(screen.source.V81 / "owner-taught-exemplar.npy")
    vision=Vision("generic")
    old_out=screen.OUT
    screen.OUT=OUT
    try:
        rows=[]
        for seed,name in CASES:
            variant=next(v for r in scenes["rows"] if r["seed"]==seed
                         for v in r["passable_variants"] if v["variant"]==name)
            row=screen.screen(variant,vision,exemplar,seed=seed,
                              annulus_margin_m=MARGIN)
            row["full_scene_path_score"]=score_path(seed,variant,row)
            rows.append(row)
    finally:
        screen.OUT=old_out
    gate=all(row["fresh_correct_target"] and
             not row["matched_absent_accepted"] and
             not row["look_range_alarm"] and
             row.get("annulus",{}).get("found") and
             row["full_scene_path_score"] is not None and
             row["full_scene_path_score"]["path_min_full_scene_m"]>=.05
             for row in rows)
    result={"scope":"V146 read-only actor full-graph 9cm selected-body point margin target route screen in both validated development variants; exact full-body all-object+wall edge scoring only afterward. No motion.",
            "source_sha256":sha(SOURCE),
            "screen_runner_source_sha256":sha(screen.SOURCE),
            "scene_audit_sha256":sha(screen.SCENES),
            "actor_point_margin_m":MARGIN,
            "rows":rows,"paired_actor_wide_route_screen_pass":gate}
    (OUT / "actor-wide-route-audit.json").write_text(json.dumps(result,indent=2)+"\n")
    (OUT / SOURCE.name).write_bytes(SOURCE.read_bytes())
    (OUT / "audit_moved_goal_reselection_v135_imported_source.py").write_bytes(screen.SOURCE.read_bytes())
    for row in rows:
        print(row["variant"],"annulus",row.get("annulus",{}).get("found"),
              "path min",None if row["full_scene_path_score"] is None else
               row["full_scene_path_score"]["path_min_full_scene_m"],flush=True)
    print("V146 paired screen",gate,flush=True)


if __name__=="__main__":main()
