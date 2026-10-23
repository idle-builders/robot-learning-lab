"""V152: actor 9cm target/info route from fresh owner-taught origin sweeps."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from audit_moved_goal_reselection_v135 import course_goal
from build_scene_pairs_v133 import body_geoms,full_body_clearance,scene_geoms
from depth_voxel import DepthVoxelMap
from goal_map_route import GoalMapRoute
from robot_geometry import RobotGeometry
from run import H,ROOT,W,sensed_range,sha
from run_moved_props_v134 import MovedPropRoom
from self_depth_mask import SelfDepthMask
from two_height_clearance import TwoHeightClearance
from upper_obstacle_side import choose_side

SOURCE=Path(__file__)
OUT=ROOT / "artifacts/ape-closed-loop-cove50-v152"
V151=ROOT / "artifacts/ape-closed-loop-cove50-v151/color-grounded-entry-audit.json"
SCENES=ROOT / "artifacts/ape-closed-loop-cove50-v133/paired-scene-validity.json"
LIGHT=1.35
MARGIN=.09


def selected_path(planner:GoalMapRoute,pose:np.ndarray) -> list[list[int]]:
    if planner.goal is None:return []
    origin=planner.cell(pose[:2])
    cursor=planner.goal
    path=[]
    while cursor!=origin:
        path.append(list(cursor))
        if cursor not in planner.last_parent:return []
        cursor=planner.last_parent[cursor]
    path.append(list(origin));path.reverse()
    return path


def score_path(room:MovedPropRoom,actual:np.ndarray,
               cells:list[list[int]]) -> dict|None:
    if len(cells)<2:return None
    obstacles=scene_geoms(room);robots=body_geoms(room,obstacles)
    points=[actual[:2].copy()]+[.04*np.asarray(cell,dtype=float) for cell in cells[1:]]
    edges=[]
    for index,(a,b) in enumerate(zip(points,points[1:]),1):
        closest=(float("inf"),"","")
        for fraction in np.linspace(0.,1.,21):
            room.actual[:]=actual
            room.actual[:2]=a+fraction*(b-a)
            room._forward()
            gap=full_body_clearance(room,obstacles,robots)
            if gap[0]<closest[0]:closest=gap
        edges.append({"edge_index":index,"min_full_scene_m":round(closest[0],4),
                      "limiting_geoms":list(closest[1:])})
    return {"path_steps":len(edges),
            "min_full_scene_m":min(row["min_full_scene_m"] for row in edges),
            "edges":edges}


def screen(seed:int,variant:dict,entry:dict,*,
           frame_root:Path|None=None,light_multiplier:float=LIGHT) -> dict:
    accepted=[r for r in entry["rows"] if r["accepted_box"] is not None]
    points=np.asarray([r["accepted_world_xyz_m"][:2] for r in accepted],dtype=float)
    if len(points)<2:
        return {"seed":seed,"variant":variant["variant"],
                "status":"insufficient_fresh_identity_views"}
    target=np.median(points,axis=0)
    spread=max(float(np.linalg.norm(p-target)) for p in points)
    if spread>.08:
        return {"seed":seed,"variant":variant["variant"],
                "status":"target_views_inconsistent","spread_m":spread}
    positions={"CreamCheese":tuple(variant["cream_cheese_xy_m"]),
               "Parmesan":tuple(variant["parmesan_xy_m"])}
    room=MovedPropRoom(positions)
    mask=SelfDepthMask(room.model,camera="lower_rgb",width=W,height=H)
    root=frame_root or ROOT / "artifacts/ape-closed-loop-cove50-v151"
    branch=root / f"seed{seed}-light{light_multiplier:.2f}-present"
    try:
        room.reset();room.layout(seed,absent=False,furnished=True)
        room.model.body_pos[room.model.body("loop_table").id]=(
            .40,(1 if seed%2 else -1)*.46,0.)
        room.set_table_height(.37)
        room.model.light_diffuse[0]*=light_multiplier
        room._forward()
        fovy=float(room.model.cam_fovy[room.model.camera("lower_rgb").id])
        voxel=DepthVoxelMap()
        final=None
        hashes=[]
        for row in entry["rows"]:
            path=branch / f"lower-angle-{row['yaw_command_rad']:+.1f}.npz"
            with np.load(path) as z:
                qpos=z["actor_qpos"].copy()
                pose=z["estimated_pose_xyyaw"].copy()
                stamp=float(z["capture_time_s"])
                filtered,_=mask.apply(z["depth_m"],qpos)
                voxel.observe(filtered,(z["camera_origin_m"],
                                        z["camera_rotation"]),stamp,
                              fovy_deg=fovy,self_masked=True)
            final=(qpos,pose,stamp)
            hashes.append(sha(path))
        qpos,pose,stamp=final
        room.data.qpos[:]=qpos
        room.actual[:]=pose
        room.estimated[:]=pose
        room.head=float(qpos[room.head_addr]);room.time_s=stamp
        room._forward()
        scans={height:sensed_range(room,seed,1520000+int(height*100),
                                   height_m=height)
               for height in (.04,.08,.11)}
        robot=RobotGeometry(room.model)
        point=TwoHeightClearance(robot,qpos,pose,scans[.11],scans[.04],
                                 voxel,stamp,range_8=scans[.08])
        planner=GoalMapRoute()
        candidate=planner.plan(scans[.11],pose,target,None,stamp,
                               searching=False,edge_clearance=point.edge_gap,
                               edge_clearance_margin_m=MARGIN,
                               point_all_graph_edges=True)
        side=choose_side(voxel,pose,target,stamp)
        kind="annulus" if candidate is not None else None
        path=selected_path(planner,pose) if candidate is not None else []
        information=None
        if not path and side["side"] is not None:
            information=course_goal(pose,target,side["side"],scans,point,set(),
                                    margin_m=MARGIN)
            if information["goal_cell"] is not None:
                kind="information"
                path=information["path_cells"]
        score=score_path(room,pose.copy(),path)
        return {"seed":seed,"variant":variant["variant"],
                "status":"goal_selected" if score is not None else "no_actor_goal",
                "actor_target_xy_m":target.tolist(),
                "actor_target_view_spread_m":round(spread,4),
                "accepted_view_count":len(points),
                "actor_history_sha256":hashes,
                "actor_pose_xyyaw":pose.tolist(),
                "side_evidence":side,"annulus_found":candidate is not None,
                "annulus_debug":planner.last_debug,
                "information_goal":information,
                "selected_kind":kind,"selected_path_cells":path,
                "evaluator_selected_path":score}
    finally:
        mask.close();room.close()


def main() -> None:
    OUT.mkdir(parents=True,exist_ok=True)
    entries=json.loads(V151.read_text())["rows"]
    scenes=json.loads(SCENES.read_text())
    rows=[]
    for seed in (85,88):
        variant=next(v for row in scenes["rows"] if row["seed"]==seed
                     for v in row["passable_variants"] if v["variant"].endswith("-2"))
        entry=next(r for r in entries if r["seed"]==seed and
                   not r["absent"] and r["light_multiplier"]==LIGHT)
        rows.append(screen(seed,variant,entry))
    gate=all(row["status"]=="goal_selected" and
             row["evaluator_selected_path"]["min_full_scene_m"]>=.05
             for row in rows)
    result={"scope":"V152 read-only 9cm actor annulus or high-depth-side information route from fresh V151 owner-crop-gated origin sweeps in seed85/88 variant2 bright rooms, evaluator full-scene selected path afterward only.",
            "source_sha256":sha(SOURCE),"v151_audit_sha256":sha(V151),
            "scene_audit_sha256":sha(SCENES),"actor_point_margin_m":MARGIN,
            "rows":rows,"paired_origin_route_screen_pass":gate}
    (OUT / "origin-route-audit.json").write_text(json.dumps(result,indent=2)+"\n")
    (OUT / SOURCE.name).write_bytes(SOURCE.read_bytes())
    for row in rows:
        print(row["seed"],row["status"],row.get("selected_kind"),
              "min",None if row.get("evaluator_selected_path") is None else
              row["evaluator_selected_path"]["min_full_scene_m"],
              "annulus",row.get("annulus_found"),
              "debug",row.get("annulus_debug"),flush=True)
    print("V152 paired screen",gate,flush=True)


if __name__=="__main__":main()
