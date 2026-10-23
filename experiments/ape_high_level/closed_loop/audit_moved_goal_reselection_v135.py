"""V135: actor-only new target/information goal at moved-prop seed88 stops."""

from __future__ import annotations

import heapq
import json
import math
from pathlib import Path

import numpy as np

import run_live_committed_course_v121 as source
from audit_side_look_target_v81 import absent_capture,infer
from depth_voxel import DepthVoxelMap
from fast_range_guard import FastRangeGuard
from footprint_route import observed_free
from goal_map_route import GoalMapRoute
from robot_geometry import RobotGeometry
from run import H,ROOT,W,Vision,iou,sensed_range,sha
from run_moved_props_v134 import MovedPropRoom
from self_depth_mask import SelfDepthMask
from two_height_clearance import TwoHeightClearance

SOURCE=Path(__file__)
OUT=ROOT / "artifacts/ape-closed-loop-cove50-v135"
SCENES=ROOT / "artifacts/ape-closed-loop-cove50-v133/paired-scene-validity.json"
V134=ROOT / "artifacts/ape-closed-loop-cove50-v134"
V90=ROOT / "artifacts/ape-closed-loop-cove50-v90"
GRID=.04
MARGIN=.065
NEIGHBORS=((1,0),(-1,0),(0,1),(0,-1),(1,1),(1,-1),(-1,1),(-1,-1))


def course_goal(pose: np.ndarray,target: np.ndarray,side_name: str,
                scans: dict[float,list[float | None]],point: TwoHeightClearance,
                visited: set[tuple[int,int]],*,margin_m: float=MARGIN) -> dict:
    forward=target-pose[:2]
    forward/=np.linalg.norm(forward)
    left=np.array([-forward[1],forward[0]])
    side=left if side_name=="left" else -left
    course=forward+side
    course/=np.linalg.norm(course)
    origin=tuple(int(round(value/GRID)) for value in pose[:2])
    queue=[(0.,0,origin)]
    costs={origin:0.}
    travel={origin:0.}
    parent={}
    gaps={}
    serial=0
    goal=None
    while queue and len(costs)<500:
        cost,_,node=heapq.heappop(queue)
        if cost>costs[node]+1e-9 or travel[node]>.8:
            continue
        xy=GRID*np.asarray(node)
        if node!=origin and float((xy-pose[:2])@course)>=.18:
            goal=node
            break
        start=pose[:2] if node==origin else xy
        for dx,dy in NEIGHBORS:
            nxt=(node[0]+dx,node[1]+dy)
            end=GRID*np.asarray(nxt)
            if np.linalg.norm(end-pose[:2])>.35:
                continue
            if not (observed_free(end,pose,scans[.11]) and
                    observed_free(end,pose,scans[.04])):
                continue
            gap=float(point.edge_gap(start,end,pose[2]))
            if gap<margin_m:
                continue
            step=float(np.linalg.norm(end-start))
            new_travel=travel[node]+step
            if new_travel>.8:
                continue
            new_cost=cost+step+.010/max(gap,.03)+(.04 if nxt in visited else 0.)
            if new_cost>=costs.get(nxt,math.inf):
                continue
            costs[nxt]=new_cost
            travel[nxt]=new_travel
            parent[nxt]=node
            gaps[nxt]=gap
            serial+=1
            heapq.heappush(queue,(new_cost,serial,nxt))
    path=[]
    if goal is not None:
        cursor=goal
        while cursor!=origin:
            path.append(cursor)
            cursor=parent[cursor]
        path.append(origin)
        path.reverse()
    return {"goal_cell":None if goal is None else list(goal),
            "path_cells":[list(cell) for cell in path],
            "edge_point_gaps_m":[round(gaps[cell],4) for cell in path[1:]],
            "expanded_cells":len(costs),
            "max_course_progress_m":round(max(float((GRID*np.asarray(cell)-pose[:2])@course)
                                                for cell in costs),4),
            "course_xy":course.tolist()}


def exact_edge(room: MovedPropRoom,actual: np.ndarray,
               pose: np.ndarray,delta: np.ndarray) -> tuple[float,str,str]:
    rows=[]
    for fraction in np.linspace(0.,1.,21):
        room.actual[:]=actual
        room.actual[:2]+=.88*fraction*delta
        room._forward()
        rows.append(room._current_obstacle_clearance())
    return min(rows,key=lambda row:row[0])


def screen(variant: dict,vision: Vision,exemplar: np.ndarray,
           *,seed: int=88,annulus_margin_m: float=MARGIN) -> dict:
    name=variant["variant"]
    positions={"CreamCheese":tuple(variant["cream_cheese_xy_m"]),
               "Parmesan":tuple(variant["parmesan_xy_m"])}
    room=MovedPropRoom(positions)
    absent_room=MovedPropRoom(positions)
    mask=SelfDepthMask(room.model,camera="lower_rgb",width=W,height=H)
    branch=OUT / name
    branch.mkdir(parents=True,exist_ok=True)
    try:
        trace_path=V134 / name / f"seed{seed}-repair/trace.json"
        trace=json.loads(trace_path.read_text())
        source_dir=trace_path.parent
        count=sum("actor_input_sha256" in e for e in trace["events"])
        final_frame=source_dir / f"frame-{count}.npz"
        with np.load(final_frame) as z:
            qpos=z["actor_qpos"].copy()
            pose=z["estimated_pose_xyyaw"].copy()
            time_s=float(z["capture_time_s"])
        voxel=source.load_initial(seed,room,mask)["voxel"]
        fovy=float(room.model.cam_fovy[room.model.camera("lower_rgb").id])
        for index in range(1,count+1):
            with np.load(source_dir / f"frame-{index}.npz") as z:
                filtered,_=mask.apply(z["depth_m"],z["actor_qpos"])
                voxel.observe(filtered,(z["camera_origin_m"],
                                        z["camera_rotation"]),
                              float(z["capture_time_s"]),
                              fovy_deg=fovy,self_masked=True)
        room.reset()
        room.layout(seed,absent=False,furnished=True)
        room.model.body_pos[room.model.body("loop_table").id]=(
            .40,(1 if seed%2 else -1)*.46,0.)
        room.set_table_height(.37)
        room.data.qpos[:]=qpos
        actual=np.asarray(trace["evaluator_final_actual_xyyaw"])
        room.actual[:]=actual
        room.estimated[:]=pose
        room.head=float(qpos[room.head_addr])
        room.time_s=time_s
        room._forward()
        memory=source.accepted_source(seed)[0]
        bearing=math.atan2(memory[1]-pose[1],memory[0]-pose[0])-pose[2]
        yaw=float(np.clip(math.atan2(math.sin(bearing),math.cos(bearing)),-1.2,1.2))
        guard4=FastRangeGuard(band_m=.05)
        guard11=FastRangeGuard(band_m=.05)
        serial=0
        def sense(height: float):
            nonlocal serial
            serial+=1
            return sensed_range(room,seed,135000+serial,height_m=height)
        guard4.update(sense(.04),room.estimated,room.time_s)
        guard11.update(sense(.11),room.estimated,room.time_s)
        def probe(substep: int,fraction: float):
            low=guard4.update(sense(.04),room.estimated,room.time_s)
            high=guard11.update(sense(.11),room.estimated,room.time_s)
            if low["status"]=="near_body_change" or high["status"]=="near_body_change":
                return {"low":low,"high":high,"action_fraction":fraction}
            return None
        room.look(yaw,"lower_rgb",safety_probe=probe)
        rgb,depth=room.capture("lower_rgb",depth=True)
        camera=room.camera_pose_estimate("lower_rgb")
        qpos=room.pose_data.qpos.copy()
        actor_frame=branch / "fresh-look-actor-input.npz"
        np.savez_compressed(actor_frame,rgb=np.asarray(rgb),depth_m=depth,
                            camera_origin_m=camera[0],camera_rotation=camera[1],
                            actor_qpos=qpos,estimated_pose_xyyaw=room.estimated.copy(),
                            capture_time_s=room.time_s)
        present=infer(vision,exemplar,rgb,depth,camera,memory,None,
                      branch / "fresh-look-present-annotated.png")
        truth=room.truth_box("TomatoSauce","lower_rgb")
        correct=(present["accepted_box"] is not None and truth is not None and
                 iou(present["accepted_box"],truth)>=.25)
        frame={"actor_qpos":qpos,"estimated_pose_xyyaw":room.estimated.copy(),
               "capture_time_s":room.time_s}
        absent_source=branch / "fresh-look-absent"
        absent_rgb,absent_depth,absent_camera,_=absent_capture(
            absent_room,seed,frame,room.actual.copy(),yaw,absent_source)
        absent=infer(vision,exemplar,absent_rgb,absent_depth,absent_camera,
                     memory,None,branch / "fresh-look-absent-annotated.png")
        target=None
        if correct:
            proposal=next(row for row in present["proposals"]
                          if row["box"]==present["accepted_box"])
            if proposal["depth_world_xyz_m"] is not None:
                target=np.asarray(proposal["depth_world_xyz_m"][:2])
        result={"variant":name,"v134_trace_sha256":sha(trace_path),
                "actor_frame_sha256":sha(actor_frame),
                "look_yaw_rad":yaw,
                "look_range_alarm":room.safety_stop_observation is not None,
                "fresh_correct_target":correct,
                "fresh_similarity":present["accepted_similarity"],
                "matched_absent_accepted":absent["accepted_box"] is not None,
                "fresh_actor_target_xy_m":None if target is None else target.tolist()}
        if target is None or result["look_range_alarm"] or result["matched_absent_accepted"]:
            return result
        filtered,_=mask.apply(depth,qpos)
        voxel.observe(filtered,camera,room.time_s,fovy_deg=fovy,self_masked=True)
        scans={h:sense(h) for h in (.04,.08,.11)}
        robot=RobotGeometry(room.model)
        point=TwoHeightClearance(robot,qpos,room.estimated,scans[.11],
                                 scans[.04],voxel,room.time_s,range_8=scans[.08])
        planner=GoalMapRoute()
        annulus=planner.plan(scans[.11],room.estimated,target,None,
                             room.time_s,searching=False,
                             edge_clearance=point.edge_gap,
                             edge_clearance_margin_m=annulus_margin_m,
                             point_all_graph_edges=True)
        annulus_row={"found":annulus is not None,
                     "debug":planner.last_debug,
                     "goal_cell":None if planner.goal is None else list(planner.goal)}
        if annulus is not None:
            a=annulus[3]
            annulus_row["first_point_gap_m"]=round(float(annulus[2]),4)
            annulus_row["detail"]=a
        result["annulus"] = annulus_row
        visited={tuple(e["detail"]["plan_cells"][1])
                 for e in trace["events"] if e.get("action")=="replanned_base_step"}
        side=json.loads((V90 / f"seed{seed}-actor/trace.json").read_text())["side"]
        info=course_goal(room.estimated,target,side,scans,point,visited)
        result["information_goal"]=info
        kind="annulus" if annulus is not None else "information"
        goal=(planner.goal if annulus is not None else
              None if info["goal_cell"] is None else tuple(info["goal_cell"]))
        if goal is not None:
            path=([] if annulus is not None else info["path_cells"])
            if annulus is not None:
                cursor=goal
                origin=planner.cell(room.estimated[:2])
                while cursor!=origin:
                    path.append(list(cursor))
                    cursor=planner.last_parent[cursor]
                path.append(list(origin))
                path.reverse()
            first=GRID*np.asarray(path[1],dtype=float)
            old_actual=room.actual.copy()
            old_estimated=room.estimated.copy()
            edge=exact_edge(room,old_actual,old_estimated,
                            first-old_estimated[:2])
            endpoint=GRID*np.asarray(goal,dtype=float)
            room.actual[:]=old_actual
            room.actual[:2]+=endpoint-old_estimated[:2]
            room._forward()
            endpoint_clear=room._current_obstacle_clearance()
            result["selected"]={"kind":kind,"goal_cell":list(goal),
                                "path_cells":path,
                                "first_edge_model_min_m":round(edge[0],4),
                                "first_edge_limit_geoms":list(edge[1:]),
                                "endpoint_model_clearance_m":round(endpoint_clear[0],4),
                                "endpoint_limit_geoms":list(endpoint_clear[1:])}
        return result
    finally:
        mask.close()
        room.close()
        absent_room.close()


def main() -> None:
    OUT.mkdir(parents=True,exist_ok=True)
    scenes=json.loads(SCENES.read_text())
    variants=next(row for row in scenes["rows"] if row["seed"]==88)["passable_variants"]
    exemplar=np.load(source.V81 / "owner-taught-exemplar.npy")
    vision=Vision("generic")
    rows=[screen(variant,vision,exemplar) for variant in variants]
    gate=all(row["fresh_correct_target"] and
             not row["matched_absent_accepted"] and
             not row["look_range_alarm"] and
             row.get("selected") is not None and
             row["selected"]["first_edge_model_min_m"]>=.05 and
             row["selected"]["endpoint_model_clearance_m"]>=.05
             for row in rows)
    result={"scope":"Read-only actor goal invalidation and reselection at exact V134 seed88 moved-distractor stops. Guarded fresh RGB-D identity+matched absent, actor-only current4/8/11cm, causal self-masked depth, selected-body full-graph annulus then bounded course view,6.5cm point rule. Evaluator geometry only after selection; no motion.",
            "source_sha256":sha(SOURCE),"scene_audit_sha256":sha(SCENES),
            "selected_design_sha256":sha(ROOT / "robot/ape-current-design.json"),
            "rows":rows,"paired_goal_reselection_screen_pass":gate}
    (OUT / "moved-goal-reselection-audit.json").write_text(json.dumps(result,indent=2)+"\n")
    (OUT / "audit_moved_goal_reselection_v135_source.py").write_bytes(SOURCE.read_bytes())
    for row in rows:
        print(row["variant"],"fresh",row["fresh_correct_target"],
              "annulus",row.get("annulus",{}).get("found"),
              "info",row.get("information_goal",{}).get("goal_cell"),
              "selected",row.get("selected",{}).get("goal_cell"),flush=True)
    print("V135 paired gate",gate,flush=True)


if __name__=="__main__":
    main()
