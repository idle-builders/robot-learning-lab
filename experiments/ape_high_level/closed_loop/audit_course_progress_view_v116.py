"""V116: actor-side course-progress waypoint screen after lost annulus route."""

from __future__ import annotations

import heapq
import json
import math
from pathlib import Path

import imageio.v2 as imageio
import numpy as np
from PIL import Image, ImageDraw

from audit_fresh_recovery_approach_v110 import accepted_source
from audit_side_look_target_v81 import absent_capture, infer
from depth_voxel import DepthVoxelMap
from footprint_route import observed_free
from robot_geometry import RobotGeometry
from run import H, ROOT, W, Room, Vision, iou, sensed_range, sha
from self_depth_mask import SelfDepthMask
from two_height_clearance import TwoHeightClearance

SOURCE=Path(__file__)
V81=ROOT / "artifacts/ape-closed-loop-cove50-v81"
V90=ROOT / "artifacts/ape-closed-loop-cove50-v90"
V104=ROOT / "artifacts/ape-closed-loop-cove50-v104"
V109=ROOT / "artifacts/ape-closed-loop-cove50-v109"
OUT=ROOT / "artifacts/ape-closed-loop-cove50-v116"
GRID_M=.04
MARGIN_M=.065
RADIUS_M=.35
MAX_COST_M=.8
MIN_COURSE_PROGRESS_M=.18
NEIGHBORS=((1,0),(-1,0),(0,1),(0,-1),
           (1,1),(1,-1),(-1,1),(-1,-1))


def actor_plan(seed: int,room: Room,robot: RobotGeometry,
               mask: SelfDepthMask, *, clearance_cost_weight: float = .002,
               physical_travel_cap: bool = False
               ) -> tuple[dict,np.ndarray,np.ndarray,float]:
    target,actual,paths,accepted,source_audit=accepted_source(seed)
    voxel=DepthVoxelMap()
    fovy=float(room.model.cam_fovy[room.model.camera("lower_rgb").id])
    hashes=[]
    for path in paths:
        with np.load(path) as z:
            qpos=z["actor_qpos"].copy()
            pose=z["estimated_pose_xyyaw"].copy()
            depth=z["depth_m"].copy()
            camera=(z["camera_origin_m"].copy(),
                    z["camera_rotation"].copy())
            stamp=float(z["capture_time_s"])
        filtered,_=mask.apply(depth,qpos)
        voxel.observe(filtered,camera,stamp,fovy_deg=fovy,self_masked=True)
        hashes.append(sha(path))
    room.reset()
    room.layout(seed,absent=False,furnished=True)
    room.model.body_pos[room.model.body("loop_table").id]=(
        .40,(1 if seed%2 else -1)*.46,0.)
    room.set_table_height(.37)
    room.data.qpos[:]=qpos
    room.actual[:]=actual
    room.estimated[:]=pose
    room.head=float(qpos[room.head_addr])
    room.time_s=stamp
    room._forward()
    side_trace_path=V90 / f"seed{seed}-actor/trace.json"
    side_trace=json.loads(side_trace_path.read_text())
    side_name=side_trace["side"]
    forward=target-pose[:2]
    forward/=np.linalg.norm(forward)
    left=np.array([-forward[1],forward[0]])
    side=left if side_name=="left" else -left
    course=forward+side
    course/=np.linalg.norm(course)
    scan11=sensed_range(room,seed,1160011,height_m=.11)
    scan8=sensed_range(room,seed,1160008,height_m=.08)
    scan4=sensed_range(room,seed,1160004,height_m=.04)
    point=TwoHeightClearance(robot,qpos,pose,scan11,scan4,voxel,stamp,
                             range_8=scan8)
    visited={tuple(cell) for cell in side_trace["actor_graph_path_cells"]}
    v104=json.loads((V104 / f"seed{seed}/trace.json").read_text())
    visited|={tuple(e["detail"]["planner_goal"]["committed_path_cells"][
        e["detail"]["planner_goal"]["route_index"]])
        for e in v104["events"] if e.get("action")=="guarded_approach_step"}
    if seed==85:
        v109=json.loads((V109 / "seed85-repair/trace.json").read_text())
        visited|={tuple(e["detail"]["plan_cells"][1])
                  for e in v109["events"] if e.get("action")=="replanned_base_step"}
    origin=tuple(int(round(float(value)/GRID_M)) for value in pose[:2])
    queue=[(0.,0,origin)]
    costs={origin:0.}
    travel={origin:0.}
    parent={}
    gaps={}
    serial=0
    goal=None
    while queue and len(costs)<500:
        cost,_,node=heapq.heappop(queue)
        if (cost>costs[node]+1e-9 or
            (travel[node] if physical_travel_cap else cost)>MAX_COST_M):
            continue
        node_xy=GRID_M*np.asarray(node,dtype=float)
        if (node!=origin and
            float((node_xy-pose[:2])@course)>=MIN_COURSE_PROGRESS_M):
            goal=node
            break
        start_xy=pose[:2] if node==origin else node_xy
        for dx,dy in NEIGHBORS:
            nxt=(node[0]+dx,node[1]+dy)
            end_xy=GRID_M*np.asarray(nxt,dtype=float)
            if np.linalg.norm(end_xy-pose[:2])>RADIUS_M:continue
            if not (observed_free(end_xy,pose,scan11) and
                    observed_free(end_xy,pose,scan4)):continue
            gap=point.edge_gap(start_xy,end_xy,pose[2])
            if gap<MARGIN_M:continue
            step_length=float(np.linalg.norm(end_xy-start_xy))
            next_travel=travel[node]+step_length
            if physical_travel_cap and next_travel>MAX_COST_M:
                continue
            new_cost=(cost+step_length+
                      clearance_cost_weight/max(gap,.03)+
                      (.04 if nxt in visited else 0.))
            if new_cost>=costs.get(nxt,math.inf):continue
            costs[nxt]=new_cost
            travel[nxt]=next_travel
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
    first_xy=None if len(path)<2 else GRID_M*np.asarray(path[1])
    selected={"seed":seed,"source_audit_sha256":sha(source_audit),
              "actor_input_sha256":hashes,
              "side_trace_sha256":sha(side_trace_path),
              "actor_pose_xyyaw":pose.tolist(),
              "actor_target_from_image_xy_m":target.tolist(),
              "accepted_source_similarity":accepted["similarity"],
              "actor_side":side_name,
              "actor_course_unit_xy":course.tolist(),
              "actor_goal_cell":goal,
              "actor_path_cells":path,
              "actor_path_edge_gaps_m":[round(gaps[node],4)
                                         for node in path[1:]],
              "actor_first_delta_xy_m":None if first_xy is None else
                  (first_xy-pose[:2]).tolist(),
              "expanded_cells":len(costs),
              "max_reachable_course_progress_m":round(max(float(
                  (GRID_M*np.asarray(node)-pose[:2])@course)
                  for node in costs),4),
              "actor_goal_course_progress_m":None if goal is None else round(float(
                  (GRID_M*np.asarray(goal)-pose[:2])@course),4),
              "actor_goal_target_gap_m":None if goal is None else round(float(
                  np.linalg.norm(GRID_M*np.asarray(goal)-target)),4),
              "actor_voxel_occupied":len(voxel.occupied),
              "actor_point_source_counts":point.source_counts()}
    return selected,qpos,actual,stamp


def evaluator_score(seed: int,room: Room,absent_room: Room,
                    vision: Vision,exemplar: np.ndarray,actor: dict,
                    qpos: np.ndarray,actual: np.ndarray,stamp: float) -> dict | None:
    goal=actor["actor_goal_cell"]
    if goal is None:return None
    branch=OUT / f"seed{seed}"
    branch.mkdir(parents=True,exist_ok=True)
    pose=np.asarray(actor["actor_pose_xyyaw"])
    target=np.asarray(actor["actor_target_from_image_xy_m"])
    first=np.asarray(actor["actor_first_delta_xy_m"])
    samples=[]
    for i in range(21):
        room.actual[:]=actual
        room.actual[:2]+=.88*first*i/20
        room._forward()
        samples.append(float(room._current_obstacle_clearance()[0]))
    first_min=min(samples)
    goal_xy=GRID_M*np.asarray(goal,dtype=float)
    room.actual[:]=actual
    room.actual[:2]+=goal_xy-pose[:2]
    room.estimated[:]=[goal_xy[0],goal_xy[1],pose[2]]
    yaw=math.atan2(target[1]-goal_xy[1],target[0]-goal_xy[0])-pose[2]
    yaw=float(np.clip(math.atan2(math.sin(yaw),math.cos(yaw)),-1.2,1.2))
    room.head=yaw
    room._forward()
    endpoint_clear=float(room._current_obstacle_clearance()[0])
    rgb,depth=room.capture("lower_rgb",depth=True)
    camera=room.camera_pose_estimate("lower_rgb")
    actor_qpos=room.pose_data.qpos.copy()
    input_path=branch / "candidate-actor-input.npz"
    np.savez_compressed(input_path,rgb=np.asarray(rgb),depth_m=depth,
                        camera_origin_m=camera[0],camera_rotation=camera[1],
                        actor_qpos=actor_qpos,
                        estimated_pose_xyyaw=room.estimated.copy(),
                        capture_time_s=stamp)
    present=infer(vision,exemplar,rgb,depth,camera,target,None,
                  branch / "present-annotated.png")
    truth=room.truth_box("TomatoSauce","lower_rgb")
    correct=(present["accepted_box"] is not None and truth is not None
             and iou(present["accepted_box"],truth)>=.25)
    frame={"actor_qpos":actor_qpos,
           "estimated_pose_xyyaw":room.estimated.copy(),
           "capture_time_s":stamp}
    absent_source=branch / "absent"
    absent_rgb,absent_depth,absent_camera,_=absent_capture(
        absent_room,seed,frame,room.actual.copy(),yaw,absent_source)
    absent=infer(vision,exemplar,absent_rgb,absent_depth,absent_camera,
                 target,None,branch / "absent-annotated.png")
    combined=Image.fromarray(np.concatenate((np.asarray(rgb),
             np.asarray(room.capture_external())),axis=1))
    draw=ImageDraw.Draw(combined)
    draw.rectangle((0,0,combined.width,34),fill=(0,0,0))
    draw.text((8,8),f"V116 seed{seed} course goal {goal} | "
              f"accepted {correct}",fill=(255,255,255))
    video_path=branch / "candidate-endpoint.mp4"
    with imageio.get_writer(video_path,fps=4,codec="libx264",quality=7) as video:
        for _ in range(8):video.append_data(np.asarray(combined))
    return {"actor_input_sha256":sha(input_path),
            "first_edge_evaluator_min_clearance_m":round(first_min,4),
            "endpoint_evaluator_model_clearance_m":round(endpoint_clear,4),
            "candidate_head_yaw_rad":round(yaw,4),
            "present_accepted_box":present["accepted_box"],
            "present_accepted_similarity":present["accepted_similarity"],
            "present_accept_correct_evaluator_only":correct,
            "matched_absent_accepted":absent["accepted_box"] is not None,
            "evaluator_truth_box":truth,
            "present_proposals":present["proposals"],
            "video_sha256":sha(video_path)}


def main() -> None:
    OUT.mkdir(parents=True,exist_ok=True)
    room=Room()
    absent_room=Room()
    mask=SelfDepthMask(room.model,camera="lower_rgb",width=W,height=H)
    exemplar_path=V81 / "owner-taught-exemplar.npy"
    exemplar=np.load(exemplar_path)
    vision=Vision("generic")
    try:
        robot=RobotGeometry(room.model)
        rows=[]
        for seed in (85,88):
            actor,qpos,actual,stamp=actor_plan(seed,room,robot,mask)
            scored=evaluator_score(seed,room,absent_room,vision,exemplar,
                                   actor,qpos,actual,stamp)
            rows.append({"seed":seed,"actor":actor,
                         "evaluator_selected_goal":scored})
    finally:
        mask.close()
        room.close()
        absent_room.close()
    gate=all(row["actor"]["actor_goal_cell"] is not None and
             row["evaluator_selected_goal"] is not None and
             row["evaluator_selected_goal"]["first_edge_evaluator_min_clearance_m"]>=.05 and
             row["evaluator_selected_goal"]["endpoint_evaluator_model_clearance_m"]>=.05 and
             row["evaluator_selected_goal"]["present_accept_correct_evaluator_only"] and
             not row["evaluator_selected_goal"]["matched_absent_accepted"]
             for row in rows)
    result={"scope":"Read-only V98-style actor course=target-image bearing plus V90 committed high-obstacle side from V109 seed85/V105 seed88 recovered starts. Current4/8/11cm virtual range, self-masked near head depth,6.5cm selected-body point graph,35cm radius/18cm course projection and visited penalty. Evaluator scores first edge, endpoint body and taught-object image/matched absent only after actor goal selection; no motion.",
            "source_sha256":sha(SOURCE),
            "v81_exemplar_sha256":sha(exemplar_path),
            "selected_design_sha256":sha(ROOT / "robot/ape-current-design.json"),
            "rows":rows,"paired_course_progress_view_screen_pass":gate}
    (OUT / "course-progress-view-audit.json").write_text(
        json.dumps(result,indent=2)+"\n")
    (OUT / "audit_course_progress_view_v116_source.py").write_bytes(
        SOURCE.read_bytes())
    for row in rows:
        score=row["evaluator_selected_goal"]
        print(row["seed"],"goal",row["actor"]["actor_goal_cell"],
              "max course",row["actor"]["max_reachable_course_progress_m"],
              "eval",None if score is None else (score[
                  "first_edge_evaluator_min_clearance_m"],score[
                  "endpoint_evaluator_model_clearance_m"],score[
                  "present_accept_correct_evaluator_only"]),flush=True)
    print("paired course-progress view screen:",gate,flush=True)


if __name__=="__main__":
    main()
