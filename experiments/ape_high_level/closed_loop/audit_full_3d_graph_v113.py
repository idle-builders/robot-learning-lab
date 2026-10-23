"""V113: screen fully 3D point-checked target graph paths."""

from __future__ import annotations

import json
import math
import time
from pathlib import Path

import numpy as np

from depth_voxel import DepthVoxelMap
from goal_map_route import GoalMapRoute
from robot_geometry import RobotGeometry
from run import H, ROOT, W, Room, sensed_range, sha
from self_depth_mask import SelfDepthMask
from two_height_clearance import TwoHeightClearance

SOURCE=Path(__file__)
V104=ROOT / "artifacts/ape-closed-loop-cove50-v104"
V105=ROOT / "artifacts/ape-closed-loop-cove50-v105"
V109=ROOT / "artifacts/ape-closed-loop-cove50-v109"
OUT=ROOT / "artifacts/ape-closed-loop-cove50-v113"


def accepted_source(seed: int) -> tuple[np.ndarray,np.ndarray,list[Path],dict,Path]:
    prior_path=V104 / f"seed{seed}/trace.json"
    prior=json.loads(prior_path.read_text())
    v104_frames=sum("actor_input_sha256" in e for e in prior["events"])
    paths=[V104 / f"seed{seed}/frame-{i}.npz"
           for i in range(1,v104_frames+1)]
    if seed==85:
        audit_path=V109 / "live-information-recovery-audit.json"
        audit=json.loads(audit_path.read_text())
        base=json.loads((V109 / "seed85-repair/trace.json").read_text())
        base_frames=sum("actor_input_sha256" in e for e in base["events"])
        paths += [V109 / f"seed85-repair/frame-{i}.npz"
                  for i in range(1,base_frames+1)]
        paths.append(V109 / "seed85-endpoint-views/view-0-actor-input.npz")
        view=audit["views"]["views"][0]
        accepted_box=view["accepted_box"]
        proposal=next(row for row in view["present_proposals"]
                      if row["box"]==accepted_box)
        actual=np.asarray(base["evaluator_final_actual_xyyaw"],dtype=float)
    else:
        audit_path=V105 / "memory-bearing-look-audit.json"
        audit=json.loads(audit_path.read_text())
        view=next(row for row in audit["rows"] if row["seed"]==seed)["views"][0]
        accepted_box=view["present"]["accepted_box"]
        proposal=next(row for row in view["present"]["proposals"]
                      if row["box"]==accepted_box)
        paths.append(V105 / "seed88/view-0-actor-input.npz")
        actual=np.asarray(prior["evaluator_final_actual_xyyaw"],dtype=float)
    if proposal["depth_world_xyz_m"] is None:
        raise AssertionError("Fresh accepted proposal lacks actor depth")
    target=np.asarray(proposal["depth_world_xyz_m"][:2],dtype=float)
    return target,actual,paths,proposal,audit_path


def screen(seed: int,room: Room,robot: RobotGeometry,
           mask: SelfDepthMask) -> dict:
    target,actual,paths,accepted,audit_path=accepted_source(seed)
    voxel=DepthVoxelMap()
    fovy=float(room.model.cam_fovy[room.model.camera("lower_rgb").id])
    inputs=[]
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
        inputs.append(sha(path))
    room.reset()
    room.layout(seed,absent=False,furnished=True)
    room.model.body_pos[room.model.body("loop_table").id]=(
        .40,(1 if seed % 2 else -1)*.46,0.)
    room.set_table_height(.37)
    room.data.qpos[:]=qpos
    room.actual[:]=actual
    room.estimated[:]=pose
    room.head=float(qpos[room.head_addr])
    room.time_s=stamp
    room._forward()
    scan11=sensed_range(room,seed,1100011,height_m=.11)
    scan8=sensed_range(room,seed,1100008,height_m=.08)
    scan4=sensed_range(room,seed,1100004,height_m=.04)
    point=TwoHeightClearance(robot,qpos,pose,scan11,scan4,voxel,stamp,
                             range_8=scan8)
    planner=GoalMapRoute()
    plan_start=time.perf_counter()
    command=planner.plan(scan11,pose,target,None,stamp,searching=False,
                         edge_clearance=point.edge_gap,
                         point_all_graph_edges=True)
    plan_wall_s=time.perf_counter()-plan_start
    first_step=None
    first_true=None
    whole_true=None
    whole_min_edge=None
    path=[]
    if command is not None:
        heading,_,_,detail=command
        delta=detail["step_distance_m"]*np.asarray([
            math.cos(heading),math.sin(heading)])
        first_step=delta.tolist()
        node=planner.goal
        origin=planner.cell(pose[:2])
        while node is not None and node!=origin:
            path.append(node)
            node=planner.last_parent.get(node)
        path.reverse()
        samples=[]
        for i in range(21):
            room.actual[:]=actual
            room.actual[:2]+=.88*delta*i/20
            room._forward()
            samples.append(float(room._current_obstacle_clearance()[0]))
        first_true=min(samples)
        samples=[]
        cursor=pose[:2].copy()
        for edge_index,node in enumerate(path):
            endpoint=planner.xy(node)
            for i in range(21):
                actor_xy=cursor+(endpoint-cursor)*i/20
                room.actual[:]=actual
                room.actual[:2]+=actor_xy-pose[:2]
                room._forward()
                clearance=float(room._current_obstacle_clearance()[0])
                samples.append((clearance,edge_index))
            cursor=endpoint
        if samples:
            whole_true,whole_min_edge=min(samples)
    return {"seed":seed,"source_audit_sha256":sha(audit_path),
            "actor_input_sha256":inputs,
            "fresh_actor_target_xy_m":target.tolist(),
            "accepted_similarity":accepted["similarity"],
            "actor_pose_xyyaw":pose.tolist(),
            "actor_target_gap_m":round(float(np.linalg.norm(
                pose[:2]-target)),4),
            "actor_range_source_counts":point.source_counts(),
            "actor_upper_voxel_occupied":len(voxel.occupied),
            "planner_command":None if command is None else {
                "heading_rad":round(command[0],4),
                "actor_first_edge_point_gap_m":round(command[2],4),
                "detail":command[3]},
            "planner_debug":planner.last_debug,
            "planner_wall_time_s":round(plan_wall_s,4),
            "actor_route_cells":path,
            "first_delta_xy_m":first_step,
            "first_edge_evaluator_min_clearance_m":None if first_true is None
                else round(first_true,4),
            "whole_route_evaluator_min_clearance_m":None if whole_true is None
                else round(whole_true,4),
            "whole_route_min_clearance_edge_index":whole_min_edge}


def main() -> None:
    OUT.mkdir(parents=True,exist_ok=True)
    room=Room()
    mask=SelfDepthMask(room.model,camera="lower_rgb",width=W,height=H)
    try:
        robot=RobotGeometry(room.model)
        rows=[screen(seed,room,robot,mask) for seed in (85,88)]
    finally:
        mask.close()
        room.close()
    gate=all(row["planner_command"] is not None and
             row["planner_command"]["detail"]["goal_kind"]=="annulus" and
             row["first_edge_evaluator_min_clearance_m"]>=.05 and
             row["whole_route_evaluator_min_clearance_m"]>=.05
             for row in rows)
    result={"scope":"Read-only full-graph all-edge selected-body6.5cm point check from V109 seed85 and V105 seed88 freshly accepted taught-object RGB-D images, causal self-masked depth and current4/8/11cm virtual range. Evaluator samples actor-chosen complete route and first edge only after selection. No motion executed.",
            "source_sha256":sha(SOURCE),
            "selected_design_sha256":sha(ROOT / "robot/ape-current-design.json"),
            "rows":rows,"full_3d_graph_screen_pass":gate}
    (OUT / "full-3d-graph-audit.json").write_text(
        json.dumps(result,indent=2)+"\n")
    (OUT / "audit_full_3d_graph_v113_source.py").write_bytes(
        SOURCE.read_bytes())
    for row in rows:
        print(row["seed"],"gap",row["actor_target_gap_m"],"goal",
              None if row["planner_command"] is None else row[
                  "planner_command"]["detail"]["goal_xy_m"],
              "first true",row["first_edge_evaluator_min_clearance_m"],
              "whole true",row["whole_route_evaluator_min_clearance_m"],
              "plan seconds",row["planner_wall_time_s"],
              flush=True)
    print("full 3D graph screen:",gate,flush=True)


if __name__=="__main__":
    main()
