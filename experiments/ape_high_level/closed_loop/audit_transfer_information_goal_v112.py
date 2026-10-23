"""V112: unchanged V108 actor information-goal rule at two V111 stops."""

from __future__ import annotations

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
from target_view import crop_surface_points, ray_status, view_ray_evidence
from two_height_clearance import TwoHeightClearance

SOURCE=Path(__file__)
V81=ROOT / "artifacts/ape-closed-loop-cove50-v81"
V111=ROOT / "artifacts/ape-closed-loop-cove50-v111"
OUT=ROOT / "artifacts/ape-closed-loop-cove50-v112"
MIN_BASELINE_M=.08
MAX_BASELINE_M=.24
POINT_MARGIN_M=.065


def actor_select(seed: int,room: Room,robot: RobotGeometry,
                 mask: SelfDepthMask,trace: dict,
                 frame_paths: list[Path]) -> dict:
    voxel=DepthVoxelMap()
    fovy=float(room.model.cam_fovy[room.model.camera("lower_rgb").id])
    hashes=[]
    for path in frame_paths:
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
    accepted=next(e for e in reversed(trace["events"])
                  if e.get("accepted_box") is not None)
    accepted_index=sum("actor_input_sha256" in e for e in
                       trace["events"][:trace["events"].index(accepted)+1])
    accepted_path=V111 / f"seed{seed}/frame-{accepted_index}.npz"
    with np.load(accepted_path) as z:
        accepted_depth=z["depth_m"].copy()
        accepted_camera=(z["camera_origin_m"].copy(),
                         z["camera_rotation"].copy())
    target=np.asarray(accepted["accepted_world_xyz_m"],dtype=float)
    surface=crop_surface_points(accepted_depth,accepted["accepted_box"],
                                accepted_camera,fovy,target)
    latest_move=next(e for e in reversed(trace["events"])
                     if e.get("action")=="guarded_approach_step")
    planned=latest_move["detail"]["planner_goal"]
    route=planned["committed_path_cells"]
    first_untraveled=planned["route_index"]+1
    scan11=sensed_range(room,seed,1120011,height_m=.11)
    scan8=sensed_range(room,seed,1120008,height_m=.08)
    scan4=sensed_range(room,seed,1120004,height_m=.04)
    point=TwoHeightClearance(robot,qpos,pose,scan11,scan4,voxel,stamp,
                             range_8=scan8)
    original_estimated=room.estimated.copy()
    original_head=room.head
    rows=[]
    chosen=None
    for index in range(first_untraveled,len(route)):
        node=route[index]
        xy=.04*np.asarray(node,dtype=float)
        baseline=float(np.linalg.norm(xy-pose[:2]))
        cursor=pose[:2].copy()
        gaps=[]
        range_cleared=True
        for prior in route[first_untraveled:index+1]:
            nxt=.04*np.asarray(prior,dtype=float)
            range_cleared &= all(observed_free(nxt,pose,scan)
                                 for scan in (scan11,scan8,scan4))
            gaps.append(float(point.edge_gap(cursor,nxt,pose[2])))
            cursor=nxt
        path_clear=(range_cleared and bool(gaps) and
                    min(gaps)>=POINT_MARGIN_M)
        yaw=math.atan2(target[1]-xy[1],target[0]-xy[0])-pose[2]
        yaw=float(np.clip(math.atan2(math.sin(yaw),math.cos(yaw)),
                          -1.2,1.2))
        room.estimated[:]=[xy[0],xy[1],pose[2]]
        room.head=yaw
        room._forward()
        camera=room.camera_pose_estimate("lower_rgb")
        center=ray_status(voxel,camera[0],target,stamp)
        view=view_ray_evidence(voxel,camera,surface,fovy,(H,W),stamp)
        enough=(view["in_frustum_rays"]>=10 and
                view["free_rays"]>=10 and
                view["free_rays"]/view["in_frustum_rays"]>=.5)
        qualifies=(MIN_BASELINE_M<=baseline<=MAX_BASELINE_M and
                   path_clear and center=="free" and enough)
        row={"route_index":index,"route_cell":node,
             "candidate_xy_m":xy.tolist(),
             "baseline_m":round(baseline,4),
             "path_range_cleared":range_cleared,
             "path_point_edge_gaps_m":[round(g,4) for g in gaps],
             "center_ray_state":center,
             "surface_ray_evidence":view,
             "head_yaw_rad":round(yaw,4),
             "actor_selected":qualifies}
        rows.append(row)
        if qualifies:
            chosen=row
            break
    room.estimated[:]=original_estimated
    room.head=original_head
    room._forward()
    return {"status":"selected" if chosen is not None else
            "no_information_goal",
            "actor_input_sha256":hashes,
            "last_accepted_actor_frame_sha256":sha(accepted_path),
            "actor_target_memory_xyz_m":target.tolist(),
            "accepted_crop_surface_points":len(surface),
            "actor_stop_pose_xyyaw":pose.tolist(),
            "current_route_goal_xy_m":planned["goal_xy_m"],
            "current_route_first_untraveled_index":first_untraveled,
            "actor_range_source_counts":point.source_counts(),
            "candidate_rows":rows,"selected":chosen}


def evaluator_score(seed: int,room: Room,absent_room: Room,
                    vision: Vision,exemplar: np.ndarray,trace: dict,
                    actor: dict,qpos: np.ndarray,stamp: float) -> dict | None:
    chosen=actor["selected"]
    if chosen is None:return None
    branch=OUT / f"seed{seed}"
    branch.mkdir(parents=True,exist_ok=True)
    xy=np.asarray(chosen["candidate_xy_m"],dtype=float)
    target=np.asarray(actor["actor_target_memory_xyz_m"][:2],dtype=float)
    pose=np.asarray(trace["final_estimated_pose_xyyaw"],dtype=float)
    actual=np.asarray(trace["evaluator_final_actual_xyyaw"],dtype=float)
    offset=actual[:2]-pose[:2]
    yaw=chosen["head_yaw_rad"]
    room.estimated[:]=[xy[0],xy[1],pose[2]]
    room.actual[:]=[xy[0]+offset[0],xy[1]+offset[1],actual[2]]
    room.data.qpos[:]=qpos
    room.head=yaw
    room.time_s=stamp
    room._forward()
    rgb,depth=room.capture("lower_rgb",depth=True)
    camera=room.camera_pose_estimate("lower_rgb")
    actor_qpos=room.pose_data.qpos.copy()
    actor_path=branch / "selected-actor-input.npz"
    np.savez_compressed(actor_path,rgb=np.asarray(rgb),depth_m=depth,
                        camera_origin_m=camera[0],camera_rotation=camera[1],
                        actor_qpos=actor_qpos,
                        estimated_pose_xyyaw=room.estimated.copy(),
                        capture_time_s=stamp)
    present=infer(vision,exemplar,rgb,depth,camera,target,None,
                  branch / "present-annotated.png")
    truth=room.truth_box("TomatoSauce","lower_rgb")
    correct=(present["accepted_box"] is not None and truth is not None and
             iou(present["accepted_box"],truth)>=.25)
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
    draw.text((8,8),f"V112 seed{seed} actor info goal "
              f"{chosen['route_cell']} | accepted {correct}",
              fill=(255,255,255))
    video_path=branch / "selected-endpoint.mp4"
    with imageio.get_writer(video_path,fps=4,codec="libx264",quality=7) as video:
        for _ in range(8):video.append_data(np.asarray(combined))
    return {"actor_input_sha256":sha(actor_path),
            "evaluator_model_clearance_m":round(float(
                room._current_obstacle_clearance()[0]),4),
            "present_accepted_box":present["accepted_box"],
            "present_accepted_similarity":present["accepted_similarity"],
            "evaluator_truth_box":truth,
            "present_accept_correct_evaluator_only":correct,
            "matched_absent_accepted":absent["accepted_box"] is not None,
            "present_proposals":present["proposals"],
            "video_sha256":sha(video_path)}


def replay(seed: int,room: Room,absent_room: Room,
           robot: RobotGeometry,mask: SelfDepthMask,
           vision: Vision,exemplar: np.ndarray) -> dict:
    trace_path=V111 / f"seed{seed}/trace.json"
    trace=json.loads(trace_path.read_text())
    frame_count=sum("actor_input_sha256" in e for e in trace["events"])
    last_path=V111 / f"seed{seed}/frame-{frame_count}.npz"
    with np.load(last_path) as z:
        qpos=z["actor_qpos"].copy()
        stamp=float(z["capture_time_s"])
    _,_,initial_paths,_,_=accepted_source(seed)
    paths=list(initial_paths)+[
        V111 / f"seed{seed}/frame-{i}.npz"
        for i in range(1,frame_count+1)]
    room.reset()
    room.layout(seed,absent=False,furnished=True)
    room.model.body_pos[room.model.body("loop_table").id]=(
        .40,(1 if seed % 2 else -1)*.46,0.)
    room.set_table_height(.37)
    room.data.qpos[:]=qpos
    room.actual[:]=trace["evaluator_final_actual_xyyaw"]
    room.estimated[:]=trace["final_estimated_pose_xyyaw"]
    room.head=float(qpos[room.head_addr])
    room.time_s=stamp
    room._forward()
    actor=actor_select(seed,room,robot,mask,trace,paths)
    scored=evaluator_score(seed,room,absent_room,vision,exemplar,
                           trace,actor,qpos,stamp)
    return {"seed":seed,"v111_trace_sha256":sha(trace_path),
            "v111_final_frame_sha256":sha(last_path),
            "actor":actor,"evaluator_selected_endpoint":scored}


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
        rows=[replay(seed,room,absent_room,robot,mask,vision,exemplar)
              for seed in (85,88)]
    finally:
        mask.close()
        room.close()
        absent_room.close()
    gate=all(row["actor"]["selected"] is not None and
             row["evaluator_selected_endpoint"] is not None and
             row["evaluator_selected_endpoint"]["evaluator_model_clearance_m"]>=.05 and
             row["evaluator_selected_endpoint"]["present_accept_correct_evaluator_only"] and
             not row["evaluator_selected_endpoint"]["matched_absent_accepted"]
             for row in rows)
    result={"scope":"Transfer unchanged V108 actor information-view selector to both exact V111 learned-identity-lost stops, using only causal selected-body RGB-D, self-masked voxel map, current4/8/11cm virtual range and retained actor route; truth only scores selected counterfactual endpoint with matched absent. No motion executed.",
            "source_sha256":sha(SOURCE),
            "v81_exemplar_sha256":sha(exemplar_path),
            "selected_design_sha256":sha(ROOT / "robot/ape-current-design.json"),
            "rows":rows,"paired_information_goal_transfer_gate_pass":gate}
    (OUT / "transfer-information-goal-audit.json").write_text(
        json.dumps(result,indent=2)+"\n")
    (OUT / "audit_transfer_information_goal_v112_source.py").write_bytes(
        SOURCE.read_bytes())
    for row in rows:
        print(row["seed"],"selected",None if row["actor"]["selected"] is None
              else row["actor"]["selected"]["route_cell"],
              "eval",None if row["evaluator_selected_endpoint"] is None
              else (row["evaluator_selected_endpoint"][
                  "evaluator_model_clearance_m"],row[
                  "evaluator_selected_endpoint"][
                  "present_accept_correct_evaluator_only"]),flush=True)
    print("paired information goal transfer gate:",gate,flush=True)


if __name__=="__main__":
    main()
