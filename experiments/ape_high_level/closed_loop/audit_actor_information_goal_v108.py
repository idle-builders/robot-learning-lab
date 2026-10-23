"""V108: actor-only information-waypoint choice, then evaluator endpoint score."""

from __future__ import annotations

import json
import math
from pathlib import Path

import imageio.v2 as imageio
import numpy as np
from PIL import Image, ImageDraw

from audit_fresh_target_approach_v101 import accepted_source
from audit_side_look_target_v81 import absent_capture, infer
from depth_voxel import DepthVoxelMap
from footprint_route import observed_free
from robot_geometry import RobotGeometry
from run import H, ROOT, W, Room, Vision, iou, sensed_range, sha
from self_depth_mask import SelfDepthMask
from target_view import crop_surface_points, ray_status, view_ray_evidence
from two_height_clearance import TwoHeightClearance

SOURCE = Path(__file__)
V81 = ROOT / "artifacts/ape-closed-loop-cove50-v81"
V104 = ROOT / "artifacts/ape-closed-loop-cove50-v104"
OUT = ROOT / "artifacts/ape-closed-loop-cove50-v108"
MIN_INFO_BASELINE_M = .08
MAX_INFO_BASELINE_M = .24
POINT_MARGIN_M = .065


def actor_select(room: Room, robot: RobotGeometry, mask: SelfDepthMask,
                 trace: dict, frame_paths: list[Path]) -> dict:
    voxel = DepthVoxelMap()
    fovy = float(room.model.cam_fovy[room.model.camera("lower_rgb").id])
    input_hashes = []
    for path in frame_paths:
        with np.load(path) as z:
            qpos = z["actor_qpos"].copy()
            depth = z["depth_m"].copy()
            camera = (z["camera_origin_m"].copy(),
                      z["camera_rotation"].copy())
            stamp = float(z["capture_time_s"])
            pose = z["estimated_pose_xyyaw"].copy()
        filtered, _ = mask.apply(depth, qpos)
        voxel.observe(filtered, camera, stamp, fovy_deg=fovy,
                      self_masked=True)
        input_hashes.append(sha(path))
    last_accepted = next(event for event in reversed(trace["events"])
                         if event.get("accepted_box") is not None)
    accepted_frame_index = sum("actor_input_sha256" in event
        for event in trace["events"][:trace["events"].index(last_accepted)+1])
    accepted_path = V104 / f"seed85/frame-{accepted_frame_index}.npz"
    with np.load(accepted_path) as z:
        accepted_depth = z["depth_m"].copy()
        accepted_camera = (z["camera_origin_m"].copy(),
                           z["camera_rotation"].copy())
    target_xyz = np.asarray(last_accepted["accepted_world_xyz_m"],
                            dtype=float)
    surface = crop_surface_points(accepted_depth,
                                  last_accepted["accepted_box"],
                                  accepted_camera, fovy, target_xyz)
    route = next(event for event in trace["events"]
                 if event["action"] == "guarded_approach_step")[
                     "detail"]["planner_goal"]["committed_path_cells"]
    latest_move = next(event for event in reversed(trace["events"])
                       if event["action"] == "guarded_approach_step")
    first_untraveled = latest_move["detail"]["planner_goal"]["route_index"]+1
    scan11 = sensed_range(room, 85, 1080011, height_m=.11)
    scan8 = sensed_range(room, 85, 1080008, height_m=.08)
    scan4 = sensed_range(room, 85, 1080004, height_m=.04)
    point = TwoHeightClearance(robot, qpos, pose, scan11, scan4,
                               voxel, stamp, range_8=scan8)
    original_estimated = room.estimated.copy()
    original_head = room.head
    candidates = []
    chosen = None
    for index in range(first_untraveled, len(route)):
        node = route[index]
        candidate = .04 * np.asarray(node,dtype=float)
        baseline = float(np.linalg.norm(candidate-pose[:2]))
        edge_gaps = []
        range_cleared = True
        cursor = pose[:2].copy()
        for prior in route[first_untraveled:index+1]:
            nxt = .04*np.asarray(prior,dtype=float)
            range_cleared &= all(observed_free(nxt,pose,scan)
                                  for scan in (scan11,scan8,scan4))
            edge_gaps.append(float(point.edge_gap(cursor,nxt,pose[2])))
            cursor = nxt
        path_clear = (range_cleared and bool(edge_gaps) and
                      min(edge_gaps) >= POINT_MARGIN_M)
        yaw = math.atan2(target_xyz[1]-candidate[1],
                         target_xyz[0]-candidate[0])-pose[2]
        yaw = float(np.clip(math.atan2(math.sin(yaw),math.cos(yaw)),
                            -1.2,1.2))
        room.estimated[:] = [candidate[0],candidate[1],pose[2]]
        room.head = yaw
        room._forward()
        camera = room.camera_pose_estimate("lower_rgb")
        center_state = ray_status(voxel,camera[0],target_xyz,stamp)
        view = view_ray_evidence(voxel,camera,surface,fovy,(H,W),stamp)
        enough_rays = (view["in_frustum_rays"] >= 10 and
                       view["free_rays"] >= 10 and
                       view["free_rays"] / view["in_frustum_rays"] >= .5)
        accepted = (MIN_INFO_BASELINE_M <= baseline <=
                    MAX_INFO_BASELINE_M and path_clear and
                    center_state == "free" and enough_rays)
        row = {"route_index":index,"route_cell":node,
               "candidate_xy_m":candidate.tolist(),
               "baseline_m":round(baseline,4),
               "path_range_cleared":range_cleared,
               "path_point_edge_gaps_m":[round(gap,4) for gap in edge_gaps],
               "center_ray_state":center_state,
               "surface_ray_evidence":view,
               "head_yaw_rad":round(yaw,4),
               "actor_selected":accepted}
        candidates.append(row)
        if accepted:
            chosen = row
            break
    room.estimated[:] = original_estimated
    room.head = original_head
    room._forward()
    return {"status":"selected" if chosen is not None else "no_information_goal",
            "last_accepted_actor_frame_sha256":sha(accepted_path),
            "actor_input_sha256":input_hashes,
            "actor_target_memory_xyz_m":target_xyz.tolist(),
            "accepted_crop_surface_points":len(surface),
            "actor_stop_pose_xyyaw":pose.tolist(),
            "actor_range_source_counts":point.source_counts(),
            "candidate_rows":candidates,
            "selected":chosen}


def evaluator_score(seed: int, room: Room, absent_room: Room,
                    vision: Vision, exemplar: np.ndarray,
                    trace: dict, actor: dict, qpos: np.ndarray,
                    stamp: float) -> dict | None:
    chosen = actor["selected"]
    if chosen is None:
        return None
    goal = np.asarray(chosen["candidate_xy_m"],dtype=float)
    target = np.asarray(actor["actor_target_memory_xyz_m"][:2],dtype=float)
    pose = np.asarray(trace["final_estimated_pose_xyyaw"],dtype=float)
    actual = np.asarray(trace["evaluator_final_actual_xyyaw"],dtype=float)
    offset = actual[:2]-pose[:2]
    yaw = chosen["head_yaw_rad"]
    room.estimated[:] = [goal[0],goal[1],pose[2]]
    room.actual[:] = [goal[0]+offset[0],goal[1]+offset[1],actual[2]]
    room.data.qpos[:] = qpos
    room.head = yaw
    room.time_s = stamp
    room._forward()
    rgb, depth = room.capture("lower_rgb",depth=True)
    camera = room.camera_pose_estimate("lower_rgb")
    actor_qpos = room.pose_data.qpos.copy()
    actor_path = OUT / "selected-actor-input.npz"
    np.savez_compressed(actor_path,rgb=np.asarray(rgb),depth_m=depth,
                        camera_origin_m=camera[0],camera_rotation=camera[1],
                        actor_qpos=actor_qpos,
                        estimated_pose_xyyaw=room.estimated.copy(),
                        capture_time_s=stamp)
    present = infer(vision,exemplar,rgb,depth,camera,target,None,
                    OUT / "selected-present-annotated.png")
    truth = room.truth_box("TomatoSauce","lower_rgb")
    correct = (present["accepted_box"] is not None and truth is not None
               and iou(present["accepted_box"],truth) >= .25)
    frame = {"actor_qpos":actor_qpos,
             "estimated_pose_xyyaw":room.estimated.copy(),
             "capture_time_s":stamp}
    absent_source = OUT / "selected-absent"
    absent_rgb,absent_depth,absent_camera,_ = absent_capture(
        absent_room,seed,frame,room.actual.copy(),yaw,absent_source)
    absent = infer(vision,exemplar,absent_rgb,absent_depth,absent_camera,
                   target,None,OUT / "selected-absent-annotated.png")
    combined = Image.fromarray(np.concatenate((np.asarray(rgb),
                    np.asarray(room.capture_external())),axis=1))
    draw=ImageDraw.Draw(combined)
    draw.rectangle((0,0,combined.width,34),fill=(0,0,0))
    draw.text((8,8),f"V108 actor information goal {chosen['route_cell']} | "
              f"accepted {correct}",fill=(255,255,255))
    video_path=OUT / "selected-endpoint.mp4"
    with imageio.get_writer(video_path,fps=4,codec="libx264",quality=7) as video:
        for _ in range(8): video.append_data(np.asarray(combined))
    return {"selected_actor_input_sha256":sha(actor_path),
            "evaluator_model_clearance_m":round(float(
                room._current_obstacle_clearance()[0]),4),
            "present_accepted_box":present["accepted_box"],
            "present_accepted_similarity":present["accepted_similarity"],
            "evaluator_truth_box":truth,
            "present_accept_correct_evaluator_only":correct,
            "matched_absent_accepted":absent["accepted_box"] is not None,
            "present_proposals":present["proposals"],
            "video_sha256":sha(video_path)}


def main() -> None:
    OUT.mkdir(parents=True,exist_ok=True)
    seed=85
    trace_path=V104 / "seed85/trace.json"
    trace=json.loads(trace_path.read_text())
    final_frame_index=sum("actor_input_sha256" in event for event in trace["events"])
    final_path=V104 / f"seed85/frame-{final_frame_index}.npz"
    with np.load(final_path) as z:
        qpos=z["actor_qpos"].copy()
        stamp=float(z["capture_time_s"])
    _,_,initial_paths,_,_=accepted_source(seed)
    frame_paths=list(initial_paths)+[
        V104 / f"seed85/frame-{index}.npz"
        for index in range(1,final_frame_index+1)]
    room=Room()
    absent_room=Room()
    mask=SelfDepthMask(room.model,camera="lower_rgb",width=W,height=H)
    exemplar_path=V81 / "owner-taught-exemplar.npy"
    exemplar=np.load(exemplar_path)
    vision=Vision("generic")
    try:
        room.reset()
        room.layout(seed,absent=False,furnished=True)
        room.model.body_pos[room.model.body("loop_table").id]=(.40,.46,0.)
        room.set_table_height(.37)
        room.data.qpos[:]=qpos
        room.actual[:]=trace["evaluator_final_actual_xyyaw"]
        room.estimated[:]=trace["final_estimated_pose_xyyaw"]
        room.head=float(qpos[room.head_addr])
        room.time_s=stamp
        room._forward()
        robot=RobotGeometry(room.model)
        actor=actor_select(room,robot,mask,trace,frame_paths)
        scored=evaluator_score(seed,room,absent_room,vision,exemplar,
                               trace,actor,qpos,stamp)
    finally:
        mask.close()
        room.close()
        absent_room.close()
    gate=(scored is not None and
          scored["evaluator_model_clearance_m"] >= .05 and
          scored["present_accept_correct_evaluator_only"] and
          not scored["matched_absent_accepted"])
    result={"scope":"Actor-only V104 seed85 lost-identity information-goal selection from saved RGB-D crop, self-masked 3D depth map, 4/8/11cm current ranges, preselected path and selected-body6.5cm edge checks. Truth and learned-object endpoint acceptance scored only after actor goal selection. Counterfactual endpoint only, no live motion.",
            "source_sha256":sha(SOURCE),
            "v104_trace_sha256":sha(trace_path),
            "v104_final_frame_sha256":sha(final_path),
            "v81_exemplar_sha256":sha(exemplar_path),
            "selected_design_sha256":sha(ROOT / "robot/ape-current-design.json"),
            "actor":actor,"evaluator_selected_endpoint":scored,
            "actor_information_goal_screen_pass":gate}
    (OUT / "actor-information-goal-audit.json").write_text(
        json.dumps(result,indent=2)+"\n")
    (OUT / "audit_actor_information_goal_v108_source.py").write_bytes(
        SOURCE.read_bytes())
    print("actor selected",None if actor["selected"] is None else actor[
        "selected"]["route_cell"],"screen",gate,flush=True)


if __name__=="__main__":
    main()
