"""V109: guarded local information move and fresh owner-object recognition."""

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
from fast_range_guard import FastRangeGuard
from robot_geometry import RobotGeometry
from run import H, ROOT, W, Room, Vision, iou, sensed_range, sha
from run_live_local_bypass_v86 import run
from self_depth_mask import SelfDepthMask

SOURCE = Path(__file__)
RUNNER = SOURCE.with_name("run_live_local_bypass_v86.py")
V81 = ROOT / "artifacts/ape-closed-loop-cove50-v81"
V104 = ROOT / "artifacts/ape-closed-loop-cove50-v104"
V108 = ROOT / "artifacts/ape-closed-loop-cove50-v108/actor-information-goal-audit.json"
OUT = ROOT / "artifacts/ape-closed-loop-cove50-v109"


def load_initial(room: Room, mask: SelfDepthMask) -> dict:
    seed=85
    trace_path=V104 / "seed85/trace.json"
    prior=json.loads(trace_path.read_text())
    frame_count=sum("actor_input_sha256" in e for e in prior["events"])
    _,_,initial_paths,_,_=accepted_source(seed)
    paths=list(initial_paths)+[
        V104 / f"seed85/frame-{index}.npz" for index in range(1,frame_count+1)]
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
    visited={tuple(e["detail"]["planner_goal"]["committed_path_cells"][
        e["detail"]["planner_goal"]["route_index"]])
             for e in prior["events"] if e.get("action")=="guarded_approach_step"}
    return {"voxel":voxel,
            "actor_qpos":qpos,
            "estimated_pose_xyyaw":pose,
            "actual_pose_xyyaw_evaluator_only":np.asarray(
                prior["evaluator_final_actual_xyyaw"],dtype=float),
            "capture_time_s":stamp,
            "visited_cells":visited,
            "provenance":{"v104_trace_sha256":sha(trace_path),
                          "actor_input_sha256":hashes}}


def endpoint_views(room: Room, absent_room: Room, vision: Vision,
                   exemplar: np.ndarray, target: np.ndarray) -> dict:
    seed=85
    branch=OUT / "seed85-endpoint-views"
    branch.mkdir(parents=True,exist_ok=True)
    bearing=math.atan2(target[1]-room.estimated[1],
                       target[0]-room.estimated[0])-room.estimated[2]
    bearing=math.atan2(math.sin(bearing),math.cos(bearing))
    yaws=[float(np.clip(bearing+offset,-1.2,1.2))
          for offset in (0.,-.25,+.25)]
    torso=FastRangeGuard(band_m=.05)
    low=FastRangeGuard(band_m=.05)
    serial=0
    rows=[]
    video_path=branch / "episode.mp4"

    def sense(height: float) -> list[float | None]:
        nonlocal serial
        serial+=1
        return sensed_range(room,seed,1090000+serial,height_m=height)

    def guard_probe():
        torso.update(sense(.11),room.estimated,room.time_s)
        low.update(sense(.04),room.estimated,room.time_s)
        def probe(substep: int,fraction: float) -> dict | None:
            upper=torso.update(sense(.11),room.estimated,room.time_s)
            lower=low.update(sense(.04),room.estimated,room.time_s)
            if (upper["status"]=="near_body_change" or
                lower["status"]=="near_body_change"):
                return {"upper":upper,"lower":lower,
                        "capture_time_s":room.time_s,
                        "action_fraction":fraction}
            return None
        return probe

    status="no_fresh_acceptance"
    with imageio.get_writer(video_path,fps=4,codec="libx264",quality=7) as video:
        for index,yaw in enumerate(yaws):
            room.look(yaw,"lower_rgb",safety_probe=guard_probe())
            if room.safety_stop_observation is not None:
                status="look_range_alarm"
            rgb,depth=room.capture("lower_rgb",depth=True)
            camera=room.camera_pose_estimate("lower_rgb")
            qpos=room.pose_data.qpos.copy()
            source=branch / f"view-{index}-actor-input.npz"
            np.savez_compressed(source,rgb=np.asarray(rgb),depth_m=depth,
                                camera_origin_m=camera[0],
                                camera_rotation=camera[1],actor_qpos=qpos,
                                estimated_pose_xyyaw=room.estimated.copy(),
                                capture_time_s=room.time_s)
            present=infer(vision,exemplar,rgb,depth,camera,target,None,
                          branch / f"view-{index}-present-annotated.png")
            truth=room.truth_box("TomatoSauce","lower_rgb")
            correct=(present["accepted_box"] is not None and truth is not None
                     and iou(present["accepted_box"],truth)>=.25)
            frame={"actor_qpos":qpos,
                   "estimated_pose_xyyaw":room.estimated.copy(),
                   "capture_time_s":room.time_s}
            absent_source=branch / f"view-{index}-absent"
            absent_rgb,absent_depth,absent_camera,_=absent_capture(
                absent_room,seed,frame,room.actual.copy(),yaw,absent_source)
            absent=infer(vision,exemplar,absent_rgb,absent_depth,
                         absent_camera,target,None,
                         branch / f"view-{index}-absent-annotated.png")
            combined=Image.fromarray(np.concatenate((np.asarray(rgb),
                    np.asarray(room.capture_external())),axis=1))
            draw=ImageDraw.Draw(combined)
            draw.rectangle((0,0,combined.width,34),fill=(0,0,0))
            draw.text((8,8),f"V109 seed85 information view {yaw:+.2f} | "
                      f"accepted {present['accepted_box'] is not None}",
                      fill=(255,255,255))
            for _ in range(4):video.append_data(np.asarray(combined))
            rows.append({"head_yaw_rad":round(yaw,4),
                         "actor_input_sha256":sha(source),
                         "capture_time_s":round(room.time_s,3),
                         "accepted_box":present["accepted_box"],
                         "accepted_similarity":present["accepted_similarity"],
                         "accepted_correct_evaluator_only":correct,
                         "evaluator_truth_box":truth,
                         "present_proposals":present["proposals"],
                         "matched_absent_accepted":absent["accepted_box"] is not None,
                         "absent_actor_input_sha256":sha(
                             absent_source.with_suffix(".npz")),
                         "range_alarm":room.safety_stop_observation,
                         "evaluator_model_clearance_m":round(float(
                             room._current_obstacle_clearance()[0]),4)})
            print("view",yaw,"correct",correct,"absent",
                  absent["accepted_box"] is not None,flush=True)
            if status=="look_range_alarm":break
            if present["accepted_box"] is not None:
                status="accepted"
                break
    return {"status":status,"views":rows,
            "correct_target_reacquired":any(row[
                "accepted_correct_evaluator_only"] for row in rows),
            "wrong_acceptance":any(row["accepted_box"] is not None and
                                    not row["accepted_correct_evaluator_only"]
                                    for row in rows),
            "matched_absent_accepted":any(row[
                "matched_absent_accepted"] for row in rows),
            "evaluator_min_model_clearance_m":round(min(
                float(room.path_min_clearance[0]),
                float(room._current_obstacle_clearance()[0])),4),
            "video_sha256":sha(video_path)}


def main() -> None:
    OUT.mkdir(parents=True,exist_ok=True)
    audit=json.loads(V108.read_text())
    if not audit["actor_information_goal_screen_pass"]:
        raise AssertionError("V108 actor goal screen failed")
    chosen=audit["actor"]["selected"]
    room=Room()
    absent_room=Room()
    mask=SelfDepthMask(room.model,camera="lower_rgb",width=W,height=H)
    try:
        robot=RobotGeometry(room.model)
        initial=load_initial(room,mask)
        base=run(85,{"actor_goal_cell":chosen["route_cell"],
                     "blocked_cell":None},room,robot,output_root=OUT,
                 video_label="V109",goal_audit_path=V108,initial=initial,
                 self_mask=mask,allow_near_masked_depth=True,max_moves=4,
                 motion_consistency_stop=True)
        views=None
        if base["status"]=="local_goal_reached":
            exemplar_path=V81 / "owner-taught-exemplar.npy"
            exemplar=np.load(exemplar_path)
            vision=Vision("generic")
            target=np.asarray(audit["actor"]["actor_target_memory_xyz_m"][:2],
                              dtype=float)
            views=endpoint_views(room,absent_room,vision,exemplar,target)
    finally:
        mask.close()
        room.close()
        absent_room.close()
    gate=(base["status"]=="local_goal_reached" and
          base["executed_moves"]<=4 and
          base["evaluator_min_path_clearance_m"]>=.05 and
          not base["evaluator_model_overlap"] and
          views is not None and views["status"]=="accepted" and
          views["correct_target_reacquired"] and
          not views["wrong_acceptance"] and
          not views["matched_absent_accepted"] and
          views["evaluator_min_model_clearance_m"]>=.05)
    result={"scope":"Live V108 actor-selected seed85 information waypoint from V104 identity-lost stop; receding local4cm replan, fresh4/8/11cm selected-body6.5cm checks, independent10Hz4/11cm guard, scan registration plus motion-consistency stop, V95 self-masked near head depth and frozen taught-object bearing-gaze gate with matched absent. Kinematic, not target-area approach.",
            "source_sha256":sha(SOURCE),
            "runner_source_sha256":sha(RUNNER),
            "v108_audit_sha256":sha(V108),
            "selected_design_sha256":sha(ROOT / "robot/ape-current-design.json"),
            "base":{key:value for key,value in base.items() if key!="events"},
            "views":views,
            "live_information_recovery_gate_pass":gate}
    (OUT / "live-information-recovery-audit.json").write_text(
        json.dumps(result,indent=2)+"\n")
    (OUT / "run_live_information_recovery_v109_source.py").write_bytes(
        SOURCE.read_bytes())
    (OUT / "run_live_local_bypass_v86_imported_source.py").write_bytes(
        RUNNER.read_bytes())
    print("live information recovery gate:",gate,flush=True)


if __name__=="__main__":
    main()
