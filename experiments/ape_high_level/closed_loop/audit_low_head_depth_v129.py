"""V129: low-band head-depth point clearance, present versus prop-removed."""

from __future__ import annotations

import json
import math
from pathlib import Path

import numpy as np

from depth_voxel import DepthVoxelMap, VOXEL_M
from robot_geometry import RobotGeometry
from run import H, ROOT, W, Room, sha
from self_depth_mask import SelfDepthMask
from two_height_clearance import TwoHeightClearance

SOURCE=Path(__file__)
OUT=ROOT / "artifacts/ape-closed-loop-cove50-v129"
TRACE=ROOT / "artifacts/ape-closed-loop-cove50-v127/seed85-repair/trace.json"
FRAME=ROOT / "artifacts/ape-closed-loop-cove50-v127/seed85-repair/frame-11.npz"
STRIDES=(4,8,12,16,24,32)
MARGIN=.065


def point_gap(robot: RobotGeometry,qpos: np.ndarray,pose: np.ndarray,
              goal: np.ndarray,voxel: DepthVoxelMap,time_s: float) -> tuple[float,int,list[float] | None]:
    points=np.asarray([[(x+.5)*VOXEL_M,(y+.5)*VOXEL_M,(z+.5)*VOXEL_M]
                       for (x,y,z),hit in voxel.occupied.items()
                       if .04 <= (z+.5)*VOXEL_M <= .16
                       and hit >= voxel.free.get((x,y,z),-math.inf)
                       and time_s-hit <= 4.],dtype=float).reshape(-1,3)
    candidate=TwoHeightClearance(robot,qpos,pose,[None]*72,[None]*72,
                                 DepthVoxelMap(),time_s)
    candidate.all_points=points
    gap=float(candidate.edge_gap(pose[:2],goal,pose[2]))
    nearest=None
    if len(points):
        best=(math.inf,None)
        delta=goal-pose[:2]
        for fraction in (.25,.5,.75,1.):
            future=pose.copy()
            future[:2]=pose[:2]+fraction*delta
            robot.set_state(robot.pose_qpos(qpos,future))
            distances=RobotGeometry.solid_distances(points,robot.primitives())
            index=int(np.argmin(distances))
            if distances[index]<best[0]:
                best=(float(distances[index]),points[index].tolist())
        nearest=best[1]
    return gap,len(points),nearest


def main() -> None:
    OUT.mkdir(parents=True,exist_ok=True)
    trace=json.loads(TRACE.read_text())
    with np.load(FRAME) as z:
        present_depth=z["depth_m"].copy()
        present_camera=(z["camera_origin_m"].copy(),z["camera_rotation"].copy())
        qpos=z["actor_qpos"].copy()
        pose=z["estimated_pose_xyyaw"].copy()
        time_s=float(z["capture_time_s"])
    goal=np.asarray(trace["actor_local_goal_cell"])*.04
    room=Room()
    mask=SelfDepthMask(room.model,camera="lower_rgb",width=W,height=H)
    rows=[]
    try:
        room.layout(85,absent=False,furnished=True)
        room.model.body_pos[room.model.body("loop_table").id]=(.40,.46,0.)
        room.set_table_height(.37)
        room.data.qpos[:]=qpos
        room.actual[:]=trace["evaluator_final_actual_xyyaw"]
        room.estimated[:]=pose
        room.head=float(qpos[room.head_addr])
        room._forward()
        room.model.body_pos[room.model.body("loop_CreamCheese").id]=(10.,0.,.052)
        room._forward()
        _,absent_depth=room.capture("lower_rgb",depth=True)
        absent_camera=room.camera_pose_estimate("lower_rgb")
        robot=RobotGeometry(room.model)
        fovy=float(room.model.cam_fovy[room.model.camera("lower_rgb").id])
        for label,depth,camera in (("present",present_depth,present_camera),
                                   ("CreamCheese_removed",absent_depth,absent_camera)):
            filtered,mask_stats=mask.apply(depth,qpos)
            for stride in STRIDES:
                voxel=DepthVoxelMap()
                summary=voxel.observe(filtered,camera,time_s,fovy_deg=fovy,
                                      stride=stride,self_masked=True)
                gap,count,nearest=point_gap(robot,qpos,pose,goal,voxel,time_s)
                rows.append({"layout":label,"stride_px":stride,
                             "actor_low_depth_gap_m":round(gap,4),
                             "rejects_65cm_edge":gap<MARGIN,
                             "low_occupied_voxels":count,
                             "nearest_low_point_xyz_m":nearest,
                             "depth_rays_used":summary["depth_rays_used"],
                             "mask_stats":mask_stats})
                print(label,stride,round(gap,4),count,flush=True)
    finally:
        mask.close()
        room.close()
    result={"scope":"Read-only actor-calibrated low-band head-depth point check at exact V127 seed85 stop; paired prop-present saved RGB-D and prop-removed counterfactual render, same estimated pose/encoders/6.5cm point rule. Counterfactual is evaluation only; no motion or physical sensor signoff.",
            "source_sha256":sha(SOURCE),"trace_sha256":sha(TRACE),
            "present_frame_sha256":sha(FRAME),"rows":rows}
    (OUT / "low-head-depth-ablation.json").write_text(json.dumps(result,indent=2)+"\n")
    (OUT / "audit_low_head_depth_v129_source.py").write_bytes(SOURCE.read_bytes())


if __name__=="__main__":
    main()
