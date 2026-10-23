"""V132: raw versus voxel-center low head depth at two stopped edges."""

from __future__ import annotations

import json
import math
from pathlib import Path

import numpy as np

from depth_voxel import DepthVoxelMap
from robot_geometry import RobotGeometry
from run import H,ROOT,W,Room,sha
from self_depth_mask import SelfDepthMask
from two_height_clearance import TwoHeightClearance
from run_live_committed_course_v121 import load_initial

SOURCE=Path(__file__)
OUT=ROOT / "artifacts/ape-closed-loop-cove50-v132"
CASES=((85,127,None),(88,131,(.36,.56)))
RESOLUTIONS=(.04,.03,.02,.01)


def project(depth: np.ndarray,camera: tuple[np.ndarray,np.ndarray],
            fovy_deg: float) -> np.ndarray:
    focal=H/(2*math.tan(math.radians(fovy_deg)/2))
    rows,cols=np.mgrid[4:H:8,4:W:8]
    distances=depth[rows,cols]
    valid=np.isfinite(distances)&(distances>.08)&(distances<1.4)
    rays=np.stack(((cols-W/2)/focal,-(rows-H/2)/focal,
                   -np.ones_like(distances)),axis=-1)[valid]
    origin,rotation=camera
    points=origin+(rays*distances[valid,None])@rotation.T
    return points[(points[:,2]>=.02)&(points[:,2]<=.16)]


def point_gap(robot: RobotGeometry,qpos: np.ndarray,pose: np.ndarray,
              goal: np.ndarray,points: np.ndarray) -> float:
    point=TwoHeightClearance(robot,qpos,pose,[None]*72,[None]*72,
                             DepthVoxelMap(),0.)
    point.all_points=points
    return float(point.edge_gap(pose[:2],goal,pose[2]))


def model_edge(room: Room,actual: np.ndarray,pose: np.ndarray,
               goal: np.ndarray) -> tuple[float,str,str]:
    delta=goal-pose[:2]
    rows=[]
    for fraction in np.linspace(0.,1.,21):
        room.actual[:]=actual
        room.actual[:2]+=.88*fraction*delta
        room._forward()
        rows.append(room._current_obstacle_clearance())
    return min(rows,key=lambda row:row[0])


def history_attribution(room: Room,mask: SelfDepthMask,
                        robot: RobotGeometry) -> dict:
    branch=ROOT / "artifacts/ape-closed-loop-cove50-v131/seed88-repair"
    voxel=load_initial(88,room,mask)["voxel"]
    fovy=float(room.model.cam_fovy[room.model.camera("lower_rgb").id])
    for index in range(1,12):
        with np.load(branch / f"frame-{index}.npz") as z:
            filtered,_=mask.apply(z["depth_m"],z["actor_qpos"])
            voxel.observe(filtered,(z["camera_origin_m"],
                                    z["camera_rotation"]),
                          float(z["capture_time_s"]),fovy_deg=fovy,
                          self_masked=True)
    with np.load(branch / "frame-11.npz") as z:
        qpos=z["actor_qpos"].copy()
        pose=z["estimated_pose_xyyaw"].copy()
        time_s=float(z["capture_time_s"])
    goal=np.asarray((.36,.56))
    by_age={}
    for age_limit in (.4,.8,1.6):
        best=(math.inf,None)
        delta=goal-pose[:2]
        for fraction in (.25,.5,.75,1.):
            future=pose.copy()
            future[:2]=pose[:2]+fraction*delta
            robot.set_state(robot.pose_qpos(qpos,future))
            primitives=robot.primitives()
            for cell,hit in voxel.occupied.items():
                z=(cell[2]+.5)*.04
                if not (.04<=z<=.16 and
                        hit>=voxel.free.get(cell,-math.inf) and
                        time_s-hit<=age_limit+1e-9):
                    continue
                xyz=(np.asarray(cell)+.5)*.04
                distance=RobotGeometry.solid_distance(xyz,primitives)
                if distance<best[0]:
                    best=(distance,{"cell":[int(c) for c in cell],"xyz_m":xyz.tolist(),
                                    "hit_time_s":hit,"age_s":time_s-hit,
                                    "edge_fraction":fraction})
        by_age[str(age_limit)]={"actor_point_gap_m":round(best[0],4),
                                "rejects_65cm_edge":best[0]<.065,
                                "nearest":best[1]}
    return by_age


def main() -> None:
    OUT.mkdir(parents=True,exist_ok=True)
    room=Room()
    mask=SelfDepthMask(room.model,camera="lower_rgb",width=W,height=H)
    robot=RobotGeometry(room.model)
    rows=[]
    try:
        fovy=float(room.model.cam_fovy[room.model.camera("lower_rgb").id])
        for seed,version,explicit_goal in CASES:
            branch=ROOT / f"artifacts/ape-closed-loop-cove50-v{version}/seed{seed}-repair"
            trace_path=branch / "trace.json"
            frame_path=branch / "frame-11.npz"
            trace=json.loads(trace_path.read_text())
            with np.load(frame_path) as z:
                depth=z["depth_m"].copy()
                qpos=z["actor_qpos"].copy()
                pose=z["estimated_pose_xyyaw"].copy()
                camera=(z["camera_origin_m"].copy(),z["camera_rotation"].copy())
            goal=(np.asarray(trace["actor_local_goal_cell"])*.04
                  if explicit_goal is None else np.asarray(explicit_goal))
            room.reset()
            room.layout(seed,absent=False,furnished=True)
            room.model.body_pos[room.model.body("loop_table").id]=(
                .40,(1 if seed%2 else -1)*.46,0.)
            room.set_table_height(.37)
            room.data.qpos[:]=qpos
            room.head=float(qpos[room.head_addr])
            actual=np.asarray(trace["evaluator_final_actual_xyyaw"])
            truth=model_edge(room,actual,pose,goal)
            filtered,_=mask.apply(depth,qpos)
            raw=project(filtered,camera,fovy)
            variants={"raw":{"points":len(raw),
                             "actor_point_gap_m":round(point_gap(
                                 robot,qpos,pose,goal,raw),4)}}
            for resolution in RESOLUTIONS:
                cells=np.unique(np.floor(raw/resolution).astype(int),axis=0)
                centers=(cells+.5)*resolution
                variants[str(resolution)]={
                    "points":len(centers),
                    "actor_point_gap_m":round(point_gap(
                        robot,qpos,pose,goal,centers),4)}
            for value in variants.values():
                value["rejects_65cm_edge"]=value["actor_point_gap_m"]<.065
            rows.append({"seed":seed,"source_version":version,
                         "trace_sha256":sha(trace_path),
                         "frame_sha256":sha(frame_path),
                         "next_edge_goal_xy_m":goal.tolist(),
                         "corrected_model_edge_min_m":round(float(truth[0]),4),
                         "corrected_limiting_scene_geom":truth[2],
                         "variants":variants})
            print(seed,"model",round(truth[0],4),
                  {k:v["actor_point_gap_m"] for k,v in variants.items()},flush=True)
    finally:
        mask.close()
        room.close()
    room=Room()
    mask=SelfDepthMask(room.model,camera="lower_rgb",width=W,height=H)
    robot=RobotGeometry(room.model)
    try:
        history=history_attribution(room,mask,robot)
    finally:
        mask.close()
        room.close()
    result={"scope":"Evaluator-labeled actor calibrated single-frame low-head-depth resolution screen at seed85 unsafe and seed88 safe stopped next edges. Same self-mask, stride8 projection,2–16cm endpoint band,6.5cm point rule; raw depth versus occupied voxel centers 4/3/2/1cm. Causal seed88 voxel-history age attribution. No actor motion.",
            "source_sha256":sha(SOURCE),"rows":rows,
            "seed88_history_by_age":history}
    (OUT / "low-depth-quantization.json").write_text(json.dumps(result,indent=2)+"\n")
    (OUT / "audit_low_depth_quantization_v132_source.py").write_bytes(SOURCE.read_bytes())
    print("seed88 history",history,flush=True)


if __name__=="__main__":
    main()
