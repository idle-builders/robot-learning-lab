"""V128: height-specific actor point gap on the CreamCheese stop edge."""

from __future__ import annotations

import json
import math
from pathlib import Path

import mujoco
import numpy as np

from depth_voxel import DepthVoxelMap
from robot_geometry import RobotGeometry
from run import ROOT, Room, sensed_range, sha
from two_height_clearance import TwoHeightClearance

SOURCE = Path(__file__)
OUT = ROOT / "artifacts/ape-closed-loop-cove50-v128"
TRACE = ROOT / "artifacts/ape-closed-loop-cove50-v127/seed85-repair/trace.json"
FRAME = ROOT / "artifacts/ape-closed-loop-cove50-v127/seed85-repair/frame-11.npz"
HEIGHTS = (.04,.08,.11)
MARGIN = .065


def source_gaps(room: Room, robot: RobotGeometry, qpos: np.ndarray,
                pose: np.ndarray, goal: np.ndarray,
                scans: dict[float,list[float | None]]) -> dict:
    point = TwoHeightClearance(robot,qpos,pose,scans[.11],scans[.04],
                               DepthVoxelMap(),0.,range_8=scans[.08])
    sources = {h:point.sources[{.04:"range_4",.08:"range_8",.11:"range_11"}[h]]
               for h in HEIGHTS}
    rows = {}
    for label,active in (("4",(.04,)),("8",(.08,)),("11",(.11,)),
                         ("8+11",(.08,.11)),("4+8+11",HEIGHTS)):
        point.all_points = np.concatenate([sources[h] for h in active])
        gap = float(point.edge_gap(pose[:2],goal,pose[2]))
        rows[label] = {"actor_point_gap_m":round(gap,4),
                       "rejects_65cm_edge":gap<MARGIN,
                       "scan_points":int(sum(len(sources[h]) for h in active))}
    return rows


def exact_ray_hits(room: Room) -> dict:
    group=np.array([1,0,0,0,0,0],dtype=np.uint8)
    rows={}
    for height in HEIGHTS:
        found={}
        for index,angle in enumerate(np.linspace(-math.pi,math.pi,72,endpoint=False)):
            origin=np.array([room.actual[0],room.actual[1],height])
            direction=np.array([math.cos(room.actual[2]+angle),
                                math.sin(room.actual[2]+angle),0.])
            geom=np.array([-1],dtype=np.int32)
            distance=mujoco.mj_ray(room.model,room.data,origin,direction,group,
                                   True,room.model.body("torso").id,geom)
            if distance>=0:
                name=room.model.geom(int(geom[0])).name
                found.setdefault(name,[]).append({"beam":index,"distance_m":round(float(distance),4)})
        rows[str(height)]={name:hits for name,hits in found.items()
                           if name.startswith("loop_")}
    return rows


def main() -> None:
    OUT.mkdir(parents=True,exist_ok=True)
    trace=json.loads(TRACE.read_text())
    with np.load(FRAME) as z:
        qpos=z["actor_qpos"].copy()
        pose=z["estimated_pose_xyyaw"].copy()
    goal=np.asarray(trace["actor_local_goal_cell"])*.04
    room=Room()
    try:
        room.layout(85,absent=False,furnished=True)
        room.model.body_pos[room.model.body("loop_table").id]=(.40,.46,0.)
        room.set_table_height(.37)
        room.data.qpos[:]=qpos
        room.actual[:]=trace["evaluator_final_actual_xyyaw"]
        room.estimated[:]=pose
        room.head=float(qpos[room.head_addr])
        room._forward()
        robot=RobotGeometry(room.model)
        ideal=source_gaps(room,robot,qpos,pose,goal,
                          {height:room.range_scan(height) for height in HEIGHTS})
        hits=exact_ray_hits(room)
        noisy=[]
        for repeat in range(100):
            scans={height:sensed_range(room,85,1280000+repeat*3+index,
                                       height_m=height)
                   for index,height in enumerate(HEIGHTS)}
            gap=source_gaps(room,robot,qpos,pose,goal,scans)
            noisy.append({"repeat":repeat,
                          "point_gap_4_m":gap["4"]["actor_point_gap_m"],
                          "point_gap_8_11_m":gap["8+11"]["actor_point_gap_m"],
                          "point_gap_all_m":gap["4+8+11"]["actor_point_gap_m"],
                          "low_rejects":gap["4"]["rejects_65cm_edge"],
                          "high_only_rejects":gap["8+11"]["rejects_65cm_edge"],
                          "all_rejects":gap["4+8+11"]["rejects_65cm_edge"]})
    finally:
        room.close()
    result={"scope":"Actor point-gap height ablation at exact V127 seed85 stopped edge, selected robot joint/estimated pose input and central 72-beam virtual scans; evaluator only ray names and full-scene clearance. 100 seeded noisy scan triplets use 2.5% dropout and 4mm Gaussian range noise. Single scene, no physical sensor placement guarantee.",
            "source_sha256":sha(SOURCE),"run_source_sha256":sha(ROOT / "experiments/ape_high_level/closed_loop/run.py"),
            "trace_sha256":sha(TRACE),"frame_sha256":sha(FRAME),
            "estimated_pose_xyyaw":pose.tolist(),"next_actor_goal_xy_m":goal.tolist(),
            "ideal_height_ablation":ideal,"evaluator_exact_ray_hits":hits,
            "noisy_replays":noisy,
            "noisy_summary":{"count":len(noisy),
                             "low_rejects":sum(r["low_rejects"] for r in noisy),
                             "high_only_rejects":sum(r["high_only_rejects"] for r in noisy),
                             "all_rejects":sum(r["all_rejects"] for r in noisy)}}
    (OUT / "low-prop-sensor-ablation.json").write_text(json.dumps(result,indent=2)+"\n")
    (OUT / "audit_low_prop_sensors_v128_source.py").write_bytes(SOURCE.read_bytes())
    print("ideal",ideal,flush=True)
    print("noisy",result["noisy_summary"],flush=True)


if __name__=="__main__":
    main()
