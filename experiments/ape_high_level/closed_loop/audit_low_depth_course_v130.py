"""V130: low head-depth actor point gap across saved safe course steps."""

from __future__ import annotations

import json
import math
from pathlib import Path

import numpy as np

from depth_voxel import DepthVoxelMap,VOXEL_M
from robot_geometry import RobotGeometry
from run import H,ROOT,W,Room,sha
from self_depth_mask import SelfDepthMask
from two_height_clearance import TwoHeightClearance

SOURCE=Path(__file__)
OUT=ROOT / "artifacts/ape-closed-loop-cove50-v130"
BASE=ROOT / "artifacts/ape-closed-loop-cove50-v127"
MARGIN=.065


def gap(point: TwoHeightClearance,points: np.ndarray,start: np.ndarray,
        end: np.ndarray,yaw: float) -> float:
    point.all_points=points
    return float(point.edge_gap(start,end,yaw))


def main() -> None:
    OUT.mkdir(parents=True,exist_ok=True)
    room=Room()
    mask=SelfDepthMask(room.model,camera="lower_rgb",width=W,height=H)
    robot=RobotGeometry(room.model)
    rows=[]
    try:
        fovy=float(room.model.cam_fovy[room.model.camera("lower_rgb").id])
        for seed in (85,88):
            branch=BASE / f"seed{seed}-repair"
            trace=json.loads((branch / "trace.json").read_text())
            moves=[e["detail"] for e in trace["events"]
                   if e.get("action")=="replanned_base_step"]
            for move in moves:
                index=int(move["move_index"])
                frame=branch / f"frame-{2*index-1}.npz"
                with np.load(frame) as z:
                    qpos=z["actor_qpos"].copy()
                    pose=z["estimated_pose_xyyaw"].copy()
                    depth=z["depth_m"].copy()
                    camera=(z["camera_origin_m"].copy(),z["camera_rotation"].copy())
                    time_s=float(z["capture_time_s"])
                future=pose[:2]+np.asarray(move["command_delta_xy_m"])
                filtered,_=mask.apply(depth,qpos)
                voxel=DepthVoxelMap()
                voxel.observe(filtered,camera,time_s,fovy_deg=fovy,
                              stride=8,self_masked=True)
                low=np.asarray([[(x+.5)*VOXEL_M,(y+.5)*VOXEL_M,(z+.5)*VOXEL_M]
                                for (x,y,z),hit in voxel.occupied.items()
                                if .04 <= (z+.5)*VOXEL_M <= .16
                                and hit>=voxel.free.get((x,y,z),-math.inf)],
                               dtype=float).reshape(-1,3)
                point=TwoHeightClearance(robot,qpos,pose,move["range_11_m"],
                                         move["range_4_m"],DepthVoxelMap(),time_s,
                                         range_8=move["range_8_m"])
                high=np.concatenate((point.sources["range_8"],point.sources["range_11"]))
                all_range=np.concatenate((high,point.sources["range_4"]))
                low_gap=gap(point,low,pose[:2],future,pose[2])
                high_gap=gap(point,high,pose[:2],future,pose[2])
                high_low_gap=gap(point,np.concatenate((high,low)),
                                 pose[:2],future,pose[2])
                all_range_gap=gap(point,all_range,pose[:2],future,pose[2])
                row={"seed":seed,"move_index":index,
                     "actor_preframe_sha256":sha(frame),
                     "low_depth_points":len(low),
                     "low_depth_gap_m":round(low_gap,4),
                     "high_range_gap_m":round(high_gap,4),
                     "high_range_plus_low_depth_gap_m":round(high_low_gap,4),
                     "all_range_gap_m":round(all_range_gap,4),
                     "low_depth_rejects":low_gap<MARGIN,
                     "high_plus_low_rejects":high_low_gap<MARGIN,
                     "all_range_rejects":all_range_gap<MARGIN}
                rows.append(row)
                print(seed,index,row["low_depth_gap_m"],
                      row["high_range_plus_low_depth_gap_m"],flush=True)
    finally:
        mask.close()
        room.close()
    summary={"executed_edges":len(rows),
             "low_depth_rejections":sum(r["low_depth_rejects"] for r in rows),
             "high_plus_low_rejections":sum(r["high_plus_low_rejects"] for r in rows),
             "all_range_rejections":sum(r["all_range_rejects"] for r in rows)}
    result={"scope":"Read-only actor low-depth candidate across eleven saved executed V127 course edges: per-edge pre-move selected joints/estimated pose/calibrated self-masked RGB-D and noisy 4/8/11cm scans; frozen 6.5cm point rule. Does not authorize motion or qualify physical sensor.",
            "source_sha256":sha(SOURCE),"corrected_v127_audit_sha256":sha(BASE / "corrected-course-comparison.json"),
            "rows":rows,"summary":summary}
    (OUT / "low-depth-course-screen.json").write_text(json.dumps(result,indent=2)+"\n")
    (OUT / "audit_low_depth_course_v130_source.py").write_bytes(SOURCE.read_bytes())
    print("summary",summary,flush=True)


if __name__=="__main__":
    main()
