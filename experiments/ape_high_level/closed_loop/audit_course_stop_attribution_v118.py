"""V118: split saved seed85 course-stop first-edge gaps by sensor source."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from audit_fresh_recovery_approach_v110 import accepted_source
from depth_voxel import DepthVoxelMap
from footprint_route import observed_free
from robot_geometry import RobotGeometry
from run import H, ROOT, W, Room, sha
from self_depth_mask import SelfDepthMask
from two_height_clearance import TwoHeightClearance

SOURCE=Path(__file__)
V116=ROOT / "artifacts/ape-closed-loop-cove50-v116/course-progress-view-audit.json"
V117=ROOT / "artifacts/ape-closed-loop-cove50-v117"
OUT=ROOT / "artifacts/ape-closed-loop-cove50-v118"
GRID_M=.04
NEIGHBORS=((1,0),(-1,0),(0,1),(0,-1),
           (1,1),(1,-1),(-1,1),(-1,-1))


def main() -> None:
    OUT.mkdir(parents=True,exist_ok=True)
    seed=85
    trace_path=V117 / "seed85-repair/trace.json"
    trace=json.loads(trace_path.read_text())
    if trace["status"]!="no_actor_bypass_route":
        raise AssertionError("Expected exact V117 seed85 stopped state")
    stop=next(e for e in reversed(trace["events"])
              if e.get("action")=="no_actor_bypass_route")
    scans={height:stop[f"range_{height}_m"] for height in (11,8,4)}
    count=sum("actor_input_sha256" in e for e in trace["events"])
    _,_,initial_paths,_,_=accepted_source(seed)
    paths=list(initial_paths)+[
        V117 / f"seed85-repair/frame-{index}.npz"
        for index in range(1,count+1)]
    room=Room()
    mask=SelfDepthMask(room.model,camera="lower_rgb",width=W,height=H)
    try:
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
        actual=np.asarray(trace["evaluator_final_actual_xyyaw"],dtype=float)
        room.reset()
        room.layout(seed,absent=False,furnished=True)
        room.model.body_pos[room.model.body("loop_table").id]=(.40,.46,0.)
        room.set_table_height(.37)
        room.data.qpos[:]=qpos
        room.actual[:]=actual
        room.estimated[:]=pose
        room.head=float(qpos[room.head_addr])
        room.time_s=stamp
        room._forward()
        robot=RobotGeometry(room.model)
        point=TwoHeightClearance(robot,qpos,pose,scans[11],scans[4],
                                 voxel,stamp,range_8=scans[8])
        course_audit=json.loads(V116.read_text())
        course_row=next(r["actor"] for r in course_audit["rows"]
                        if r["seed"]==seed)
        course=np.asarray(course_row["actor_course_unit_xy"])
        goal_xy=GRID_M*np.asarray(course_row["actor_goal_cell"])
        origin=(int(round(pose[0]/GRID_M)),
                int(round(pose[1]/GRID_M)))
        rows=[]
        original_points=point.all_points
        for dx,dy in NEIGHBORS:
            node=(origin[0]+dx,origin[1]+dy)
            end=GRID_M*np.asarray(node,dtype=float)
            delta=end-pose[:2]
            known={str(height):observed_free(end,pose,scans[height])
                   for height in (11,8,4)}
            source_gap={}
            for name,points in point.sources.items():
                point.all_points=points
                source_gap[name]=float(point.edge_gap(pose[:2],end,pose[2]))
            point.all_points=original_points
            actor_gap=float(point.edge_gap(pose[:2],end,pose[2]))
            samples=[]
            for index in range(21):
                room.actual[:]=actual
                room.actual[:2]+=.88*delta*index/20
                room._forward()
                samples.append(float(room._current_obstacle_clearance()[0]))
            model_gap=min(samples)
            actor_pass=all(known.values()) and actor_gap>=.065
            row={"neighbor_cell":node,
                 "command_delta_xy_m":delta.tolist(),
                 "actor_course_progress_m":round(float(delta@course),4),
                 "actor_goal_distance_change_m":round(float(
                     np.linalg.norm(goal_xy-pose[:2])-
                     np.linalg.norm(goal_xy-end)),4),
                 "actor_range_known_free":known,
                 "actor_source_gap_m":{name:round(gap,4)
                                       for name,gap in source_gap.items()},
                 "actor_point_gap_m":round(actor_gap,4),
                 "actor_authorized":actor_pass,
                 "evaluator_min_model_clearance_m":round(model_gap,4),
                 "evaluator_safe_5cm":model_gap>=.05,
                 "evaluator_false_block":not actor_pass and model_gap>=.05}
            rows.append(row)
    finally:
        mask.close()
        room.close()
    result={"scope":"Read-only attribution of exact V117 seed85 no-route stop. Saved actor 4/8/11cm range and causal V95 self-masked head depth split by source under unchanged selected-body6.5cm point rule; only after actor classification score 21 selected-model samples along each 0.4s slip step. Evaluator truth does not authorize motion.",
            "source_sha256":sha(SOURCE),
            "v117_trace_sha256":sha(trace_path),
            "v116_course_audit_sha256":sha(V116),
            "selected_design_sha256":sha(ROOT / "robot/ape-current-design.json"),
            "actor_input_sha256":hashes,
            "actor_pose_xyyaw":pose.tolist(),
            "evaluator_actual_xyyaw":actual.tolist(),
            "actor_goal_xy_m":goal_xy.tolist(),
            "actor_point_source_counts":point.source_counts(),
            "neighbors":rows,
            "safe_courseward_false_blocks":sum(r[
                "evaluator_false_block"] and r[
                "actor_course_progress_m"]>.01 for r in rows)}
    (OUT / "course-stop-attribution-audit.json").write_text(
        json.dumps(result,indent=2)+"\n")
    (OUT / "audit_course_stop_attribution_v118_source.py").write_bytes(
        SOURCE.read_bytes())
    for row in rows:
        print(row["neighbor_cell"],"course",row[
            "actor_course_progress_m"],"actor",row["actor_point_gap_m"],
            "true",row["evaluator_min_model_clearance_m"],
            "cause",min(row["actor_source_gap_m"],key=row[
                "actor_source_gap_m"].get),flush=True)
    print("safe courseward false blocks:",result[
        "safe_courseward_false_blocks"],flush=True)


if __name__=="__main__":
    main()
