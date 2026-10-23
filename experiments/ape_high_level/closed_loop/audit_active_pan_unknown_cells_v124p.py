"""Locate remaining V123p first-back-step unknown shell samples.

The exact sampled points and LOS checks are diagnostics only. They do not
turn an unobserved 4cm voxel into actor-certified free space.
"""

from __future__ import annotations

import json
import math
from pathlib import Path

import numpy as np

from articulated_person_scene import START_HAND_XYZ_M, add_person, set_person_pose
import audit_blocked_range_mounts_v319 as sensor
import direct_structural_rooms_v270 as structural
from depth_voxel import DepthVoxelMap
from direct_structural_rooms_v270 import selected_visual
from pre_show_actor_planner import ROUTE_FREE_AGE_S, freshness_filtered
from robot_geometry import RobotGeometry
from run import ROOT, sha
from run_articulated_owner_low_drift_v92 import OfferDriftRoom
from run_moving_low_clutter_v302 import STOOL_SIZE
from run_owner_exit_height_v96 import CASES
from run_pre_show_reserve_screen_v100 import (COMMAND_STEP_M, DIRECTIONS,
                                              START_S, V98)
from run_pre_show_active_pan_v123p import DECISION_S
from audit_pre_show_unknown_los_v116 import (CANDIDATE_MOUNTS,
                                             actual_point, line_of_sight)


SOURCE=Path(__file__)
V123=ROOT/"artifacts/ape-closed-loop-cove50-active-pan/report.json"
OUT=ROOT/"artifacts/ape-closed-loop-cove50-active-pan/unknown-cells.json"
LAYOUT="yaw+1.4-lower_depth-outward_plus_dense_inward_diagnostic"
RAY_NAMES=("rear_low_near","rear_mid_near","right_upper",
           "left_low_inward","right_low_inward")


def unknown_shell_at(robot: RobotGeometry, voxel_map: DepthVoxelMap,
                     qpos: np.ndarray, pose: np.ndarray,
                     z_min: float, z_max: float) -> tuple[list[dict],dict]:
    current=pose.copy()
    future=current.copy()
    future[:2]+=COMMAND_STEP_M*np.asarray(DIRECTIONS["back"])
    heading=np.array((-1.,0.,0.))
    visited=set()
    rows=[]
    for fraction in (.25,.5,.75,1.):
        candidate=current+fraction*(future-current)
        robot.set_state(robot.pose_qpos(qpos,candidate))
        solids=robot.primitives()
        for point,normal in robot.surface(solids):
            if point[2]<z_min or point[2]>z_max:
                continue
            if float(normal@heading)<.2:
                continue
            if robot.solid_distance(point,solids)<-.005:
                continue
            sample=point+.05*normal
            cell=voxel_map.cell(sample)
            if cell in visited:
                continue
            visited.add(cell)
            rows.append({"cell":cell,"sample":sample,
                         "state":voxel_map.state(sample,DECISION_S)})
    reference=robot.swept_exterior_coverage(
        voxel_map,qpos,current,future,DECISION_S,
        margin_m=.05,z_min_m=z_min,z_max_m=z_max)
    counts={label:sum(x["state"]==label for x in rows)
            for label in ("free","occupied","unknown")}
    assert all(counts[label]==reference[label] for label in counts)
    return [r for r in rows if r["state"]=="unknown"],reference


def main() -> None:
    v123=json.loads(V123.read_text())
    v98=json.loads((V98/"live-moving-owner-offer-audit.json").read_text())
    old_model=structural.structural_model
    structural.structural_model=lambda kind,side:add_person(
        old_model(kind,side))
    rows=[]
    try:
        for record in v123["rows"]:
            seed,case=record["seed"],record["case"]
            prior=next(r for r in v98["rows"] if
                       r["condition"]=="moving" and r["case"][0]==seed)
            source=ROOT/record["source_packet_path"]
            assert sha(source)==record["source_packet_sha256"]
            with np.load(source) as saved:
                qpos=saved["actor_qpos"].copy()
                estimated=saved["estimated_pose_xyyaw"].copy()
            room=OfferDriftRoom(case=tuple(prior["case"]),
                                pose_source="command",branch=OUT.parent)
            try:
                room.layout(seed,absent=False,furnished=True)
                obstacle=CASES[case]
                if obstacle is not None:
                    position,size=obstacle
                    room.model.body_pos[room.model.body("loop_box").id]=position
                    room.model.geom("loop_box_geom").size[:]=size
                room.data.qpos[:]=qpos
                room.actual[:]=(qpos[0],qpos[1],
                                2*math.atan2(qpos[6],qpos[3]))
                room.estimated[:]=estimated
                room.head=1.4
                room.time_s=DECISION_S
                set_person_pose(room.model,START_HAND_XYZ_M,
                                carried_item="loop_TomatoSauce")
                room.obstacle_xy[3]=(0.,.306)
                room.model.body_pos[room.model.body(
                    "loop_stool").id]=(0.,.306,STOOL_SIZE[2])
                room._forward()
                voxel_map=DepthVoxelMap()
                fovy=float(room.model.cam_fovy[
                    room.model.camera("lower_rgb").id])
                old_head=ROOT/record["old_head_path"]
                assert sha(old_head)==record["old_head_sha256"]
                with np.load(old_head) as packet:
                    voxel_map.observe(packet["filtered_depth_m"],
                                      (packet["camera_origin_estimated_m"],
                                       packet["camera_rotation_estimated"]),
                                      START_S,fovy_deg=fovy,stride=4,
                                      max_depth_m=1.4,self_masked=True)
                new_meta=record["head_packets"]["1.4"]["lower_rgb"]
                assert sha(ROOT/new_meta["path"])==new_meta["sha256"]
                with np.load(ROOT/new_meta["path"]) as packet:
                    voxel_map.observe(packet["filtered_depth_m"],
                                      (packet["camera_origin_estimated_m"],
                                       packet["camera_rotation_estimated"]),
                                      DECISION_S,fovy_deg=new_meta[
                                          "fovy_deg"],stride=4,
                                      max_depth_m=1.4,self_masked=True)
                ray_path=ROOT/record["range_packet_path"]
                assert sha(ray_path)==record["range_packet_sha256"]
                rays=json.loads(ray_path.read_text())["ray_packets"]
                for name in RAY_NAMES:
                    assert all(ray["capture_time_s"]==DECISION_S
                               for ray in rays[name])
                    sensor.add_packet(voxel_map,rays[name],DECISION_S)
                voxel_map=freshness_filtered(
                    voxel_map,DECISION_S,ROUTE_FREE_AGE_S)
                robot=RobotGeometry(room.model)
                bands={}
                for band,lo,hi in (("lower",.02,.18),("upper",.18,.43)):
                    unknown,coverage=unknown_shell_at(
                        robot,voxel_map,qpos,estimated,lo,hi)
                    reported=record["alternatives"][LAYOUT]["routes"][
                        "back2_diag2"]["pose_samples"]["estimated"][
                            "checked_steps"][0][band]
                    assert coverage==reported
                    items=[]
                    for x in unknown:
                        target=actual_point(x["sample"],estimated,
                                            room.actual)
                        sight={name:line_of_sight(room,target,mount)["status"]
                               for name,mount in CANDIDATE_MOUNTS.items()}
                        items.append({
                            "voxel_cell":[int(v) for v in x["cell"]],
                            "actor_sample_xyz_m":np.round(
                                x["sample"],5).tolist(),
                            "evaluator_reprojected_xyz_m":np.round(
                                target,5).tolist(),
                            "mount_sample_los_status":sight})
                    bands[band]={"coverage":coverage,
                                 "unknown_samples":items,
                                 "candidate_mount_clear_sample_counts":{
                                     name:sum(i["mount_sample_los_status"][
                                         name]=="sample_clear_los"
                                         for i in items)
                                     for name in CANDIDATE_MOUNTS}}
                rows.append({"seed":seed,"case":case,
                             "source_packet_path":record[
                                 "source_packet_path"],
                             "bands":bands})
                print(seed,case,{k:len(v["unknown_samples"])
                                 for k,v in bands.items()},flush=True)
            finally:
                room.close()
    finally:
        structural.structural_model=old_model
    report={
        "schema":"robot-learning-lab.ape-active-pan-unknown-cells.v1",
        "source_sha256":sha(SOURCE),
        "v123_report_sha256":sha(V123),
        "selected_visual_sha256":sha(selected_visual()),
        "layout":LAYOUT,
        "scope":"Exact first-back-step sampled shell cells left unknown after V123p's +1.4 lower RGB-D look and dense inward diagnostic pair. LOS candidate status uses evaluator actual pose and exact points, not whole-voxel coverage, actor evidence or a hardware recommendation. No route is authorized.",
        "rows":rows}
    OUT.write_text(json.dumps(report,indent=2)+"\n")
    print(OUT)


if __name__=="__main__":
    main()
