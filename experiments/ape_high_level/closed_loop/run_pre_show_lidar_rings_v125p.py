"""Compare all-around near-body ray geometry on the blocked meeting route.

These are virtual 2-D and three-elevation lidar-style scans from historical
side/rear mount origins. They are not optical device models or a parts choice.
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
from pre_show_actor_planner import (ROUTE_FREE_AGE_S, freshness_filtered,
                                    route_screen)
from robot_geometry import RobotGeometry
from run import ROOT, sha
from run_articulated_owner_low_drift_v92 import OfferDriftRoom
from run_moving_low_clutter_v302 import STOOL_SIZE
from run_owner_exit_height_v96 import CASES
from run_pre_show_reserve_screen_v100 import SEQUENCES, START_S, V98
from run_pre_show_sensor_coverage_v102 import MOUNTS
from run_pre_show_active_pan_v123p import DECISION_S


SOURCE=Path(__file__)
OUT=ROOT/"artifacts/ape-closed-loop-cove50-lidar-rings"
V123=ROOT/"artifacts/ape-closed-loop-cove50-active-pan/report.json"
BASE_RAYS=("rear_low_near","rear_mid_near","right_upper")
DENSE_RAYS=("left_low_inward","right_low_inward")
RING_ORIGINS={
    "left_low":MOUNTS["left_low"][0],
    "right_low":MOUNTS["right_low"][0],
    "rear_high":(-.06,0.,.35),
}
RING_MOUNTS={f"{origin}_{mode}":(offset,0.)
             for origin,offset in RING_ORIGINS.items()
             for mode in ("one_plane","three_plane")}
RING_LAYOUTS={
    "none":(),
    "left_low_one_plane":("left_low_one_plane",),
    "both_low_one_plane":("left_low_one_plane","right_low_one_plane"),
    "all_one_plane":("left_low_one_plane","right_low_one_plane",
                     "rear_high_one_plane"),
    "left_low_three_plane":("left_low_three_plane",),
    "both_low_three_plane":("left_low_three_plane",
                            "right_low_three_plane"),
    "all_three_plane":("left_low_three_plane","right_low_three_plane",
                       "rear_high_three_plane"),
}
ROUTES={name:commands for name,commands in SEQUENCES.items() if commands}


def add_head(voxel_map: DepthVoxelMap, path: Path,
             time_s: float, fovy: float) -> None:
    with np.load(path) as packet:
        voxel_map.observe(packet["filtered_depth_m"],
                          (packet["camera_origin_estimated_m"],
                           packet["camera_rotation_estimated"]),
                          time_s,fovy_deg=fovy,stride=4,
                          max_depth_m=1.4,self_masked=True)


def main() -> None:
    OUT.mkdir(parents=True,exist_ok=True)
    v123=json.loads(V123.read_text())
    v98=json.loads((V98/"live-moving-owner-offer-audit.json").read_text())
    old_model=structural.structural_model
    old_mounts=sensor.MOUNTS.copy()
    old_h=sensor.HORIZONTAL_DEG.copy()
    old_v=sensor.VERTICAL_DEG.copy()
    old_range=sensor.RANGE_MAX_M
    old_noise=sensor.RANGE_NOISE_STD_M
    old_dropout=sensor.DROPOUT_FRACTION
    structural.structural_model=lambda kind,side:add_person(
        old_model(kind,side))
    sensor.MOUNTS.clear()
    sensor.MOUNTS.update(MOUNTS)
    sensor.MOUNTS.update(RING_MOUNTS)
    rows=[]
    try:
        for record in v123["rows"]:
            seed,case=record["seed"],record["case"]
            prior=next(r for r in v98["rows"] if
                       r["condition"]=="moving" and r["case"][0]==seed)
            source=ROOT/record["source_packet_path"]
            assert sha(source)==record["source_packet_sha256"]
            with np.load(source) as packet:
                qpos=packet["actor_qpos"].copy()
                estimated=packet["estimated_pose_xyyaw"].copy()
            room=OfferDriftRoom(case=tuple(prior["case"]),
                                pose_source="command",branch=OUT)
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
                robot=RobotGeometry(room.model)
                robot.set_state(qpos)
                solid=robot.primitives()
                mount_gaps={}
                for name,(offset,_) in RING_MOUNTS.items():
                    origin=np.array([*room.actual[:2],0.])+sensor.rotate(
                        float(room.actual[2]),np.asarray(offset))
                    mount_gaps[name]=round(robot.solid_distance(
                        origin,solid),5)
                sensor.HORIZONTAL_DEG=np.arange(-180.,180.,2.)
                sensor.RANGE_MAX_M=.60
                sensor.RANGE_NOISE_STD_M=.005
                sensor.DROPOUT_FRACTION=.15
                rays={}
                counts={}
                for name in RING_MOUNTS:
                    sensor.VERTICAL_DEG=(np.array((-15.,0.,15.))
                                         if name.endswith("three_plane") else
                                         np.array((0.,)))
                    rays[name],counts[name]=sensor.range_packet(
                        room,estimated,name,seed,DECISION_S)
                branch=OUT/f"seed{seed}-{case}"
                branch.mkdir(parents=True,exist_ok=True)
                path=branch/"lidar-style-rays.json"
                path.write_text(json.dumps({"capture_time_s":DECISION_S,
                                            "estimated_pose_xyyaw":estimated.tolist(),
                                            "mounts":RING_MOUNTS,
                                            "ray_packets":rays,
                                            "counts":counts},indent=2)+"\n")
                head_meta=record["head_packets"]["1.4"]["lower_rgb"]
                old_head=ROOT/record["old_head_path"]
                new_head=ROOT/head_meta["path"]
                assert sha(old_head)==record["old_head_sha256"]
                assert sha(new_head)==head_meta["sha256"]
                head_map=DepthVoxelMap()
                fovy=float(room.model.cam_fovy[
                    room.model.camera("lower_rgb").id])
                add_head(head_map,old_head,START_S,fovy)
                add_head(head_map,new_head,DECISION_S,head_meta["fovy_deg"])
                prior_rays=json.loads((ROOT/record[
                    "range_packet_path"]).read_text())["ray_packets"]
                layouts={}
                for dense in (False,True):
                    base_map=DepthVoxelMap()
                    base_map.free=head_map.free.copy()
                    base_map.occupied=head_map.occupied.copy()
                    base_map.last_time_s=head_map.last_time_s
                    for name in (*BASE_RAYS,*(DENSE_RAYS if dense else ())):
                        sensor.add_packet(base_map,prior_rays[name],
                                          DECISION_S)
                    for layout,names in RING_LAYOUTS.items():
                        voxel_map=DepthVoxelMap()
                        voxel_map.free=base_map.free.copy()
                        voxel_map.occupied=base_map.occupied.copy()
                        voxel_map.last_time_s=base_map.last_time_s
                        for name in names:
                            sensor.add_packet(voxel_map,rays[name],DECISION_S)
                        fresh=freshness_filtered(
                            voxel_map,DECISION_S,ROUTE_FREE_AGE_S)
                        quick=route_screen(
                            robot,fresh,qpos,estimated,ROUTES,DECISION_S,
                            pose_offsets={"estimated":(0.,0.,0.)})
                        possible=any(r["coverage_status"]=="observed_free"
                                     for r in quick.values())
                        screened=(route_screen(
                            robot,fresh,qpos,estimated,ROUTES,DECISION_S)
                            if possible else quick)
                        label=(("dense_inward" if dense else "outward")+
                               "_plus_"+layout)
                        layouts[label]={
                            "dense_inward_diagnostic":dense,
                            "ring_layout":layout,
                            "motion_fresh_free_cells":len(fresh.free),
                            "occupied_cells":len(fresh.occupied),
                            "pose_screen_scope":("all_five_offsets" if
                                                  possible else
                                                  "estimated_only_rejects_all"),
                            "routes":screened,
                            "map_supported_routes":[name for name,r in
                                screened.items() if r[
                                    "coverage_status"]=="observed_free"],
                        }
                for dense,key in ((False,"outward"),(True,"dense_inward")):
                    baseline=layouts[key+"_plus_none"]["routes"]
                    prior_key=("yaw+1.4-lower_depth-outward" if not dense
                               else "yaw+1.4-lower_depth-outward_plus_dense_inward_diagnostic")
                    previous=record["alternatives"][prior_key]["routes"]
                    assert baseline==previous
                rows.append({"seed":seed,"case":case,
                             "v123_range_packet_path":record[
                                 "range_packet_path"],
                             "v123_range_packet_sha256":record[
                                 "range_packet_sha256"],
                             "new_packet_path":str(path.relative_to(ROOT)),
                             "new_packet_sha256":sha(path),
                             "mount_origin_to_historical_solid_gap_m":mount_gaps,
                             "packet_counts":counts,
                             "layouts":layouts})
                print(seed,case,"supported",{name:v[
                    "map_supported_routes"] for name,v in layouts.items()
                    if v["map_supported_routes"]},flush=True)
            finally:
                room.close()
    finally:
        structural.structural_model=old_model
        sensor.MOUNTS.clear()
        sensor.MOUNTS.update(old_mounts)
        sensor.HORIZONTAL_DEG=old_h
        sensor.VERTICAL_DEG=old_v
        sensor.RANGE_MAX_M=old_range
        sensor.RANGE_NOISE_STD_M=old_noise
        sensor.DROPOUT_FRACTION=old_dropout
    report={
        "schema":"robot-learning-lab.ape-pre-show-lidar-ring-proxy.v1",
        "source_sha256":sha(SOURCE),
        "v123_report_sha256":sha(V123),
        "selected_visual_sha256":sha(selected_visual()),
        "decision_time_s":DECISION_S,
        "ring_mounts":RING_MOUNTS,
        "ring_layouts":RING_LAYOUTS,
        "azimuth_spacing_deg":2.,
        "vertical_elevations_deg":{"one_plane":[0.],
                                   "three_plane":[-15.,0.,15.]},
        "range_max_m":.60,"range_noise_std_m":.005,"dropout":.15,
        "scope":"Same V123p frozen pre-show pose and +1.4rad lower RGB-D look, plus hypothetical all-around 2deg rays from low side and rear-high origins at16.6s. One horizontal plane versus three elevations are geometry proxies, not a physical lidar model or packaged sensor. Actor map uses timestamped noisy first hits and estimated pose; evaluator scene creates rays only. Route remains unauthorized without known-free shell, pose/dynamic bound and selected-body skill.",
        "rows":rows}
    report_path=OUT/"report.json"
    report_path.write_text(json.dumps(report,indent=2)+"\n")
    print(report_path)


if __name__=="__main__":
    main()
