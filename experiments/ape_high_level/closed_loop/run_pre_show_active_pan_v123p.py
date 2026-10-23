"""Test one pre-show head look against actor meeting-route unknown space.

The ape stays at V98's saved 16.2s pose. A 0.4s head turn and fresh mounted
near-range packets precede a route decision at 16.6s. This checks whether an
active look makes a meeting route observable; it does not execute that route.
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
from run import H, ROOT, W, sha
from run_articulated_owner_low_drift_v92 import OfferDriftRoom
from run_compact_near_sensors_v118 import PROXY_MOUNTS
from run_inward_low_mount_packets_v117 import BASE, NEW_MOUNTS
from run_moving_low_clutter_v302 import STOOL_SIZE
from run_owner_exit_height_v96 import CASES
from run_pre_show_reserve_screen_v100 import SEQUENCES, START_S, V98
from run_pre_show_sensor_coverage_v102 import (FOV_H_DEG, FOV_V_DEG,
                                                GRID_H, GRID_V, MOUNTS)
from self_depth_mask import SelfDepthMask


SOURCE = Path(__file__)
OUT = ROOT / "artifacts/ape-closed-loop-cove50-active-pan"
V109 = ROOT / "artifacts/ape-closed-loop-cove50-v109"
DECISION_S = 16.6
HEAD_LOOKS_RAD = (0.,-1.4,1.4)
RANGE_LAYOUTS = {
    "outward": BASE,
    "outward_plus_inward_tof8": (*BASE,"left_tof8_level",
                                 "right_tof8_level"),
    "outward_plus_short_stereo": (*BASE,"left_short_stereo_proxy",
                                   "right_short_stereo_proxy"),
    "outward_plus_dense_inward_diagnostic": (
        *BASE,"left_low_inward","right_low_inward"),
}
CANDIDATES = {name:commands for name,commands in SEQUENCES.items()
              if commands}


def capture_head(room: OfferDriftRoom, seed: int, case: str,
                 angle: float, camera: str) -> dict:
    room.head = angle
    room.time_s = DECISION_S
    room._forward()
    rgb,depth = room.capture(camera,depth=True)
    mask = SelfDepthMask(room.model,camera=camera,
                         width=W,height=H,agreement_m=.04)
    try:
        filtered,stats = mask.apply(depth,room.data.qpos.copy(),
                                    max_depth_m=1.4)
    finally:
        mask.close()
    cam_pose = room.camera_pose_estimate(camera)
    fovy = float(room.model.cam_fovy[room.model.camera(camera).id])
    branch = OUT/f"seed{seed}-{case}"
    branch.mkdir(parents=True,exist_ok=True)
    path = branch/f"{camera}-yaw{angle:+.1f}.npz"
    np.savez_compressed(path,rgb=np.asarray(rgb),
                        filtered_depth_m=filtered,
                        camera_origin_estimated_m=cam_pose[0],
                        camera_rotation_estimated=cam_pose[1],
                        actor_qpos=room.data.qpos.copy(),
                        estimated_pose_xyyaw=room.estimated.copy(),
                        capture_time_s=DECISION_S)
    return {"path":str(path.relative_to(ROOT)),"sha256":sha(path),
            "fovy_deg":fovy,"mask_stats":stats,
            "valid_depth_pixels":int(np.isfinite(filtered).sum())}


def add_head(voxel_map: DepthVoxelMap, meta: dict) -> None:
    with np.load(ROOT/meta["path"]) as packet:
        voxel_map.observe(packet["filtered_depth_m"],
                          (packet["camera_origin_estimated_m"],
                           packet["camera_rotation_estimated"]),
                          float(packet["capture_time_s"]),
                          fovy_deg=meta["fovy_deg"],stride=4,
                          max_depth_m=1.4,self_masked=True)


def scan_compact_now(room: OfferDriftRoom, pose: np.ndarray,
                     seed: int, name: str) -> tuple[list[dict],dict]:
    if "tof8" in name:
        centers=(np.arange(8)+.5)*45./8.-22.5
        sensor.HORIZONTAL_DEG=centers
        sensor.VERTICAL_DEG=centers
        sensor.RANGE_MAX_M=.60
        sensor.RANGE_NOISE_STD_M=.012
    else:
        sensor.HORIZONTAL_DEG=(np.arange(32)+.5)*87./32.-43.5
        sensor.VERTICAL_DEG=(np.arange(24)+.5)*58./24.-29.
        sensor.RANGE_MAX_M=.50
        sensor.RANGE_NOISE_STD_M=.01
    sensor.DROPOUT_FRACTION=.15
    rays,counts=sensor.range_packet(room,pose,name,seed,DECISION_S)
    if "stereo" in name:
        rays=[ray for ray in rays if ray["measured_range_m"]>=.07]
        counts["valid_after_7cm_min"]=len(rays)
    assert all(ray["capture_time_s"]==DECISION_S for ray in rays)
    return rays,counts


def main() -> None:
    OUT.mkdir(parents=True,exist_ok=True)
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
    sensor.MOUNTS.update(NEW_MOUNTS)
    sensor.MOUNTS.update(PROXY_MOUNTS)
    rows=[]
    try:
        for seed in (170,172):
            prior=next(r for r in v98["rows"] if
                       r["condition"]=="moving" and r["case"][0]==seed)
            source=(V98/"moving"/f"seed{seed}-present"/
                    f"seed{seed}-present/owner-guard-00.npz")
            with np.load(source) as packet:
                qpos=packet["actor_qpos"].copy()
                estimated=packet["estimated_pose_xyyaw"].copy()
            for case in ("clear","hand_close"):
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
                    room.time_s=DECISION_S
                    set_person_pose(room.model,START_HAND_XYZ_M,
                                    carried_item="loop_TomatoSauce")
                    room.obstacle_xy[3]=(0.,.306)
                    room.model.body_pos[room.model.body(
                        "loop_stool").id]=(0.,.306,STOOL_SIZE[2])
                    room.head=0.
                    room._forward()
                    heads={}
                    for angle in HEAD_LOOKS_RAD:
                        heads[str(angle)]={camera:capture_head(
                            room,seed,case,angle,camera)
                            for camera in ("lower_rgb","upper_rgb")}
                    room.head=0.
                    room._forward()
                    rays={}
                    counts={}
                    sensor.HORIZONTAL_DEG=np.linspace(
                        -FOV_H_DEG/2,FOV_H_DEG/2,GRID_H)
                    sensor.VERTICAL_DEG=np.linspace(
                        -FOV_V_DEG/2,FOV_V_DEG/2,GRID_V)
                    sensor.RANGE_MAX_M=.60
                    sensor.RANGE_NOISE_STD_M=.005
                    sensor.DROPOUT_FRACTION=.15
                    for name in (*BASE,"left_low_inward",
                                 "right_low_inward"):
                        rays[name],counts[name]=sensor.range_packet(
                            room,estimated,name,seed,DECISION_S)
                    for name in ("left_tof8_level","right_tof8_level",
                                 "left_short_stereo_proxy",
                                 "right_short_stereo_proxy"):
                        rays[name],counts[name]=scan_compact_now(
                            room,estimated,seed,name)
                    branch=OUT/f"seed{seed}-{case}"
                    ray_path=branch/"current-active-pan-rays.json"
                    ray_path.write_text(json.dumps({
                        "capture_time_s":DECISION_S,
                        "estimated_pose_xyyaw":estimated.tolist(),
                        "ray_packets":rays,
                        "counts":counts},indent=2)+"\n")
                    old_head=V109/f"seed{seed}-{case}/current-head-rgbd.npz"
                    old_fovy=float(room.model.cam_fovy[
                        room.model.camera("lower_rgb").id])
                    robot=RobotGeometry(room.model)
                    alternatives={}
                    with np.load(old_head) as old:
                        old_depth=old["filtered_depth_m"].copy()
                        old_camera=(
                            old["camera_origin_estimated_m"].copy(),
                            old["camera_rotation_estimated"].copy())
                    old_map=DepthVoxelMap()
                    old_map.observe(old_depth,old_camera,START_S,
                                    fovy_deg=old_fovy,stride=4,
                                    max_depth_m=1.4,self_masked=True)
                    for angle in HEAD_LOOKS_RAD:
                        low_map=DepthVoxelMap()
                        low_map.free=old_map.free.copy()
                        low_map.occupied=old_map.occupied.copy()
                        low_map.last_time_s=old_map.last_time_s
                        add_head(low_map,heads[str(angle)]["lower_rgb"])
                        for upper_depth in (False,True):
                            head_map=DepthVoxelMap()
                            head_map.free=low_map.free.copy()
                            head_map.occupied=low_map.occupied.copy()
                            head_map.last_time_s=low_map.last_time_s
                            if upper_depth:
                                add_head(head_map,heads[str(angle)][
                                    "upper_rgb"])
                            for layout,names in RANGE_LAYOUTS.items():
                                voxel_map=DepthVoxelMap()
                                voxel_map.free=head_map.free.copy()
                                voxel_map.occupied=head_map.occupied.copy()
                                voxel_map.last_time_s=head_map.last_time_s
                                for name in names:
                                    sensor.add_packet(voxel_map,rays[name],
                                                      DECISION_S)
                                fresh=freshness_filtered(
                                    voxel_map,DECISION_S,ROUTE_FREE_AGE_S)
                                quick=route_screen(
                                    robot,fresh,qpos,estimated,
                                    CANDIDATES,DECISION_S,
                                    pose_offsets={"estimated":(0.,0.,0.)})
                                possible=any(r["coverage_status"]==
                                             "observed_free"
                                             for r in quick.values())
                                routes=(route_screen(
                                    robot,fresh,qpos,estimated,
                                    CANDIDATES,DECISION_S) if possible
                                    else quick)
                                key=f"yaw{angle:+.1f}-{'dual_depth' if upper_depth else 'lower_depth'}-{layout}"
                                alternatives[key]={
                                    "head_yaw_rad":angle,
                                    "upper_depth_diagnostic":upper_depth,
                                    "range_layout":layout,
                                    "pose_screen_scope":(
                                        "all_five_offsets" if possible else
                                        "estimated_only_rejects_all"),
                                    "motion_fresh_free_cells":len(fresh.free),
                                    "occupied_cells":len(fresh.occupied),
                                    "routes":routes,
                                    "map_supported_routes":[n for n,r in
                                        routes.items() if r[
                                            "coverage_status"]=="observed_free"],
                                }
                    rows.append({
                        "seed":seed,"case":case,
                        "source_packet_path":str(source.relative_to(ROOT)),
                        "source_packet_sha256":sha(source),
                        "old_head_path":str(old_head.relative_to(ROOT)),
                        "old_head_sha256":sha(old_head),
                        "head_packets":heads,
                        "range_packet_path":str(ray_path.relative_to(ROOT)),
                        "range_packet_sha256":sha(ray_path),
                        "alternatives":alternatives,
                    })
                    print(seed,case,"supported",
                          {key:v["map_supported_routes"] for key,v in
                           alternatives.items() if v[
                               "map_supported_routes"]},flush=True)
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
        "schema":"robot-learning-lab.ape-pre-show-active-pan.v1",
        "source_sha256":sha(SOURCE),
        "v98_report_sha256":sha(V98/"live-moving-owner-offer-audit.json"),
        "selected_visual_sha256":sha(selected_visual()),
        "decision_time_s":DECISION_S,
        "head_yaw_candidates_rad":HEAD_LOOKS_RAD,
        "range_layouts":RANGE_LAYOUTS,
        "scope":"Frozen V98 16.2s historical pose, one 0.4s pre-show head yaw candidate followed by fresh lower RGB-D and optional hypothetical upper depth plus current noisy/dropout outward/inward range at16.6s. Actor map screens five fixed routes with five pose-offset samples and 0.4s free age; evaluator geometry is not used for route choice. Head slews/optical distortions and selected-body motion are not executed; upper depth is a diagnostic virtual sensor. No actor-authorized route without pose bound, dynamic reserve, physical mounts or qualified skill.",
        "rows":rows,
    }
    path=OUT/"report.json"
    path.write_text(json.dumps(report,indent=2)+"\n")
    print(path)


if __name__=="__main__":
    main()
