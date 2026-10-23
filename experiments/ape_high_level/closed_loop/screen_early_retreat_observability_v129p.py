"""V129p: ask whether the early retreat is supported by actor sensor space.

The V128p meeting stations are useful only if a swept route can be certified
from fresh observations. This is a coverage screen, not a motion command.
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
from pre_show_actor_planner import (POSE_OFFSETS, ROUTE_FREE_AGE_S,
                                    freshness_filtered)
from robot_geometry import RobotGeometry
from run import H, ROOT, W, sha
from run_articulated_owner_low_drift_v92 import OfferDriftRoom
from run_compact_near_sensors_v118 import PROXY_MOUNTS
from run_inward_low_mount_packets_v117 import NEW_MOUNTS
from run_moving_low_clutter_v302 import STOOL_SIZE
from run_pre_show_sensor_coverage_v102 import (FOV_H_DEG, FOV_V_DEG,
                                                GRID_H, GRID_V, MOUNTS)
from run_pre_show_lidar_rings_v125p import RING_MOUNTS
from self_depth_mask import SelfDepthMask
import validate_early_meeting_v128p as v128p


SOURCE = Path(__file__)
OUT = ROOT / "artifacts/ape-closed-loop-cove50-early-retreat-observability"
V98 = v128p.V98
DECISION_S = 8.0
HEAD_LOOKS = (0., 1.4)
OUTWARD = ("rear_low_near", "rear_mid_near", "right_upper")
DENSE = ("left_low_inward", "right_low_inward")
RINGS = ("left_low_three_plane", "right_low_three_plane",
         "rear_high_three_plane")
LAYOUTS = {
    "camera_only": (),
    "outward": OUTWARD,
    "outward_dense": (*OUTWARD, *DENSE),
    "outward_3d_rings": (*OUTWARD, *RINGS),
    "outward_dense_3d_rings": (*OUTWARD, *DENSE, *RINGS),
}


def capture_head(room: OfferDriftRoom, seed: int, angle: float) -> dict:
    room.head = angle
    room.time_s = DECISION_S
    room._forward()
    rgb, depth = room.capture("lower_rgb", depth=True)
    mask = SelfDepthMask(room.model, camera="lower_rgb", width=W, height=H,
                         agreement_m=.04)
    try:
        filtered, stats = mask.apply(depth, room.data.qpos.copy(),
                                     max_depth_m=1.4)
    finally:
        mask.close()
    pose = room.camera_pose_estimate("lower_rgb")
    branch = OUT / f"seed{seed}"
    branch.mkdir(parents=True, exist_ok=True)
    path = branch / f"lower-yaw{angle:+.1f}.npz"
    np.savez_compressed(path, rgb=np.asarray(rgb),
                        filtered_depth_m=filtered,
                        camera_origin_estimated_m=pose[0],
                        camera_rotation_estimated=pose[1],
                        actor_qpos=room.data.qpos.copy(),
                        estimated_pose_xyyaw=room.estimated.copy(),
                        capture_time_s=DECISION_S)
    return {"path": str(path.relative_to(ROOT)), "sha256": sha(path),
            "fovy_deg": float(room.model.cam_fovy[
                room.model.camera("lower_rgb").id]),
            "mask_stats": stats,
            "valid_depth_pixels": int(np.isfinite(filtered).sum())}


def screen(robot: RobotGeometry, voxel_map: DepthVoxelMap,
           qpos: np.ndarray, pose: np.ndarray,
           offset_xy: tuple[float, float],
           pose_offsets: dict[str, tuple[float, float, float]]) -> dict:
    variants = {}
    for posture in ("neutral", "left_raised"):
        posture_qpos = qpos.copy()
        if posture == "left_raised":
            posture_qpos[robot.model.joint(
                "left_shoulder_pitch").qposadr[0]] = -.6
        for name, perturbation in pose_offsets.items():
            current = pose + np.asarray(perturbation)
            future = current.copy()
            future[:2] += offset_xy
            heights = {}
            for height, z_min, z_max in (("lower", .02, .18),
                                         ("upper", .18, .43)):
                heights[height] = robot.swept_exterior_coverage(
                    voxel_map, posture_qpos, current, future, DECISION_S,
                    margin_m=.05, z_min_m=z_min, z_max_m=z_max)
            variants[f"{posture}:{name}"] = heights
    statuses = [entry[h]["status"] for entry in variants.values()
                for h in ("upper", "lower")]
    return {"coverage_status": (
        "observed_free" if all(x == "free" for x in statuses) else
        "occupied" if "occupied" in statuses else "unknown"),
        "pose_posture_samples": variants}


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    v98 = json.loads((V98 / "live-moving-owner-offer-audit.json").read_text())
    original_model = structural.structural_model
    original_mounts = sensor.MOUNTS.copy()
    original_h = sensor.HORIZONTAL_DEG.copy()
    original_v = sensor.VERTICAL_DEG.copy()
    original_range = sensor.RANGE_MAX_M
    original_noise = sensor.RANGE_NOISE_STD_M
    original_dropout = sensor.DROPOUT_FRACTION
    structural.structural_model = lambda kind, side: add_person(
        original_model(kind, side))
    sensor.MOUNTS.clear()
    sensor.MOUNTS.update(MOUNTS)
    sensor.MOUNTS.update(NEW_MOUNTS)
    sensor.MOUNTS.update(PROXY_MOUNTS)
    sensor.MOUNTS.update(RING_MOUNTS)
    rows = []
    try:
        for seed in v128p.SEEDS:
            prior = next(row for row in v98["rows"]
                         if row["condition"] == "moving" and
                         row["case"][0] == seed)
            source = (V98 / "moving" / f"seed{seed}-present" /
                      f"seed{seed}-present/side-probe-00.npz")
            with np.load(source) as packet:
                qpos = packet["actor_qpos"].copy()
                estimate = packet["estimated_pose_xyyaw"].copy()
            room = OfferDriftRoom(case=tuple(prior["case"]),
                                  pose_source="command", branch=OUT)
            try:
                room.layout(seed, absent=False, furnished=True)
                room.data.qpos[:] = qpos
                room.actual[:] = (qpos[0], qpos[1],
                                  2*math.atan2(qpos[6], qpos[3]))
                room.estimated[:] = estimate
                room.time_s = DECISION_S
                set_person_pose(room.model, START_HAND_XYZ_M,
                                carried_item="loop_TomatoSauce")
                room.obstacle_xy[3] = (0., .306)
                room.model.body_pos[room.model.body(
                    "loop_stool").id] = (0., .306, STOOL_SIZE[2])
                room._forward()
                head = {angle: capture_head(room, seed, angle)
                        for angle in HEAD_LOOKS}
                room.head = 1.4
                room._forward()
                rays = {}
                counts = {}
                sensor.RANGE_MAX_M = .60
                sensor.RANGE_NOISE_STD_M = .005
                sensor.DROPOUT_FRACTION = .15
                sensor.HORIZONTAL_DEG = np.linspace(
                    -FOV_H_DEG/2, FOV_H_DEG/2, GRID_H)
                sensor.VERTICAL_DEG = np.linspace(
                    -FOV_V_DEG/2, FOV_V_DEG/2, GRID_V)
                for name in (*OUTWARD, *DENSE):
                    rays[name], counts[name] = sensor.range_packet(
                        room, estimate, name, seed, DECISION_S)
                sensor.HORIZONTAL_DEG = np.arange(-180., 180., 2.)
                sensor.VERTICAL_DEG = np.array((-15., 0., 15.))
                for name in RINGS:
                    rays[name], counts[name] = sensor.range_packet(
                        room, estimate, name, seed, DECISION_S)
                ray_path = OUT / f"seed{seed}/range-packets.json"
                ray_path.write_text(json.dumps({"capture_time_s": DECISION_S,
                                                "ray_packets": rays,
                                                "counts": counts}, indent=2)+"\n")
                robot = RobotGeometry(room.model)
                alternatives = {}
                for angle, meta in head.items():
                    with np.load(ROOT / meta["path"]) as packet:
                        base = DepthVoxelMap()
                        base.observe(packet["filtered_depth_m"],
                                     (packet["camera_origin_estimated_m"],
                                      packet["camera_rotation_estimated"]),
                                     DECISION_S, fovy_deg=meta["fovy_deg"],
                                     stride=4, max_depth_m=1.4,
                                     self_masked=True)
                    for layout, mounts in LAYOUTS.items():
                        current = DepthVoxelMap()
                        current.free = base.free.copy()
                        current.occupied = base.occupied.copy()
                        current.last_time_s = base.last_time_s
                        for name in mounts:
                            sensor.add_packet(current, rays[name], DECISION_S)
                        fresh = freshness_filtered(
                            current, DECISION_S, ROUTE_FREE_AGE_S)
                        routes = {}
                        for station, offset in v128p.STATIONS.items():
                            quick = screen(robot, fresh, qpos, estimate,
                                           offset, {"estimated": (0., 0., 0.)})
                            routes[station] = (screen(robot, fresh, qpos,
                                estimate, offset, POSE_OFFSETS)
                                if quick["coverage_status"] == "observed_free"
                                else quick)
                        alternatives[f"yaw{angle:+.1f}-{layout}"] = {
                            "head_yaw_rad": angle, "layout": layout,
                            "fresh_free_cells": len(fresh.free),
                            "occupied_cells": len(fresh.occupied),
                            "routes": routes}
                row = {"seed": seed,
                       "source_packet_path": str(source.relative_to(ROOT)),
                       "source_packet_sha256": sha(source),
                       "head_packets": {str(k): v for k, v in head.items()},
                       "range_packet_path": str(ray_path.relative_to(ROOT)),
                       "range_packet_sha256": sha(ray_path),
                       "range_counts": counts,
                       "alternatives": alternatives}
                rows.append(row)
                supported = [(name, station)
                             for name, alternative in alternatives.items()
                             for station, route in alternative["routes"].items()
                             if route["coverage_status"] == "observed_free"]
                print(seed, "supported", supported, flush=True)
            finally:
                room.close()
    finally:
        sensor.MOUNTS.clear()
        sensor.MOUNTS.update(original_mounts)
        sensor.HORIZONTAL_DEG = original_h
        sensor.VERTICAL_DEG = original_v
        sensor.RANGE_MAX_M = original_range
        sensor.RANGE_NOISE_STD_M = original_noise
        sensor.DROPOUT_FRACTION = original_dropout
        structural.structural_model = original_model
    report = {
        "schema": "robot-learning-lab.ape-early-retreat-observability.v1",
        "source_sha256": sha(SOURCE),
        "v128_report_sha256": sha(v128p.OUT / "report.json"),
        "selected_visual_sha256": sha(selected_visual()),
        "scope": "At the saved7.6s staging pose in the furnished historical "
        "Cove-50 scene, simulated fresh self-masked lower RGB-D at8.0s, "
        "noisy near-range packets, and virtual three-elevation all-around "
        "rings test direct swept12cm and16cm retreat bands. Both neutral "
        "and raised-left-arm solids are sampled with5cm outward margin. "
        "Fresh unknown remains unknown. This is actor-map coverage only; "
        "mount packaging, metric pose error, skill execution and dynamic "
        "obstacle prediction are not qualified.",
        "rows": rows,
    }
    (OUT / "report.json").write_text(json.dumps(report, indent=2)+"\n")
    print(OUT / "report.json")


if __name__ == "__main__":
    main()
