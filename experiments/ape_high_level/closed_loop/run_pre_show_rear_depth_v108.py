"""V108: compare hypothetical high rear depth views at the pre-show pose.

Dense ray grids stand in for a rear depth sensor and expose self-occlusion.
Their free-space contribution is tested against V102's unchanged head/rear
pair and generous full ring. This does not select a physical camera.
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
from robot_geometry import RobotGeometry
from run import H, ROOT, W, sha
from run_articulated_owner_low_drift_v92 import OfferDriftRoom
from run_moving_low_clutter_v302 import STOOL_SIZE
from run_owner_exit_height_v96 import CASES
from run_pre_show_reserve_screen_v100 import (COMMAND_STEP_M, DIRECTIONS,
                                              SEQUENCES, START_S, V98)
from run_pre_show_sensor_coverage_v102 import LAYOUTS, MOUNTS
from self_depth_mask import SelfDepthMask


SOURCE = Path(__file__)
OUT = ROOT / "artifacts/ape-closed-loop-cove50-v108"
V102 = ROOT / "artifacts/ape-closed-loop-cove50-v102"
HIGH_MOUNTS = {
    "rear_high_mid": ((-.06, 0., .35), math.pi),
    "rear_crown": ((-.03, 0., .45), math.pi),
}
GRID_H_V = (81, 61)
FOV_H_V_DEG = (120., 100.)
LAYOUTS_V108 = {
    "head_rear_low_mid_near": LAYOUTS["head_rear_low_mid_near"],
    "head_rear_low_mid_near_high": (*LAYOUTS["head_rear_low_mid_near"],
                                     "rear_high_mid"),
    "head_rear_low_mid_near_crown": (*LAYOUTS["head_rear_low_mid_near"],
                                      "rear_crown"),
    "head_full_ring_low_upper": LAYOUTS["head_full_ring_low_upper"],
    "head_full_ring_low_upper_high": (*LAYOUTS["head_full_ring_low_upper"],
                                       "rear_high_mid"),
}


def covered_steps(robot: RobotGeometry, voxel_map: DepthVoxelMap,
                  qpos: np.ndarray, pose: np.ndarray) -> list[dict]:
    current = pose.copy()
    result = []
    for command in SEQUENCES["back2_diag2"]:
        future = current.copy()
        future[:2] += COMMAND_STEP_M * np.asarray(DIRECTIONS[command])
        result.append({
            "command": command,
            "upper": robot.swept_exterior_coverage(
                voxel_map, qpos, current, future, START_S,
                margin_m=.05, z_min_m=.18, z_max_m=.43),
            "lower": robot.swept_exterior_coverage(
                voxel_map, qpos, current, future, START_S,
                margin_m=.05, z_min_m=.02, z_max_m=.18),
        })
        current = future
    return result


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    v98 = json.loads((V98 / "live-moving-owner-offer-audit.json").read_text())
    v102 = json.loads((V102 / "report.json").read_text())
    old_model = structural.structural_model
    old_mounts = sensor.MOUNTS.copy()
    old_h = sensor.HORIZONTAL_DEG.copy()
    old_v = sensor.VERTICAL_DEG.copy()
    structural.structural_model = lambda kind, side: add_person(old_model(kind, side))
    sensor.MOUNTS.clear()
    sensor.MOUNTS.update(MOUNTS)
    sensor.MOUNTS.update(HIGH_MOUNTS)
    sensor.HORIZONTAL_DEG = np.linspace(-FOV_H_V_DEG[0]/2,
                                          FOV_H_V_DEG[0]/2, GRID_H_V[0])
    sensor.VERTICAL_DEG = np.linspace(-FOV_H_V_DEG[1]/2,
                                        FOV_H_V_DEG[1]/2, GRID_H_V[1])
    rows = []
    try:
        for seed in (170, 172):
            prior = next(row for row in v98["rows"]
                         if row["condition"] == "moving" and row["case"][0] == seed)
            for case in ("clear", "hand_close"):
                row102 = next(row for row in v102["rows"]
                              if row["seed"] == seed and row["case"] == case)
                old_packet_path = ROOT / row102["actor_packet_path"]
                if sha(old_packet_path) != row102["actor_packet_sha256"]:
                    raise ValueError("V102 actor packet changed")
                actor = json.loads(old_packet_path.read_text())
                qpos = np.asarray(actor["actor_qpos"], dtype=float)
                pose = np.asarray(actor["estimated_pose_xyyaw"], dtype=float)
                room = OfferDriftRoom(case=tuple(prior["case"]),
                                      pose_source="command", branch=OUT / "scene")
                try:
                    room.layout(seed, absent=False, furnished=True)
                    obstacle = CASES[case]
                    if obstacle is not None:
                        position, size = obstacle
                        room.model.body_pos[room.model.body("loop_box").id] = position
                        room.model.geom("loop_box_geom").size[:] = size
                    room.data.qpos[:] = qpos
                    room.actual[:] = (qpos[0], qpos[1],
                                      2*math.atan2(qpos[6], qpos[3]))
                    room.estimated[:] = pose
                    room.head = float(qpos[room.head_addr])
                    room.time_s = START_S
                    set_person_pose(room.model, START_HAND_XYZ_M,
                                    carried_item="loop_TomatoSauce")
                    room.obstacle_xy[3] = (0., .306)
                    room.model.body_pos[room.model.body("loop_stool").id] = (
                        0., .306, STOOL_SIZE[2])
                    room._forward()
                    robot = RobotGeometry(room.model)
                    robot.set_state(qpos)
                    own_solids = robot.primitives()
                    mount_gaps = {}
                    high_packets = {}
                    counts = {}
                    for name, (offset, _) in HIGH_MOUNTS.items():
                        world_origin = np.array([*room.actual[:2], 0.]) + (
                            sensor.rotate(float(room.actual[2]),
                                          np.asarray(offset)))
                        mount_gaps[name] = round(robot.solid_distance(
                            world_origin, own_solids), 5)
                        rays, counts[name] = sensor.range_packet(
                            room, pose, name, seed, START_S)
                        high_packets[name] = rays
                    _, depth = room.capture("lower_rgb", depth=True)
                    mask = SelfDepthMask(room.model, camera="lower_rgb",
                                         width=W, height=H, agreement_m=.04)
                    try:
                        filtered, mask_stats = mask.apply(
                            depth, room.data.qpos.copy(), max_depth_m=1.4)
                    finally:
                        mask.close()
                    fovy = float(room.model.cam_fovy[
                        room.model.camera("lower_rgb").id])
                    base = DepthVoxelMap()
                    base.observe(filtered, room.camera_pose_estimate(
                        "lower_rgb"), START_S, fovy_deg=fovy,
                        stride=4, max_depth_m=1.4, self_masked=True)
                    layouts = {}
                    for layout, names in LAYOUTS_V108.items():
                        voxel_map = DepthVoxelMap()
                        voxel_map.free = base.free.copy()
                        voxel_map.occupied = base.occupied.copy()
                        voxel_map.last_time_s = START_S
                        for name in names:
                            rays = (high_packets[name] if name in HIGH_MOUNTS
                                    else actor["ray_packets"][name])
                            sensor.add_packet(voxel_map, rays, START_S)
                        layouts[layout] = covered_steps(
                            robot, voxel_map, qpos, pose)
                    branch = OUT / f"seed{seed}-{case}"
                    branch.mkdir(parents=True, exist_ok=True)
                    packet_path = branch / "high-rear-actor-rays.json"
                    packet_path.write_text(json.dumps({
                        "capture_time_s": START_S,
                        "estimated_pose_xyyaw": pose.tolist(),
                        "hypothetical_mounts": HIGH_MOUNTS,
                        "ray_packets": high_packets,
                    }, indent=2) + "\n")
                    rows.append({
                        "seed": seed, "case": case,
                        "v102_actor_packet_path": row102["actor_packet_path"],
                        "v102_actor_packet_sha256": row102["actor_packet_sha256"],
                        "new_actor_packet_path": str(packet_path.relative_to(ROOT)),
                        "new_actor_packet_sha256": sha(packet_path),
                        "high_mount_to_historical_body_solid_gap_m": mount_gaps,
                        "high_ray_counts": counts,
                        "head_mask_stats": mask_stats,
                        "coverage": layouts,
                    })
                finally:
                    room.close()
    finally:
        structural.structural_model = old_model
        sensor.MOUNTS.clear()
        sensor.MOUNTS.update(old_mounts)
        sensor.HORIZONTAL_DEG = old_h
        sensor.VERTICAL_DEG = old_v
    report = {
        "schema": "robot-learning-lab.ape-pre-show-rear-depth.v1",
        "source_sha256": sha(SOURCE),
        "v98_report_sha256": sha(V98 / "live-moving-owner-offer-audit.json"),
        "v102_report_sha256": sha(V102 / "report.json"),
        "selected_visual_sha256": sha(selected_visual()),
        "hypothetical_mounts": HIGH_MOUNTS,
        "grid_h_v": GRID_H_V,
        "fov_deg_h_v": FOV_H_V_DEG,
        "range_m": sensor.RANGE_MAX_M,
        "noise_std_m": sensor.RANGE_NOISE_STD_M,
        "dropout_fraction": sensor.DROPOUT_FRACTION,
        "layouts": LAYOUTS_V108,
        "scope": "Hypothetical high rear 81x61 first-hit arrays with 120x100deg FOV, 0.6m max range, 5mm noise and 15% dropout, added to unchanged V102 head/rear pair and intentionally generous full ring in two seeds and two block cases. Records all actor rays, self-first counts and historical-body solid gaps. Dense ray grids approximate depth coverage but are not camera optical models or packageable placements. Swept-shell 4cm voxel query is diagnostic, not motion permission; scene and body remain historical.",
        "rows": rows,
    }
    path = OUT / "report.json"
    path.write_text(json.dumps(report, indent=2) + "\n")
    for row in rows:
        print(row["seed"], row["case"], row["high_ray_counts"], {
            name: (steps[0]["upper"]["unknown"],
                   steps[0]["lower"]["unknown"])
            for name, steps in row["coverage"].items()})
    print(path)


if __name__ == "__main__":
    main()
