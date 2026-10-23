"""V102: actor-side observability of V100's back-then-diagonal reserve path.

Optimistic, hypothetical first-hit rear/right arrays are compared with the
current lower head depth at the pre-show pose. A single snapshot cannot prove
future dynamic clearance; this checks whether the candidate is even observed.
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
from self_depth_mask import SelfDepthMask


SOURCE = Path(__file__)
OUT = ROOT / "artifacts/ape-closed-loop-cove50-v102"
MOUNTS = {
    "rear_mid_near": ((-.07, 0., .23), math.pi),
    "rear_low_near": ((-.07, 0., .10), math.pi),
    "rear_mid_standoff": ((-.15, 0., .23), math.pi),
    "rear_low_standoff": ((-.15, 0., .10), math.pi),
    "right_upper": ((0., -.155, .23), -math.pi/2),
    "right_low": ((0., -.155, .10), -math.pi/2),
    "left_upper": ((0., .155, .23), math.pi/2),
    "left_low": ((0., .155, .10), math.pi/2),
    "front_upper": ((.14, 0., .23), 0.),
    "front_low": ((.12, 0., .10), 0.),
}
FOV_H_DEG = 110.
FOV_V_DEG = 80.
GRID_H = 31
GRID_V = 21
LAYOUTS = {
    "head_only": (),
    "head_rear_mid_near": ("rear_mid_near",),
    "head_rear_low_mid_near": ("rear_low_near", "rear_mid_near"),
    "head_rear_low_mid_near_right_upper":
        ("rear_low_near", "rear_mid_near", "right_upper"),
    "head_rear_low_mid_standoff":
        ("rear_low_standoff", "rear_mid_standoff"),
    "head_near_rear_both_side_upper":
        ("rear_low_near", "rear_mid_near", "left_upper", "right_upper"),
    "head_full_ring_low_upper":
        ("rear_low_near", "rear_mid_near", "right_upper", "right_low",
         "left_upper", "left_low", "front_upper", "front_low"),
}


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    v98 = json.loads((V98 / "live-moving-owner-offer-audit.json").read_text())
    old_model = structural.structural_model
    old_mounts = sensor.MOUNTS.copy()
    old_h = sensor.HORIZONTAL_DEG.copy()
    old_v = sensor.VERTICAL_DEG.copy()
    structural.structural_model = lambda kind, side: add_person(
        old_model(kind, side))
    sensor.MOUNTS.clear()
    sensor.MOUNTS.update(MOUNTS)
    sensor.HORIZONTAL_DEG = np.linspace(-FOV_H_DEG/2, FOV_H_DEG/2, GRID_H)
    sensor.VERTICAL_DEG = np.linspace(-FOV_V_DEG/2, FOV_V_DEG/2, GRID_V)
    rows = []
    try:
        for seed in (170, 172):
            prior = next(row for row in v98["rows"]
                         if row["condition"] == "moving" and
                         row["case"][0] == seed)
            source = (V98 / "moving" / f"seed{seed}-present" /
                      f"seed{seed}-present/owner-guard-00.npz")
            with np.load(source) as packet:
                qpos = packet["actor_qpos"].copy()
                pose = packet["estimated_pose_xyyaw"].copy()
            for case in ("clear", "hand_close"):
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
                    room.model.body_pos[room.model.body(
                        "loop_stool").id] = (0., .306, STOOL_SIZE[2])
                    room._forward()
                    robot = RobotGeometry(room.model)
                    robot.set_state(qpos)
                    own_primitives = robot.primitives()
                    mount_solid_gaps = {}
                    for name, (offset, _) in MOUNTS.items():
                        actual_origin = (np.array([*room.actual[:2], 0.]) +
                                         sensor.rotate(float(room.actual[2]),
                                                       np.asarray(offset)))
                        mount_solid_gaps[name] = round(
                            robot.solid_distance(actual_origin,
                                                 own_primitives), 5)
                    rgb, depth = room.capture("lower_rgb", depth=True)
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
                    packets = {}
                    packet_stats = {}
                    for name in MOUNTS:
                        rays, counts = sensor.range_packet(
                            room, pose, name, seed, START_S)
                        packets[name] = rays
                        packet_stats[name] = counts
                    sequence = SEQUENCES["back2_diag2"]
                    coverage = {}
                    for layout, names in LAYOUTS.items():
                        voxel_map = DepthVoxelMap()
                        voxel_map.free = base.free.copy()
                        voxel_map.occupied = base.occupied.copy()
                        voxel_map.last_time_s = START_S
                        for name in names:
                            sensor.add_packet(voxel_map, packets[name], START_S)
                        steps = []
                        current = pose.copy()
                        for command in sequence:
                            future = current.copy()
                            future[:2] += COMMAND_STEP_M*np.asarray(
                                DIRECTIONS[command])
                            steps.append({
                                "command": command,
                                "upper_shell": robot.swept_exterior_coverage(
                                    voxel_map, qpos, current, future, START_S,
                                    margin_m=.05, z_min_m=.18, z_max_m=.43),
                                "lower_shell": robot.swept_exterior_coverage(
                                    voxel_map, qpos, current, future, START_S,
                                    margin_m=.05, z_min_m=.02, z_max_m=.18),
                            })
                            current = future
                        coverage[layout] = steps
                    branch = OUT / f"seed{seed}-{case}"
                    branch.mkdir(parents=True, exist_ok=True)
                    packet_path = branch / "actor-observation.json"
                    packet_path.write_text(json.dumps({
                        "capture_time_s": START_S,
                        "estimated_pose_xyyaw": pose.tolist(),
                        "actor_qpos": qpos.tolist(),
                        "head_depth_mask_stats": mask_stats,
                        "head_free_cells": len(base.free),
                        "head_occupied_cells": len(base.occupied),
                        "mounts": MOUNTS,
                        "mount_origin_to_historical_body_solid_gap_m":
                            mount_solid_gaps,
                        "ray_packets": packets,
                    }, indent=2) + "\n")
                    rows.append({"seed": seed, "case": case,
                                 "source_packet_path": str(source.relative_to(ROOT)),
                                 "source_packet_sha256": sha(source),
                                 "actor_packet_path": str(packet_path.relative_to(ROOT)),
                                 "actor_packet_sha256": sha(packet_path),
                                 "mount_counts": packet_stats,
                                 "mount_origin_to_historical_body_solid_gap_m":
                                     mount_solid_gaps,
                                 "head_mask_stats": mask_stats,
                                 "coverage": coverage})
                finally:
                    room.close()
    finally:
        structural.structural_model = old_model
        sensor.MOUNTS.clear()
        sensor.MOUNTS.update(old_mounts)
        sensor.HORIZONTAL_DEG = old_h
        sensor.VERTICAL_DEG = old_v
    report = {"schema": "robot-learning-lab.ape-pre-show-sensor-coverage.v1",
              "source_sha256": sha(SOURCE),
              "v98_report_sha256": sha(V98 / "live-moving-owner-offer-audit.json"),
              "selected_visual_sha256": sha(selected_visual()),
              "mounts": MOUNTS,
              "layouts": LAYOUTS,
              "fov_deg_h_v": (FOV_H_DEG, FOV_V_DEG),
              "grid_h_v": (GRID_H, GRID_V),
              "sequence": SEQUENCES["back2_diag2"],
              "scope": "Single 16.2s pre-show snapshot in paired furnished owner scenes. Current self-masked ideal lower head depth is compared with hypothetical noisy/dropout near-body rear low+mid, optimistic 15cm-standoff rear low+mid, side views and an intentionally generous eight-array upper/lower full ring, all 31x21 first-hit. Retains source-tagged rays, mount-to-historical-body solid gaps and free/occupied/unknown 5cm-buffered swept-shell coverage for four commanded 8cm steps. Standoff and full-ring layouts are observability bounds, not packaging choices; 4cm voxel/held-joint shell is diagnostic, not a physical corridor certificate. Future prop motion, pose/slip uncertainty, floor classification and selected-body action are absent.",
              "rows": rows}
    path = OUT / "report.json"
    path.write_text(json.dumps(report, indent=2) + "\n")
    for row in rows:
        print(row["seed"], row["case"],
              {layout: [(step["upper_shell"]["status"],
                          step["lower_shell"]["status"])
                         for step in steps]
               for layout, steps in row["coverage"].items()})
    print(path)


if __name__ == "__main__":
    main()
