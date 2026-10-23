"""V103: isolate dropout from persistent unknown space before a back step.

Repeated independent noisy ray packets are generated at one frozen pre-show
scene and pose. This is a sensor upper bound, not elapsed owner interaction.
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
from robot_geometry import RobotGeometry
from run import H, ROOT, W, sha
from run_articulated_owner_low_drift_v92 import OfferDriftRoom
from run_moving_low_clutter_v302 import STOOL_SIZE
from run_owner_exit_height_v96 import CASES
from run_pre_show_sensor_coverage_v102 import (GRID_H, GRID_V, FOV_H_DEG,
                                                FOV_V_DEG, LAYOUTS, MOUNTS)
from self_depth_mask import SelfDepthMask


SOURCE = Path(__file__)
OUT = ROOT / "artifacts/ape-closed-loop-cove50-v103"
V98 = ROOT / "artifacts/ape-closed-loop-cove50-v98"
REPEATS = (1, 2, 4, 8)
LAYOUT_NAMES = ("head_rear_low_mid_near", "head_full_ring_low_upper")


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
                    room.time_s = 16.2
                    set_person_pose(room.model, START_HAND_XYZ_M,
                                    carried_item="loop_TomatoSauce")
                    room.obstacle_xy[3] = (0., .306)
                    room.model.body_pos[room.model.body("loop_stool").id] = (
                        0., .306, STOOL_SIZE[2])
                    room._forward()
                    robot = RobotGeometry(room.model)
                    _, depth = room.capture("lower_rgb", depth=True)
                    mask = SelfDepthMask(room.model, camera="lower_rgb",
                                         width=W, height=H, agreement_m=.04)
                    try:
                        filtered, _ = mask.apply(
                            depth, room.data.qpos.copy(), max_depth_m=1.4)
                    finally:
                        mask.close()
                    fovy = float(room.model.cam_fovy[
                        room.model.camera("lower_rgb").id])
                    base = DepthVoxelMap()
                    base.observe(filtered, room.camera_pose_estimate(
                        "lower_rgb"), 16.2, fovy_deg=fovy,
                        stride=4, max_depth_m=1.4, self_masked=True)
                    maps = {}
                    for name in LAYOUT_NAMES:
                        voxel_map = DepthVoxelMap()
                        voxel_map.free = base.free.copy()
                        voxel_map.occupied = base.occupied.copy()
                        voxel_map.last_time_s = 16.2
                        maps[name] = voxel_map
                    future = pose.copy()
                    future[0] -= .08
                    observations = []
                    repeated_rays = {}
                    for repeat in range(1, max(REPEATS)+1):
                        packets = {}
                        counts = {}
                        for mount in sorted({m for layout in LAYOUT_NAMES
                                             for m in LAYOUTS[layout]}):
                            rays, count = sensor.range_packet(
                                room, pose, mount,
                                seed*100 + repeat, 16.2)
                            packets[mount] = rays
                            counts[mount] = count
                            for layout in LAYOUT_NAMES:
                                if mount in LAYOUTS[layout]:
                                    sensor.add_packet(maps[layout], rays, 16.2)
                        repeated_rays[str(repeat)] = packets
                        if repeat in REPEATS:
                            coverage = {}
                            for layout, voxel_map in maps.items():
                                coverage[layout] = {
                                    "upper": robot.swept_exterior_coverage(
                                        voxel_map, qpos, pose, future, 16.2,
                                        margin_m=.05, z_min_m=.18,
                                        z_max_m=.43),
                                    "lower": robot.swept_exterior_coverage(
                                        voxel_map, qpos, pose, future, 16.2,
                                        margin_m=.05, z_min_m=.02,
                                        z_max_m=.18),
                                }
                            observations.append({"repeat_count": repeat,
                                                 "coverage": coverage,
                                                 "latest_mount_counts": counts})
                    branch = OUT / f"seed{seed}-{case}"
                    branch.mkdir(parents=True, exist_ok=True)
                    packet_path = branch / "repeated-actor-rays.json"
                    packet_path.write_text(json.dumps({
                        "estimated_pose_xyyaw": pose.tolist(),
                        "actor_qpos": qpos.tolist(),
                        "frozen_scene_time_s": 16.2,
                        "independent_noise_seed_rule":
                            "seed*100 + repeat_index",
                        "ray_packets_by_repeat": repeated_rays,
                    }, indent=2) + "\n")
                    rows.append({"seed": seed, "case": case,
                                 "source_packet_path": str(source.relative_to(ROOT)),
                                 "source_packet_sha256": sha(source),
                                 "actor_packets_path": str(packet_path.relative_to(ROOT)),
                                 "actor_packets_sha256": sha(packet_path),
                                 "observations": observations})
                finally:
                    room.close()
    finally:
        structural.structural_model = old_model
        sensor.MOUNTS.clear()
        sensor.MOUNTS.update(old_mounts)
        sensor.HORIZONTAL_DEG = old_h
        sensor.VERTICAL_DEG = old_v
    report = {"schema": "robot-learning-lab.ape-pre-show-repeat-coverage.v1",
              "source_sha256": sha(SOURCE),
              "v98_report_sha256": sha(V98 / "live-moving-owner-offer-audit.json"),
              "mounts": MOUNTS,
              "layouts": {name: LAYOUTS[name] for name in LAYOUT_NAMES},
              "repeats": REPEATS,
              "scope": "Frozen pre-show pose/scene with ideal self-masked head depth and hypothetical 31x21 noisy first-hit arrays. Repeats 1/2/4/8 independently seeded range packets at identical geometry to separate random dropout from persistent unknown swept space. This is an optimistic static observability diagnostic, not real elapsed owner time, dynamic free-space permission, deployable packaging or selected-body clearance.",
              "rows": rows}
    path = OUT / "report.json"
    path.write_text(json.dumps(report, indent=2) + "\n")
    for row in rows:
        print(row["seed"], row["case"], [
            (entry["repeat_count"],
             [(layout, entry["coverage"][layout]["upper"]["unknown"],
               entry["coverage"][layout]["lower"]["unknown"])
              for layout in LAYOUT_NAMES])
            for entry in row["observations"]])
    print(path)


if __name__ == "__main__":
    main()
