"""V117: packet-level test of V116's inward low-side visibility idea.

Matched 31x21, 110x80 degree, 60 cm first-hit arrays add two inward side
views to V102's retained head/rear/right-upper actor map. This compares
actual noisy/dropout rays against V116's LOS upper bound; it cannot select a
physical mount or authorize a step while upper/lower cells remain unknown.
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
from run import ROOT, sha
from run_articulated_owner_low_drift_v92 import OfferDriftRoom
from run_moving_low_clutter_v302 import STOOL_SIZE
from run_owner_exit_height_v96 import CASES
from run_pre_show_reserve_screen_v100 import (COMMAND_STEP_M, DIRECTIONS,
                                              SEQUENCES, START_S, V98)
from run_pre_show_sensor_coverage_v102 import (FOV_H_DEG, FOV_V_DEG,
                                                GRID_H, GRID_V, MOUNTS)


SOURCE = Path(__file__)
OUT = ROOT / "artifacts/ape-closed-loop-cove50-v117"
V102 = ROOT / "artifacts/ape-closed-loop-cove50-v102"
V109 = ROOT / "artifacts/ape-closed-loop-cove50-v109"
NEW_MOUNTS = {
    "left_low_inward": (MOUNTS["left_low"][0], -math.pi/2),
    "right_low_inward": (MOUNTS["right_low"][0], math.pi/2),
    "rear_high_mid": ((-.06, 0., .35), math.pi),
}
BASE = ("rear_low_near", "rear_mid_near", "right_upper")
LAYOUTS = {
    "outward_baseline": BASE,
    "plus_inward_low_pair": (*BASE, "left_low_inward", "right_low_inward"),
    "plus_rear_high": (*BASE, "rear_high_mid"),
    "plus_inward_low_and_rear_high": (
        *BASE, "left_low_inward", "right_low_inward", "rear_high_mid"),
}


def route_coverage(robot: RobotGeometry, voxel_map: DepthVoxelMap,
                   qpos: np.ndarray, pose: np.ndarray) -> list[dict]:
    current = pose.copy()
    steps = []
    for command in SEQUENCES["back2_diag2"]:
        future = current.copy()
        future[:2] += COMMAND_STEP_M*np.asarray(DIRECTIONS[command])
        steps.append({
            "command": command,
            "upper": robot.swept_exterior_coverage(
                voxel_map, qpos, current, future, START_S,
                margin_m=.05, z_min_m=.18, z_max_m=.43),
            "lower": robot.swept_exterior_coverage(
                voxel_map, qpos, current, future, START_S,
                margin_m=.05, z_min_m=.02, z_max_m=.18),
        })
        current = future
    return steps


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    v98 = json.loads((V98 / "live-moving-owner-offer-audit.json").read_text())
    v102 = json.loads((V102 / "report.json").read_text())
    old_model = structural.structural_model
    old_mounts = sensor.MOUNTS.copy()
    old_h = sensor.HORIZONTAL_DEG.copy()
    old_v = sensor.VERTICAL_DEG.copy()
    structural.structural_model = lambda kind, side: add_person(
        old_model(kind, side))
    sensor.MOUNTS.clear()
    sensor.MOUNTS.update(MOUNTS)
    sensor.MOUNTS.update(NEW_MOUNTS)
    sensor.HORIZONTAL_DEG = np.linspace(-FOV_H_DEG/2, FOV_H_DEG/2, GRID_H)
    sensor.VERTICAL_DEG = np.linspace(-FOV_V_DEG/2, FOV_V_DEG/2, GRID_V)
    rows = []
    try:
        for seed in (170, 172):
            prior = next(row for row in v98["rows"]
                         if row["condition"] == "moving" and
                         row["case"][0] == seed)
            for case in ("clear", "hand_close"):
                prior_row = next(row for row in v102["rows"]
                                 if row["seed"] == seed and
                                 row["case"] == case)
                actor_path = ROOT / prior_row["actor_packet_path"]
                if sha(actor_path) != prior_row["actor_packet_sha256"]:
                    raise AssertionError("V102 ray packet changed")
                actor = json.loads(actor_path.read_text())
                head_path = (V109 / f"seed{seed}-{case}" /
                             "current-head-rgbd.npz")
                with np.load(head_path) as packet:
                    depth = packet["filtered_depth_m"].copy()
                    camera = (packet["camera_origin_estimated_m"].copy(),
                              packet["camera_rotation_estimated"].copy())
                qpos = np.asarray(actor["actor_qpos"], dtype=float)
                pose = np.asarray(actor["estimated_pose_xyyaw"], dtype=float)
                room = OfferDriftRoom(case=tuple(prior["case"]),
                                      pose_source="command", branch=OUT)
                try:
                    room.layout(seed, absent=False, furnished=True)
                    obstacle = CASES[case]
                    if obstacle is not None:
                        position, size = obstacle
                        room.model.body_pos[room.model.body(
                            "loop_box").id] = position
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
                    solid = robot.primitives()
                    new_packets = {}
                    mount_gaps = {}
                    packet_counts = {}
                    for name, (offset, _) in NEW_MOUNTS.items():
                        origin = (np.array([*room.actual[:2], 0.]) +
                                  sensor.rotate(float(room.actual[2]),
                                                np.asarray(offset)))
                        mount_gaps[name] = round(robot.solid_distance(
                            origin, solid), 5)
                        new_packets[name], packet_counts[name] = (
                            sensor.range_packet(room, pose, name,
                                                seed, START_S))
                    fovy = float(room.model.cam_fovy[
                        room.model.camera("lower_rgb").id])
                    base = DepthVoxelMap()
                    base.observe(depth, camera, START_S, fovy_deg=fovy,
                                 stride=4, max_depth_m=1.4,
                                 self_masked=True)
                    layouts = {}
                    for label, names in LAYOUTS.items():
                        voxel_map = DepthVoxelMap()
                        voxel_map.free = base.free.copy()
                        voxel_map.occupied = base.occupied.copy()
                        voxel_map.last_time_s = START_S
                        for name in names:
                            rays = (new_packets[name] if name in NEW_MOUNTS
                                    else actor["ray_packets"][name])
                            sensor.add_packet(voxel_map, rays, START_S)
                        layouts[label] = route_coverage(
                            robot, voxel_map, qpos, pose)
                    old = prior_row["coverage"][
                        "head_rear_low_mid_near_right_upper"]
                    for index, step in enumerate(old):
                        for band in ("upper", "lower"):
                            key = band+"_shell"
                            if layouts["outward_baseline"][index][band] != step[key]:
                                raise AssertionError("V102 baseline mismatch")
                    branch = OUT / f"seed{seed}-{case}"
                    branch.mkdir(parents=True, exist_ok=True)
                    packet_path = branch / "new-actor-rays.json"
                    packet_path.write_text(json.dumps({
                        "capture_time_s": START_S,
                        "estimated_pose_xyyaw": pose.tolist(),
                        "new_mounts": NEW_MOUNTS,
                        "ray_packets": new_packets,
                    }, indent=2)+"\n")
                    rows.append({
                        "seed": seed, "case": case,
                        "v102_actor_packet_path": str(actor_path.relative_to(ROOT)),
                        "v102_actor_packet_sha256": sha(actor_path),
                        "v109_head_packet_path": str(head_path.relative_to(ROOT)),
                        "v109_head_packet_sha256": sha(head_path),
                        "new_actor_packet_path": str(packet_path.relative_to(ROOT)),
                        "new_actor_packet_sha256": sha(packet_path),
                        "mount_origin_to_historical_solid_gap_m": mount_gaps,
                        "new_mount_counts": packet_counts,
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
        "schema": "robot-learning-lab.ape-inward-low-mount-packets.v1",
        "source_sha256": sha(SOURCE),
        "v102_report_sha256": sha(V102 / "report.json"),
        "v109_report_sha256": sha(V109 / "report.json"),
        "selected_visual_sha256": sha(selected_visual()),
        "new_mounts": NEW_MOUNTS,
        "layouts": LAYOUTS,
        "grid_h_v": [GRID_H, GRID_V],
        "fov_h_v_deg": [FOV_H_DEG, FOV_V_DEG],
        "range_max_m": sensor.RANGE_MAX_M,
        "noise_std_m": sensor.RANGE_NOISE_STD_M,
        "dropout_fraction": sensor.DROPOUT_FRACTION,
        "scope": "Same current head RGB-D and retained V102 outward rays at16.2s, plus new inward low-side and high rear first-hit arrays with matched V102 31x21 angular resolution, 60cm range, 5mm noise and15% dropout. Exact V102 baseline reproduced on four back/diagonal steps. Actor map sees rays and estimated pose only; evaluator scene sets ray hits. No actuator, selected-body hardware package, optics, pose bound or motion authorization; two fixed development scenes and seeds.",
        "rows": rows,
    }
    path = OUT / "report.json"
    path.write_text(json.dumps(report, indent=2)+"\n")
    for row in rows:
        print(row["seed"], row["case"], {
            label: (steps[0]["upper"]["unknown"],
                    steps[0]["lower"]["unknown"])
            for label, steps in row["coverage"].items()})
    print(path)


if __name__ == "__main__":
    main()
