"""V200p: optimistic 2D lidar scan-plane role screen in changed owner rooms.

The official A1/A2 size, cadence, angular pitch and minimum range constrain
ideal MuJoCo first-hit rays. Near, self and no-return statuses stay unknown;
the actor packets contain no simulator geom IDs. This is not a calibrated
device, optics, reflectance, motion-distortion or mount simulation.
"""

from __future__ import annotations

import json
import math
from pathlib import Path

import mujoco
import numpy as np

from articulated_person_scene import (FAR_HAND_XYZ_M, START_HAND_XYZ_M,
                                      add_person, set_person_pose)
from compare_standoff_range_layouts_v165p import front_left_roi
from depth_voxel import DepthVoxelMap
import direct_structural_rooms_v270 as structural
from robot_geometry import RobotGeometry
from run import ROOT, sha
import run_heldout_furniture_v196p as v196p
import run_live_articulated_owner_v343 as owner
import run_pre_show_reserve_screen_v100 as v100
import run_standoff_scene_variants_v160p as v160p
import run_varied_furniture_owner_v191p as v191p
import screen_preshow_camera_shell_v181p as v181p
from screen_standoff_front_cue_v149p import front_left_stat


SOURCE = Path(__file__)
OUT = ROOT / "artifacts/ape-closed-loop-cove50-planar-lidar-owner"
DEVICES = {
    "a1_spec_proxy": {"minimum_m": .15, "maximum_m": 12.,
                      "scan_hz": 5.5, "angle_step_deg": 1.,
                      "radius_m": .0484, "height_m": .055,
                      "mass_kg": .170},
    "a2_spec_proxy": {"minimum_m": .20, "maximum_m": 12.,
                      "scan_hz": 10., "angle_step_deg": .45,
                      "radius_m": .038, "height_m": .041,
                      "mass_kg": .190},
}
MOUNTS = {
    "low_forward": (.17, 0., .07),
    "low_left_outboard": (0., .18, .07),
    "mid_forward": (.17, 0., .30),
}
START_S = 16.2
END_S = 18.2
REFERENCE_END_S = 16.5
NOVEL_DISTANCE_M = .04
MIN_NOVEL_POINTS = 4
STATUSES = {0: "valid_hit", 1: "near_invalid", 2: "self_first",
            3: "no_target"}


def rotate(yaw: float, local: np.ndarray) -> np.ndarray:
    c, s = math.cos(yaw), math.sin(yaw)
    return np.array((c*local[0]-s*local[1],
                     s*local[0]+c*local[1], local[2]))


def is_robot_hit(room: v191p.FurnitureRoom, geom_id: int) -> bool:
    body = int(room.model.geom_bodyid[geom_id])
    robot_root = room.model.body("pelvis").id
    while body > 0 and body != robot_root:
        body = int(room.model.body_parentid[body])
    return body == robot_root


def scan(room: v191p.FurnitureRoom, device: dict,
         mount: tuple[float, float, float],
         angles: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    actual_origin = np.asarray((*room.actual[:2], 0.)) + rotate(
        float(room.actual[2]), np.asarray(mount))
    actor_origin = np.asarray((*room.estimated[:2], 0.)) + rotate(
        float(room.estimated[2]), np.asarray(mount))
    ranges = np.full(len(angles), np.nan, dtype=np.float32)
    status = np.full(len(angles), 3, dtype=np.uint8)
    points = []
    group = np.ones(6, dtype=np.uint8)
    for index, angle in enumerate(angles):
        direction_local = np.asarray((math.cos(float(angle)),
                                      math.sin(float(angle)), 0.))
        direction = rotate(float(room.actual[2]), direction_local)
        geom = np.array([-1], dtype=np.int32)
        distance = mujoco.mj_ray(room.model, room.data, actual_origin,
                                 direction, group, True, -1, geom)
        if distance < 0 or distance > device["maximum_m"]:
            continue
        if geom[0] >= 0 and is_robot_hit(room, int(geom[0])):
            status[index] = 2
            continue
        if distance < device["minimum_m"]:
            status[index] = 1
            continue
        status[index] = 0
        ranges[index] = float(distance)
        actor_direction = rotate(float(room.estimated[2]), direction_local)
        points.append(actor_origin + distance*actor_direction)
    return ranges, status, np.asarray(points, dtype=float).reshape(-1, 3)


def set_scene_time(room: v191p.FurnitureRoom, t: float,
                   start_head: float) -> None:
    room.time_s = t
    shown = min(1., max(0., (t-16.6)/.4))
    hand = START_HAND_XYZ_M+shown*(FAR_HAND_XYZ_M-START_HAND_XYZ_M)
    set_person_pose(room.model, hand, carried_item="loop_TomatoSauce")
    if t >= 17.-1e-6:
        room.model.body_pos[room.model.body(
            "loop_TomatoSauce").id] = owner.OWNER_SHOW_XYZ_M
    look = min(1., max(0., (t-17.)/.4))
    room.head = start_head*(1.-look)
    room._forward()


def one(room: v191p.FurnitureRoom, seed: int, layout: str,
        motion: str, device_name: str, mount_name: str,
        shell: np.ndarray, body_solid_gap: float,
        start_head: float) -> dict:
    device = DEVICES[device_name]
    mount = MOUNTS[mount_name]
    angles = np.deg2rad(np.arange(0., 360., device["angle_step_deg"]))
    times = [round(START_S+i/device["scan_hz"], 5)
             for i in range(int(math.floor((END_S-START_S)*
                                           device["scan_hz"]))+1)]
    branch = OUT / layout / motion / f"seed{seed}" / device_name / mount_name
    branch.mkdir(parents=True, exist_ok=True)
    baseline = []
    last_count, last_centroid = 0, None
    first_cue = None
    packet_rows = []
    preshow_map = DepthVoxelMap()
    for index, t in enumerate(times):
        set_scene_time(room, t, start_head)
        ranges, status, points = scan(room, device, mount, angles)
        selected = points[front_left_roi(points, room.estimated.copy())]
        if t <= REFERENCE_END_S:
            baseline.append(selected.copy())
        reference = (np.concatenate(baseline) if baseline else
                     np.empty((0, 3)))
        novel = np.empty((0, 3))
        if t > REFERENCE_END_S and len(reference) and len(selected):
            distances = np.linalg.norm(
                selected[:, None, :]-reference[None, :, :], axis=2)
            novel = selected[np.min(distances, axis=1) > NOVEL_DISTANCE_M]
        centroid = np.mean(novel, axis=0) if len(novel) else None
        motion_m = (float(np.linalg.norm(centroid-last_centroid))
                    if centroid is not None and last_centroid is not None
                    else None)
        cue = (len(novel) >= MIN_NOVEL_POINTS and
               (last_count < MIN_NOVEL_POINTS or
                (motion_m is not None and motion_m >= .012)))
        if cue and first_cue is None:
            first_cue = t
        last_count, last_centroid = len(novel), centroid
        path = branch / f"planar-{index:02d}.npz"
        np.savez_compressed(path, ranges_m=ranges, status=status,
                            azimuth_rad=angles.astype(np.float32),
                            actor_origin_m=np.asarray((*room.estimated[:2], 0.))+
                                rotate(float(room.estimated[2]),
                                       np.asarray(mount)),
                            estimated_pose_xyyaw=room.estimated.copy(),
                            capture_time_s=t)
        if t <= REFERENCE_END_S:
            origin = np.asarray((*room.estimated[:2], 0.)) + rotate(
                float(room.estimated[2]), np.asarray(mount))
            for point in points:
                direction = point-origin
                distance = float(np.linalg.norm(direction))
                if distance <= .06:
                    continue
                for along in np.arange(.04, distance-.06, .04):
                    preshow_map.free[preshow_map.cell(
                        origin+along*direction/distance)] = t
            preshow_map.last_time_s = t
        _, p10 = front_left_stat(points, room.estimated.copy())
        packet_rows.append({
            "capture_time_s": t,
            "actor_packet": str(path.relative_to(ROOT)),
            "actor_packet_sha256": sha(path),
            "status_counts": {name: int(np.count_nonzero(status == code))
                              for code, name in STATUSES.items()},
            "front_left_return_count": len(selected),
            "front_left_p10_x_m": p10,
            "new_low_returns": len(novel),
            "new_surface_cue": bool(cue),
        })
    coverage = v181p.shell_state(preshow_map, shell, REFERENCE_END_S)
    return {"layout": layout, "motion": motion, "seed": seed,
            "device": device_name, "mount": mount_name,
            "origin_to_body_solid_gap_m": round(body_solid_gap, 5),
            "nominal_radial_package_clearance_m": round(
                body_solid_gap-device["radius_m"], 5),
            "point_mount_radially_clear_of_solid": bool(
                body_solid_gap >= device["radius_m"]),
            "preshow_shell": coverage,
            "first_new_low_surface_cue_s": first_cue,
            "packets": packet_rows}


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    pointer = json.loads((ROOT / "robot/ape-current-design.json").read_text())
    if pointer["status"] != "rejected_no_selected_mechanical_design":
        raise AssertionError("Historical-body scope changed")
    old_model = structural.structural_model
    structural.structural_model = lambda kind, side: add_person(
        old_model(kind, side))
    v160p.VARIANTS["fast_shifted"] = v196p.NEW_PATH
    v160p.VARIANTS["stationary_shifted"] = {
        **v196p.NEW_PATH, "approach_start_s": 30.,
        "approach_end_s": 31.}
    rows = []
    try:
        v98 = json.loads((v100.V98 /
                          "live-moving-owner-offer-audit.json").read_text())
        for layout, cfg in v196p.LAYOUTS.items():
            v191p.LAYOUTS[layout] = cfg
            v191p.FurnitureRoom.layout_name = layout
            for motion in ("moving", "stationary"):
                v191p.FurnitureRoom.variant = v160p.VARIANTS[
                    "fast_shifted" if motion == "moving" else
                    "stationary_shifted"]
                for seed in (170, 172):
                    prior = next(r for r in v98["rows"] if
                                 r["condition"] == "moving" and
                                 r["case"][0] == seed)
                    source = (v100.V98 / "moving" / f"seed{seed}-present" /
                              f"seed{seed}-present/side-probe-00.npz")
                    with np.load(source, allow_pickle=False) as packet:
                        qpos = packet["actor_qpos"].copy()
                        estimate = packet["estimated_pose_xyyaw"].copy()
                    room = v191p.FurnitureRoom(
                        case=tuple(prior["case"]), pose_source="command",
                        branch=OUT)
                    try:
                        room.layout(seed, absent=False, furnished=True)
                        qpos[0] += -.30
                        estimate[0] += -.30
                        qpos[room.model.joint(
                            "left_shoulder_pitch").qposadr[0]] = .6
                        room.data.qpos[:] = qpos
                        room.actual[:] = (qpos[0], qpos[1], 0.)
                        room.estimated[:] = estimate
                        room.head = float(qpos[room.head_addr])
                        start_head = room.head
                        set_scene_time(room, START_S, start_head)
                        robot = RobotGeometry(room.model)
                        shell = v181p.front_left_low_shell(
                            robot, qpos, estimate)
                        robot.set_state(qpos)
                        solids = robot.primitives()
                        for device in DEVICES:
                            for mount_name, mount in MOUNTS.items():
                                origin = np.asarray((*room.actual[:2], 0.)) + rotate(
                                    float(room.actual[2]), np.asarray(mount))
                                gap = RobotGeometry.solid_distance(
                                    origin, solids)
                                rows.append(one(room, seed, layout, motion,
                                                device, mount_name, shell, gap,
                                                start_head))
                                print(layout, motion, seed, device, mount_name,
                                      "cue", rows[-1]["first_new_low_surface_cue_s"],
                                      "shell", rows[-1]["preshow_shell"]["free"],
                                      "mount", rows[-1]["nominal_radial_package_clearance_m"],
                                      flush=True)
                    finally:
                        room.close()
    finally:
        structural.structural_model = old_model
        del v160p.VARIANTS["fast_shifted"]
        del v160p.VARIANTS["stationary_shifted"]
    report = {"schema": "robot-learning-lab.ape-planar-lidar-owner.v1",
              "source_sha256": sha(SOURCE),
              "v196_source_sha256": sha(Path(v196p.__file__)),
              "current_design_status": pointer["status"],
              "devices": DEVICES, "mounts_body_xyz_m": MOUNTS,
              "status_codes": STATUSES,
              "reference_end_s": REFERENCE_END_S,
              "novel_distance_m": NOVEL_DISTANCE_M,
              "minimum_novel_returns": MIN_NOVEL_POINTS,
              "manufacturer_sources": [
                  "https://www.slamtec.com/en/lidar/a1spec",
                  "https://www.slamtec.com/en/lidar/a2spec"],
              "scope": "Optimistic ideal 2D horizontal first-hit MuJoCo rays "
                       "at official A1/A2 cadence, angular step, min range "
                       "and package radius in two V196 owner rooms, moving "
                       "and stationary low prop, two lighting seeds. No "
                       "reflectance, optical noise, spinning distortion, "
                       "multiple echo, shell/fastener fit, moving robot or "
                       "physical sensor claim. Actor packets omit truth IDs; "
                       "no-hit and too-near readings mark unknown, not free.",
              "rows": rows}
    (OUT / "report.json").write_text(json.dumps(report, indent=2) + "\n")
    (OUT / SOURCE.name).write_bytes(SOURCE.read_bytes())
    print(OUT / "report.json")


if __name__ == "__main__":
    main()
