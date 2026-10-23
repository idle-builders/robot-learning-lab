"""Compare low left-side range viewpoints against the pre-show body shell.

These are virtual multizone ray layouts on the rejected historical body.
Range max-distance rays are an optimistic visibility bound; their physical
no-return semantics and package fit have not been established.
"""

from __future__ import annotations

import json
import math
from pathlib import Path

import numpy as np

from articulated_person_scene import START_HAND_XYZ_M, add_person, set_person_pose
import audit_blocked_range_mounts_v319 as range_model
from depth_voxel import DepthVoxelMap
import direct_structural_rooms_v270 as structural
from robot_geometry import RobotGeometry
from run import ROOT, sha
import run_pre_show_reserve_screen_v100 as v100
import run_standoff_scene_variants_v160p as v160p
import screen_preshow_camera_shell_v181p as v181p
import bound_preshow_camera_misses_v183p as v183p


SOURCE = Path(__file__)
OUT = ROOT / "artifacts/ape-closed-loop-cove50-low-side-shell-range"
TIME_S = 16.2
MOUNTS = {
    "near_left_outward": ((0., .095, .11), math.pi/2),
    "near_left_diagonal": ((.03, .095, .11), math.pi/4),
    "far_left_inward_bound": ((0., .18, .11), -math.pi/2),
    "far_left_outward_control": ((0., .18, .11), math.pi/2),
}
HORIZONTAL_DEG = np.linspace(-55., 55., 31)
VERTICAL_DEG = np.linspace(-40., 40., 21)


def camera_maps(source: dict) -> tuple[DepthVoxelMap, DepthVoxelMap]:
    returned = []
    ideal = []
    for camera in v181p.CAMERAS:
        saved = next(p for p in source["pre_show_camera_packets"]
                     if p["camera"] == camera and
                     p["capture_time_s"] == TIME_S)
        path = ROOT / saved["packet_path"]
        if sha(path) != saved["packet_sha256"]:
            raise AssertionError("Saved camera packet changed")
        with np.load(path, allow_pickle=False) as packet:
            depth = packet["filtered_depth_m"].copy()
            origin = packet["camera_origin_xyz_m"].copy()
            rotation = packet["camera_rotation"].copy()
            fovy = float(packet["fovy_deg"])
        one = DepthVoxelMap()
        one.observe(depth, (origin, rotation), TIME_S,
                    fovy_deg=fovy, stride=v181p.DEPTH_STRIDE,
                    self_masked=True)
        optimistic = DepthVoxelMap()
        optimistic.free = one.free.copy()
        optimistic.occupied = one.occupied.copy()
        optimistic.frames = one.frames
        optimistic.last_time_s = one.last_time_s
        v183p.add_simulator_miss_rays(
            optimistic, depth, origin, rotation, fovy, TIME_S)
        returned.append(one)
        ideal.append(optimistic)
    return v181p.merged(*returned), v181p.merged(*ideal)


def state(voxel_map: DepthVoxelMap, shell: np.ndarray) -> dict:
    return {**v183p.states(voxel_map, shell, TIME_S),
            "samples": len(shell)}


def scan_case(source: dict, prior: dict) -> dict:
    case_id = source["source_case_id"]
    branch = OUT / case_id
    branch.mkdir(parents=True, exist_ok=True)
    with np.load(ROOT / source["shell_packet"],
                 allow_pickle=False) as packet:
        shell = packet["shell_samples_xyz_m"].copy()
        qpos = packet["actor_qpos"].copy()
        estimate = packet["estimated_pose_xyyaw"].copy()
    camera_returned, camera_ideal = camera_maps(source)
    v160p.VariantRoom.variant = v160p.VARIANTS[source["variant"]]
    room = v160p.VariantRoom(case=tuple(prior["case"]),
                             pose_source="command", branch=branch)
    try:
        room.layout(source["seed"], absent=False, furnished=True)
        room.data.qpos[:] = qpos
        room.actual[:] = (qpos[0], qpos[1],
                          2*math.atan2(qpos[6], qpos[3]))
        room.estimated[:] = estimate
        room.head = float(qpos[room.head_addr])
        room.time_s = TIME_S
        set_person_pose(room.model, START_HAND_XYZ_M,
                        carried_item="loop_TomatoSauce")
        room._forward()
        robot = RobotGeometry(room.model)
        robot.set_state(qpos)
        solids = robot.primitives()
        rows = []
        for name, (offset, _) in MOUNTS.items():
            actual_origin = np.array([*room.actual[:2], 0.]) + (
                range_model.rotate(float(room.actual[2]),
                                   np.asarray(offset)))
            body_gap = RobotGeometry.solid_distance(actual_origin, solids)
            rays, counts = range_model.range_packet(
                room, estimate, name, source["seed"], TIME_S)
            packet_path = branch / f"{name}-actor-rays.json"
            packet_path.write_text(json.dumps({
                "source_case_id": case_id,
                "mount": name,
                "capture_time_s": TIME_S,
                "rays": rays,
            }, separators=(",", ":")) + "\n")
            hit_only = DepthVoxelMap()
            range_model.add_packet(
                hit_only, [ray for ray in rays if ray["endpoint_hit"]],
                TIME_S)
            with_misses = DepthVoxelMap()
            range_model.add_packet(with_misses, rays, TIME_S)
            rows.append({
                "mount": name,
                "actor_ray_packet": str(packet_path.relative_to(ROOT)),
                "actor_ray_packet_sha256": sha(packet_path),
                "origin_to_historical_collision_solid_gap_m": round(
                    body_gap, 5),
                "ray_counts": counts,
                "range_hit_only": state(hit_only, shell),
                "range_with_max_distance_rays": state(with_misses, shell),
                "cameras_return_only_plus_range_hit_only": state(
                    v181p.merged(camera_returned, hit_only), shell),
                "cameras_return_only_plus_range_max_rays": state(
                    v181p.merged(camera_returned, with_misses), shell),
                "optimistic_cameras_plus_range_max_rays": state(
                    v181p.merged(camera_ideal, with_misses), shell),
            })
        return {
            "source_case_id": case_id,
            "variant": source["variant"],
            "seed": source["seed"],
            "station_x_m": source["station_x_m"],
            "source_shell_packet": source["shell_packet"],
            "source_shell_packet_sha256": source[
                "shell_packet_sha256"],
            "shell_cells": len(shell),
            "camera_return_only": state(camera_returned, shell),
            "camera_optimistic_miss_as_free": state(camera_ideal, shell),
            "mount_rows": rows,
        }
    finally:
        room.close()


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    source_file = v181p.OUT / "report.json"
    source = json.loads(source_file.read_text())
    v98_file = v100.V98 / "live-moving-owner-offer-audit.json"
    v98 = json.loads(v98_file.read_text())
    old_model = structural.structural_model
    old_mounts = range_model.MOUNTS
    old_h = range_model.HORIZONTAL_DEG
    old_v = range_model.VERTICAL_DEG
    structural.structural_model = lambda kind, side: add_person(
        old_model(kind, side))
    range_model.MOUNTS = MOUNTS
    range_model.HORIZONTAL_DEG = HORIZONTAL_DEG
    range_model.VERTICAL_DEG = VERTICAL_DEG
    rows = []
    try:
        for item in source["rows"]:
            prior = next(row for row in v98["rows"]
                         if row["condition"] == "moving" and
                         row["case"][0] == item["seed"])
            rows.append(scan_case(item, prior))
            print(item["source_case_id"], [(r["mount"], r[
                "range_with_max_distance_rays"]["free"], r[
                "optimistic_cameras_plus_range_max_rays"]["free"])
                for r in rows[-1]["mount_rows"]], flush=True)
    finally:
        range_model.MOUNTS = old_mounts
        range_model.HORIZONTAL_DEG = old_h
        range_model.VERTICAL_DEG = old_v
        structural.structural_model = old_model
    summary = {}
    for mount in MOUNTS:
        group = [next(r for r in case["mount_rows"] if r["mount"] == mount)
                 for case in rows]
        summary[mount] = {
            "cases": len(group),
            "shell_cells": sum(case["shell_cells"] for case in rows),
            "body_gap_range_m": [min(r[
                "origin_to_historical_collision_solid_gap_m"] for r in group),
                max(r["origin_to_historical_collision_solid_gap_m"]
                    for r in group)],
            "range_hit_only_free": sum(r["range_hit_only"]["free"]
                                       for r in group),
            "range_max_ray_free": sum(r[
                "range_with_max_distance_rays"]["free"] for r in group),
            "return_camera_plus_range_max_free": sum(r[
                "cameras_return_only_plus_range_max_rays"]["free"]
                for r in group),
            "optimistic_camera_plus_range_max_free": sum(r[
                "optimistic_cameras_plus_range_max_rays"]["free"]
                for r in group),
        }
    report = {
        "schema": "robot-learning-lab.ape-low-side-shell-range.v1",
        "source_sha256": sha(SOURCE),
        "v181_report_sha256": sha(source_file),
        "range_model_source_sha256": sha(Path(range_model.__file__)),
        "v98_report_sha256": sha(v98_file),
        "capture_time_s": TIME_S,
        "mounts_body_xyz_yaw_rad": MOUNTS,
        "fov_h_v_deg": [110., 80.],
        "ray_grid_h_v": [31, 21],
        "range_max_m": range_model.RANGE_MAX_M,
        "range_noise_sigma_m": range_model.RANGE_NOISE_STD_M,
        "dropout_fraction": range_model.DROPOUT_FRACTION,
        "scope": "One simulated 16.2s pre-show first-hit packet per candidate "
                 "view, on the same eight stationary historical-body "
                 "scenes as V181. Sources are tagged, actor extrinsics use "
                 "estimated pose, and 15% dropout/5mm range noise come "
                 "from V319's virtual multizone model. Hit-only rays and "
                 "optimistic max-distance rays are separated. Candidate "
                 "origin collision-solid gaps are not module/shell mounts. "
                 "No selected body, physical no-hit semantics, response "
                 "action, whole-body coverage or hardware choice follows.",
        "summary": summary,
        "rows": rows,
    }
    (OUT / "report.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
