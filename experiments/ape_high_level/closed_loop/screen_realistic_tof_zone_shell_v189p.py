"""Screen low-shell geometry with published 8x8 multizone FoV envelopes.

Only zone-center first-hit rays are simulated. They are not a faithful model
of ST sensor zone footprints, status codes, crosstalk or reflectance limits.
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
import screen_low_side_range_shell_v186p as v186p


SOURCE = Path(__file__)
OUT = ROOT / "artifacts/ape-closed-loop-cove50-tof-zone-shell-screen"
TIME_S = 16.2
MOUNTS = {name: v186p.MOUNTS[name] for name in (
    "near_left_outward", "near_left_diagonal", "far_left_inward_bound")}
FOOTPRINTS = {
    "vl53l8cx_45deg_8x8": {
        "fov_deg": 45.,
        "official_spec": "https://www.st.com/content/st_com/en/technical-documents/DS14161.html",
    },
    "vl53l7cx_60deg_8x8": {
        "fov_deg": 60.,
        "official_spec": "https://www.st.com/resource/en/datasheet/vl53l7cx.pdf",
    },
}
NOISE_PROBES_M = (.005, .012)


def zone_centers(fov_deg: float) -> np.ndarray:
    return (np.arange(8)+.5-4.)*fov_deg/8.


def state(voxel_map: DepthVoxelMap, shell: np.ndarray) -> dict:
    return {**v186p.v183p.states(voxel_map, shell, TIME_S),
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
    camera_returned, camera_ideal = v186p.camera_maps(source)
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
        for footprint, spec in FOOTPRINTS.items():
            centers = zone_centers(spec["fov_deg"])
            range_model.HORIZONTAL_DEG = centers
            range_model.VERTICAL_DEG = centers
            for noise_m in NOISE_PROBES_M:
                range_model.RANGE_NOISE_STD_M = noise_m
                for mount, (offset, _) in MOUNTS.items():
                    actual_origin = np.array([*room.actual[:2], 0.]) + (
                        range_model.rotate(float(room.actual[2]),
                                           np.asarray(offset)))
                    body_gap = RobotGeometry.solid_distance(
                        actual_origin, solids)
                    rays, counts = range_model.range_packet(
                        room, estimate, mount, source["seed"], TIME_S)
                    packet_path = branch / (
                        f"{footprint}-noise{int(noise_m*1000):02d}mm-"
                        f"{mount}.json")
                    packet_path.write_text(json.dumps({
                        "source_case_id": case_id,
                        "footprint": footprint,
                        "mount": mount,
                        "noise_sigma_m": noise_m,
                        "capture_time_s": TIME_S,
                        "rays": rays,
                    }, separators=(",", ":")) + "\n")
                    hit_only = DepthVoxelMap()
                    range_model.add_packet(
                        hit_only, [ray for ray in rays
                                   if ray["endpoint_hit"]], TIME_S)
                    max_rays = DepthVoxelMap()
                    range_model.add_packet(max_rays, rays, TIME_S)
                    rows.append({
                        "footprint": footprint,
                        "noise_sigma_m": noise_m,
                        "mount": mount,
                        "actor_ray_packet": str(packet_path.relative_to(ROOT)),
                        "actor_ray_packet_sha256": sha(packet_path),
                        "origin_to_historical_collision_solid_gap_m": round(
                            body_gap, 5),
                        "ray_counts": counts,
                        "range_hit_only": state(hit_only, shell),
                        "range_with_max_distance_rays": state(
                            max_rays, shell),
                        "return_cameras_plus_range_hit_only": state(
                            v181p.merged(camera_returned, hit_only), shell),
                        "return_cameras_plus_range_max_rays": state(
                            v181p.merged(camera_returned, max_rays), shell),
                        "optimistic_cameras_plus_range_max_rays": state(
                            v181p.merged(camera_ideal, max_rays), shell),
                    })
        return {
            "source_case_id": case_id,
            "source_shell_packet": source["shell_packet"],
            "source_shell_packet_sha256": source[
                "shell_packet_sha256"],
            "shell_cells": len(shell),
            "rows": rows,
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
    old_noise = range_model.RANGE_NOISE_STD_M
    structural.structural_model = lambda kind, side: add_person(
        old_model(kind, side))
    range_model.MOUNTS = MOUNTS
    rows = []
    try:
        for item in source["rows"]:
            prior = next(row for row in v98["rows"]
                         if row["condition"] == "moving" and
                         row["case"][0] == item["seed"])
            rows.append(scan_case(item, prior))
            print(item["source_case_id"], [(r["footprint"],
                r["noise_sigma_m"], r["mount"],
                r["optimistic_cameras_plus_range_max_rays"]["free"])
                for r in rows[-1]["rows"]], flush=True)
    finally:
        range_model.MOUNTS = old_mounts
        range_model.HORIZONTAL_DEG = old_h
        range_model.VERTICAL_DEG = old_v
        range_model.RANGE_NOISE_STD_M = old_noise
        structural.structural_model = old_model
    summary = {}
    for footprint in FOOTPRINTS:
        for noise_m in NOISE_PROBES_M:
            for mount in MOUNTS:
                group = [next(r for r in case["rows"]
                              if r["footprint"] == footprint and
                              r["noise_sigma_m"] == noise_m and
                              r["mount"] == mount) for case in rows]
                summary[f"{footprint}/{int(noise_m*1000)}mm/{mount}"] = {
                    "cases": len(group),
                    "shell_cells": sum(case["shell_cells"] for case in rows),
                    "range_hit_only_free": sum(r[
                        "range_hit_only"]["free"] for r in group),
                    "range_max_ray_free": sum(r[
                        "range_with_max_distance_rays"]["free"]
                        for r in group),
                    "return_camera_plus_range_max_free": sum(r[
                        "return_cameras_plus_range_max_rays"]["free"]
                        for r in group),
                    "optimistic_camera_plus_range_max_free": sum(r[
                        "optimistic_cameras_plus_range_max_rays"]["free"]
                        for r in group),
                }
    report = {
        "schema": "robot-learning-lab.ape-tof-zone-shell-screen.v1",
        "source_sha256": sha(SOURCE),
        "v181_report_sha256": sha(source_file),
        "range_model_source_sha256": sha(Path(range_model.__file__)),
        "footprints": FOOTPRINTS,
        "zone_center_grid": "8x8 square, evenly centered across the "
                            "published horizontal/vertical detection FoV",
        "noise_sigma_probes_m": NOISE_PROBES_M,
        "dropout_fraction": range_model.DROPOUT_FRACTION,
        "range_limit_m": range_model.RANGE_MAX_M,
        "mounts_body_xyz_yaw_rad": MOUNTS,
        "scope": "One 16.2s virtual first-hit packet per candidate in each of "
                 "eight matched historical held poses. ST VL53L8CX 45x45 "
                 "and VL53L7CX 60x60 nominal 8x8 zone centers replace "
                 "V186's 31x21 110x80 rays. This center-ray proxy omits "
                 "finite zone footprint, multitarget mixing, status codes, "
                 "ambient/reflectance failures and package fit. Gaussian "
                 "5/12mm stress and 15% dropout are illustrative, not "
                 "datasheet performance distributions. Hit-only and "
                 "optimistic max-distance-ray free space stay separate. "
                 "No physical sensor selection or motion permission.",
        "summary": summary,
        "rows": rows,
    }
    (OUT / "report.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
