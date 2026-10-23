"""V118: compact 8x8 ToF and close stereo proxies versus V117 dense rays.

The 8x8/45x45 geometry follows ST's VL53L8CX detection volume; each zone is
represented by one central first-hit ray, not the device's histogram or zone
response. The 32x24/87x58 short-stereo proxy follows D405 field/min-range,
but omits stereo matching failures. No product or sensor count is selected.
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
from run_inward_low_mount_packets_v117 import (BASE, V102, V109,
                                                 route_coverage)
from run_moving_low_clutter_v302 import STOOL_SIZE
from run_owner_exit_height_v96 import CASES
from run_pre_show_reserve_screen_v100 import START_S, V98
from run_pre_show_sensor_coverage_v102 import MOUNTS


SOURCE = Path(__file__)
OUT = ROOT / "artifacts/ape-closed-loop-cove50-v118"
V117 = ROOT / "artifacts/ape-closed-loop-cove50-v117"
ST_TOF_DATASHEET = "https://www.st.com/resource/en/datasheet/vl53l8cx.pdf"
D405_PAGE = "https://www.realsenseai.com/products/stereo-depth-camera-d405/"
SIDE_ORIGINS = {"left": MOUNTS["left_low"][0],
                "right": MOUNTS["right_low"][0]}
PROXY_MOUNTS = {
    f"{side}_tof8_level": (offset, -math.pi/2 if side == "left"
                          else math.pi/2)
    for side, offset in SIDE_ORIGINS.items()
}
PROXY_MOUNTS.update({
    f"{side}_tof8_up": (offset, -math.pi/2 if side == "left"
                       else math.pi/2)
    for side, offset in SIDE_ORIGINS.items()
})
PROXY_MOUNTS.update({
    f"{side}_short_stereo_proxy": (offset, -math.pi/2 if side == "left"
                                   else math.pi/2)
    for side, offset in SIDE_ORIGINS.items()
})
PROXY_MOUNTS.update({
    f"{side}_wide8_diagnostic": (offset, -math.pi/2 if side == "left"
                                 else math.pi/2)
    for side, offset in SIDE_ORIGINS.items()
})
PROXY_MOUNTS.update({
    f"{side}_narrow_dense_diagnostic": (offset, -math.pi/2 if side == "left"
                                        else math.pi/2)
    for side, offset in SIDE_ORIGINS.items()
})
LAYOUTS = {
    "outward_baseline": (),
    "tof8_pair_level": ("left_tof8_level", "right_tof8_level"),
    "tof8_four_level_up": ("left_tof8_level", "right_tof8_level",
                            "left_tof8_up", "right_tof8_up"),
    "short_stereo_pair_proxy": ("left_short_stereo_proxy",
                                "right_short_stereo_proxy"),
    "wide8_pair_diagnostic": ("left_wide8_diagnostic",
                              "right_wide8_diagnostic"),
    "narrow_dense_pair_diagnostic": ("left_narrow_dense_diagnostic",
                                     "right_narrow_dense_diagnostic"),
    "v117_dense_inward_pair": ("left_low_inward", "right_low_inward"),
}


def configure_and_scan(room: OfferDriftRoom, pose: np.ndarray,
                       seed: int, name: str) -> tuple[list[dict], dict]:
    if "tof8" in name or "wide8_diagnostic" in name or "narrow_dense_diagnostic" in name:
        # ST's measured detection volume is 45 x 45 degrees. Ray centers
        # represent eight equal zones, leaving half-zone margins.
        if "wide8_diagnostic" in name:
            sensor.HORIZONTAL_DEG = (np.arange(8)+.5)*110./8.-55.
            sensor.VERTICAL_DEG = (np.arange(8)+.5)*80./8.-40.
        elif "narrow_dense_diagnostic" in name:
            sensor.HORIZONTAL_DEG = (np.arange(31)+.5)*45./31.-22.5
            sensor.VERTICAL_DEG = (np.arange(21)+.5)*45./21.-22.5
        else:
            centers = (np.arange(8)+.5)*45./8.-22.5
            sensor.HORIZONTAL_DEG = centers
            sensor.VERTICAL_DEG = centers + (22.5 if name.endswith("_up") else 0.)
        sensor.RANGE_MAX_M = .60  # matched near-body planning horizon
        sensor.RANGE_NOISE_STD_M = .012  # ~ST 8x8 short-range accuracy scale
        sensor.DROPOUT_FRACTION = .15
    else:
        # An optimistic lower-resolution ray proxy for D405 depth geometry;
        # actual stereo texture, IR and occlusion behavior is not simulated.
        sensor.HORIZONTAL_DEG = (np.arange(32)+.5)*87./32.-43.5
        sensor.VERTICAL_DEG = (np.arange(24)+.5)*58./24.-29.
        sensor.RANGE_MAX_M = .50
        sensor.RANGE_NOISE_STD_M = .01  # 2% at 50 cm
        sensor.DROPOUT_FRACTION = .15
    rays, counts = sensor.range_packet(room, pose, name, seed, START_S)
    if "stereo" in name:
        # D405's published 7 cm Min-Z. These center rays already reject
        # self-first hits in range_packet; this is a further near-range gate.
        rays = [ray for ray in rays
                if ray["measured_range_m"] >= .07]
        counts["valid_after_7cm_min"] = len(rays)
    return rays, counts


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    v98 = json.loads((V98 / "live-moving-owner-offer-audit.json").read_text())
    v117 = json.loads((V117 / "report.json").read_text())
    old_model = structural.structural_model
    old_mounts = sensor.MOUNTS.copy()
    old_h = sensor.HORIZONTAL_DEG.copy()
    old_v = sensor.VERTICAL_DEG.copy()
    old_range = sensor.RANGE_MAX_M
    old_noise = sensor.RANGE_NOISE_STD_M
    old_dropout = sensor.DROPOUT_FRACTION
    structural.structural_model = lambda kind, side: add_person(
        old_model(kind, side))
    sensor.MOUNTS.clear()
    sensor.MOUNTS.update(MOUNTS)
    sensor.MOUNTS.update(PROXY_MOUNTS)
    rows = []
    try:
        for seed in (170, 172):
            prior = next(row for row in v98["rows"]
                         if row["condition"] == "moving" and
                         row["case"][0] == seed)
            for case in ("clear", "hand_close"):
                previous = next(row for row in v117["rows"]
                                if row["seed"] == seed and
                                row["case"] == case)
                actor_path = ROOT / previous["v102_actor_packet_path"]
                if sha(actor_path) != previous["v102_actor_packet_sha256"]:
                    raise AssertionError("V102 actor packet changed")
                actor = json.loads(actor_path.read_text())
                dense_path = ROOT / previous["new_actor_packet_path"]
                if sha(dense_path) != previous["new_actor_packet_sha256"]:
                    raise AssertionError("V117 dense packet changed")
                dense = json.loads(dense_path.read_text())["ray_packets"]
                head_path = ROOT / previous["v109_head_packet_path"]
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
                                      2*math.atan2(qpos[6],qpos[3]))
                    room.estimated[:] = pose
                    room.head = float(qpos[room.head_addr])
                    room.time_s = START_S
                    set_person_pose(room.model, START_HAND_XYZ_M,
                                    carried_item="loop_TomatoSauce")
                    room.obstacle_xy[3] = (0., .306)
                    room.model.body_pos[room.model.body(
                        "loop_stool").id] = (0., .306, STOOL_SIZE[2])
                    room._forward()
                    new_rays = {}
                    new_counts = {}
                    for name in PROXY_MOUNTS:
                        new_rays[name], new_counts[name] = (
                            configure_and_scan(room, pose, seed, name))
                    fovy = float(room.model.cam_fovy[
                        room.model.camera("lower_rgb").id])
                    base = DepthVoxelMap()
                    base.observe(depth, camera, START_S,
                                 fovy_deg=fovy, stride=4,
                                 max_depth_m=1.4, self_masked=True)
                    robot = RobotGeometry(room.model)
                    coverage = {}
                    for layout, added in LAYOUTS.items():
                        voxel_map = DepthVoxelMap()
                        voxel_map.free = base.free.copy()
                        voxel_map.occupied = base.occupied.copy()
                        voxel_map.last_time_s = START_S
                        for name in BASE:
                            sensor.add_packet(voxel_map,
                                              actor["ray_packets"][name],
                                              START_S)
                        for name in added:
                            rays = dense[name] if name in dense else new_rays[name]
                            sensor.add_packet(voxel_map, rays, START_S)
                        coverage[layout] = route_coverage(
                            robot, voxel_map, qpos, pose)
                    if coverage["outward_baseline"] != previous[
                            "coverage"]["outward_baseline"]:
                        raise AssertionError("V117 baseline changed")
                    if coverage["v117_dense_inward_pair"] != previous[
                            "coverage"]["plus_inward_low_pair"]:
                        raise AssertionError("V117 dense comparison changed")
                    branch = OUT / f"seed{seed}-{case}"
                    branch.mkdir(parents=True, exist_ok=True)
                    packet_path = branch / "compact-proxy-actor-rays.json"
                    packet_path.write_text(json.dumps({
                        "capture_time_s": START_S,
                        "estimated_pose_xyyaw": pose.tolist(),
                        "proxy_mounts": PROXY_MOUNTS,
                        "ray_packets": new_rays,
                    }, indent=2)+"\n")
                    rows.append({
                        "seed": seed, "case": case,
                        "v102_actor_packet_path": str(actor_path.relative_to(ROOT)),
                        "v102_actor_packet_sha256": sha(actor_path),
                        "v117_dense_packet_path": str(dense_path.relative_to(ROOT)),
                        "v117_dense_packet_sha256": sha(dense_path),
                        "v109_head_packet_path": str(head_path.relative_to(ROOT)),
                        "v109_head_packet_sha256": sha(head_path),
                        "compact_actor_packet_path": str(packet_path.relative_to(ROOT)),
                        "compact_actor_packet_sha256": sha(packet_path),
                        "new_packet_counts": new_counts,
                        "coverage": coverage,
                    })
                finally:
                    room.close()
    finally:
        structural.structural_model = old_model
        sensor.MOUNTS.clear()
        sensor.MOUNTS.update(old_mounts)
        sensor.HORIZONTAL_DEG = old_h
        sensor.VERTICAL_DEG = old_v
        sensor.RANGE_MAX_M = old_range
        sensor.RANGE_NOISE_STD_M = old_noise
        sensor.DROPOUT_FRACTION = old_dropout
    report = {
        "schema": "robot-learning-lab.ape-compact-near-sensor-proxies.v1",
        "source_sha256": sha(SOURCE),
        "v117_report_sha256": sha(V117 / "report.json"),
        "selected_visual_sha256": sha(selected_visual()),
        "reference_specs": {
            "st_vl53l8cx": {"url": ST_TOF_DATASHEET,
                "zones": [8,8], "detection_fov_deg": [45,45],
                "8x8_continuous_rate_hz": 15,
                "short_range_accuracy_scale_m": .012},
            "realsense_d405": {"url": D405_PAGE,
                "depth_fov_deg": [87,58], "ideal_range_m": [.07,.50],
                "depth_accuracy_at_50cm_fraction": .02,
                "module_dimensions_mm": [36.5,19.4,10.5]},
        },
        "proxy_limits": "ToF zone center rays omit finite zone response, multi-target returns, ambient/reflectance effects and cover-glass crosstalk. The wide8 and narrow-dense patterns are diagnostics to separate angular sampling from FoV, not devices. Short-stereo 32x24 rays omit stereo matching/texture and USB/compute load; 15% dropout is an experiment assumption. Same historical body and one fixed pose/time. Published rates are not simulated timing. No hardware selection or movement authorization.",
        "layouts": LAYOUTS,
        "rows": rows,
    }
    path = OUT / "report.json"
    path.write_text(json.dumps(report, indent=2)+"\n")
    for row in rows:
        print(row["seed"], row["case"], {
            name: (steps[0]["upper"]["unknown"],
                   steps[0]["lower"]["unknown"])
            for name, steps in row["coverage"].items()})
    print(path)


if __name__ == "__main__":
    main()
