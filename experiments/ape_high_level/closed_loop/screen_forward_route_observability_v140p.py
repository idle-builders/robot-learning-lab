"""V140p: actor-map screen for the front-visible meeting station.

The camera and ray packets are actor evidence at8.0s; known simulator
clearance is excluded from the route decision.
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
from run import ROOT, sha
from run_articulated_owner_low_drift_v92 import OfferDriftRoom
from run_moving_low_clutter_v302 import STOOL_SIZE
from run_pre_show_sensor_coverage_v102 import (FOV_H_DEG, FOV_V_DEG,
                                                GRID_H, GRID_V, MOUNTS)
import screen_early_retreat_observability_v129p as v129p
import screen_forward_meeting_v139p as v139p
import validate_early_meeting_v128p as v128p


SOURCE = Path(__file__)
OUT = ROOT / "artifacts/ape-closed-loop-cove50-forward-route-observability"
STATIONS = {"front12": (.12, 0.), "front16_right4": (.16, -.04)}
RANGE_LAYOUTS = {
    "camera_only": (),
    "front_low": ("front_low",),
    "front_low_upper": ("front_low", "front_upper"),
    "front_low_upper_side": ("front_low", "front_upper", "right_upper"),
}


def camera_map(meta: dict) -> DepthVoxelMap:
    path = ROOT / meta["path"]
    assert sha(path) == meta["sha256"]
    voxel_map = DepthVoxelMap()
    with np.load(path) as packet:
        voxel_map.observe(packet["filtered_depth_m"],
                          (packet["camera_origin_estimated_m"],
                           packet["camera_rotation_estimated"]),
                          v129p.DECISION_S, fovy_deg=meta["fovy_deg"],
                          stride=4, max_depth_m=1.4, self_masked=True)
    return voxel_map


def route(robot: RobotGeometry, voxel_map: DepthVoxelMap,
          qpos: np.ndarray, pose: np.ndarray,
          offset_xy: tuple[float, float],
          perturbations: dict[str, tuple[float, float, float]]) -> dict:
    shoulder = robot.model.joint("left_shoulder_pitch").qposadr[0]
    variants = {}
    for posture, angle in (("neutral", 0.), ("left_forward", .6)):
        posture_qpos = qpos.copy()
        posture_qpos[shoulder] = angle
        for label, perturbation in perturbations.items():
            start = pose + np.asarray(perturbation)
            end = start.copy()
            end[:2] += offset_xy
            variants[f"{posture}:{label}"] = {
                height: robot.swept_exterior_coverage(
                    voxel_map, posture_qpos, start, end,
                    v129p.DECISION_S, margin_m=.05,
                    z_min_m=lo, z_max_m=hi)
                for height, lo, hi in (("lower", .02, .18),
                                       ("upper", .18, .43))}
    statuses = [band["status"] for sample in variants.values()
                for band in sample.values()]
    return {"coverage_status": (
                "observed_free" if all(x == "free" for x in statuses) else
                "occupied" if "occupied" in statuses else "unknown"),
            "pose_posture_samples": variants}


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    v129 = json.loads((v129p.OUT / "report.json").read_text())
    v98 = json.loads((v128p.V98 /
                      "live-moving-owner-offer-audit.json").read_text())
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
    sensor.HORIZONTAL_DEG = np.linspace(-FOV_H_DEG/2, FOV_H_DEG/2, GRID_H)
    sensor.VERTICAL_DEG = np.linspace(-FOV_V_DEG/2, FOV_V_DEG/2, GRID_V)
    sensor.RANGE_MAX_M = .60
    sensor.RANGE_NOISE_STD_M = .005
    sensor.DROPOUT_FRACTION = .15
    rows = []
    try:
        for record in v129["rows"]:
            seed = record["seed"]
            prior = next(row for row in v98["rows"]
                         if row["condition"] == "moving" and
                         row["case"][0] == seed)
            source = ROOT / record["source_packet_path"]
            assert sha(source) == record["source_packet_sha256"]
            with np.load(source) as packet:
                qpos = packet["actor_qpos"].copy()
                pose = packet["estimated_pose_xyyaw"].copy()
            room = OfferDriftRoom(case=tuple(prior["case"]),
                                  pose_source="command", branch=OUT)
            try:
                room.layout(seed, absent=False, furnished=True)
                room.data.qpos[:] = qpos
                room.actual[:] = (qpos[0], qpos[1],
                                  2*math.atan2(qpos[6], qpos[3]))
                room.estimated[:] = pose
                room.head = 0.
                room.time_s = v129p.DECISION_S
                set_person_pose(room.model, START_HAND_XYZ_M,
                                carried_item="loop_TomatoSauce")
                room.obstacle_xy[3] = (0., .306)
                room.model.body_pos[room.model.body(
                    "loop_stool").id] = (0., .306, STOOL_SIZE[2])
                room._forward()
                rays = {}
                counts = {}
                for name in ("front_low", "front_upper", "right_upper"):
                    rays[name], counts[name] = sensor.range_packet(
                        room, pose, name, seed, v129p.DECISION_S)
                packet_path = OUT / f"seed{seed}-front-rays.json"
                packet_path.write_text(json.dumps({
                    "capture_time_s": v129p.DECISION_S,
                    "ray_packets": rays, "counts": counts}, indent=2)+"\n")
                robot = RobotGeometry(room.model)
                alternatives = {}
                for angle in (0., 1.4):
                    camera = record["head_packets"][str(angle)]
                    for layout, mounts in RANGE_LAYOUTS.items():
                        voxel_map = camera_map(camera)
                        for name in mounts:
                            sensor.add_packet(voxel_map, rays[name],
                                              v129p.DECISION_S)
                        fresh = freshness_filtered(
                            voxel_map, v129p.DECISION_S,
                            ROUTE_FREE_AGE_S)
                        routes = {}
                        for station, offset in STATIONS.items():
                            quick = route(robot, fresh, qpos, pose, offset,
                                          {"estimated": (0., 0., 0.)})
                            routes[station] = (route(robot, fresh, qpos, pose,
                                offset, POSE_OFFSETS)
                                if quick["coverage_status"] == "observed_free"
                                else quick)
                        key = f"yaw{angle:+.1f}-{layout}"
                        alternatives[key] = {"head_yaw_rad": angle,
                                             "range_layout": layout,
                                             "routes": routes}
                rows.append({"seed": seed,
                             "source_packet_path": record[
                                 "source_packet_path"],
                             "new_packet_path": str(packet_path.relative_to(ROOT)),
                             "new_packet_sha256": sha(packet_path),
                             "range_counts": counts,
                             "alternatives": alternatives})
                supported = [(key, station)
                             for key, item in alternatives.items()
                             for station, route_result in item["routes"].items()
                             if route_result["coverage_status"] == "observed_free"]
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
    report = {"schema": "robot-learning-lab.ape-forward-route-observability.v1",
              "source_sha256": sha(SOURCE),
              "v129_report_sha256": sha(v129p.OUT / "report.json"),
              "v139_report_sha256": sha(v139p.OUT / "screen.json"),
              "selected_visual_sha256": sha(selected_visual()),
              "scope": "Actor-map screen at8.0s for two evaluator-promising "
              "forward owner stations. Saved fresh lower RGB-D at neutral "
              "or +1.4rad head yaw is combined with noisy mounted "
              "front-low/upper and side ray arrays. Neutral and +0.6rad "
              "left-arm 5cm-buffered swept shells require observed free "
              "at estimated pose and, if that passes, illustrative pose "
              "offsets. No actor move, physical mount or future person "
              "occupancy claim is made.",
              "rows": rows}
    (OUT / "report.json").write_text(json.dumps(report, indent=2)+"\n")
    print(OUT / "report.json")


if __name__ == "__main__":
    main()
