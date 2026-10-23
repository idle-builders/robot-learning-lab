"""V135p: targeted rear-left upward 8x8 range probes for early retreat.

The residual V133p cells lie behind and left of the body, above common low
range fields. These virtual first-hit packets test whether a directed mount
fills that gap; they do not model actual ToF optics or packaged hardware.
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
import screen_early_retreat_observability_v129p as v129p
import validate_early_meeting_v128p as v128p


SOURCE = Path(__file__)
OUT = ROOT / "artifacts/ape-closed-loop-cove50-rear-left-up-probe"
PROBES = {
    "rear_low_left_up8": ((-.07, 0., .10), math.radians(120.), 55.),
    "rear_mid_left_up8": ((-.07, 0., .23), math.radians(120.), 35.),
}
BASE_LAYOUTS = {
    "outward_dense": v129p.LAYOUTS["outward_dense"],
    "outward_dense_3d_rings": v129p.LAYOUTS[
        "outward_dense_3d_rings"],
}


def head_map(record: dict) -> DepthVoxelMap:
    meta = record["head_packets"]["1.4"]
    path = ROOT / meta["path"]
    assert sha(path) == meta["sha256"]
    result = DepthVoxelMap()
    with np.load(path) as packet:
        result.observe(packet["filtered_depth_m"],
                       (packet["camera_origin_estimated_m"],
                        packet["camera_rotation_estimated"]),
                       v129p.DECISION_S, fovy_deg=meta["fovy_deg"],
                       stride=4, max_depth_m=1.4, self_masked=True)
    return result


def route(robot: RobotGeometry, voxel_map: DepthVoxelMap,
          qpos: np.ndarray, pose: np.ndarray,
          pose_offsets: dict[str, tuple[float, float, float]]) -> dict:
    result = {}
    shoulder = robot.model.joint("left_shoulder_pitch").qposadr[0]
    for posture, angle in (("neutral", 0.), ("left_forward", .6)):
        posture_qpos = qpos.copy()
        posture_qpos[shoulder] = angle
        for label, perturbation in pose_offsets.items():
            start = pose + np.asarray(perturbation)
            end = start.copy()
            end[:2] += v128p.STATIONS["back12"]
            result[f"{posture}:{label}"] = {
                height: robot.swept_exterior_coverage(
                    voxel_map, posture_qpos, start, end, v129p.DECISION_S,
                    margin_m=.05, z_min_m=lo, z_max_m=hi)
                for height, lo, hi in (("lower", .02, .18),
                                       ("upper", .18, .43))}
    statuses = [band["status"] for sample in result.values()
                for band in sample.values()]
    return {"coverage_status": (
                "observed_free" if all(x == "free" for x in statuses) else
                "occupied" if "occupied" in statuses else "unknown"),
            "pose_posture_samples": result}


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
    sensor.MOUNTS.update(v129p.MOUNTS)
    sensor.MOUNTS.update({name: (offset, bearing)
                          for name, (offset, bearing, _) in PROBES.items()})
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
                room.head = 1.4
                room.time_s = v129p.DECISION_S
                set_person_pose(room.model, START_HAND_XYZ_M,
                                carried_item="loop_TomatoSauce")
                room.obstacle_xy[3] = (0., .306)
                room.model.body_pos[room.model.body(
                    "loop_stool").id] = (0., .306, STOOL_SIZE[2])
                room._forward()
                robot = RobotGeometry(room.model)
                rays = json.loads((ROOT / record[
                    "range_packet_path"]).read_text())["ray_packets"]
                counts = {}
                sensor.HORIZONTAL_DEG = np.linspace(-22.5, 22.5, 8)
                sensor.RANGE_MAX_M = .60
                sensor.RANGE_NOISE_STD_M = .012
                sensor.DROPOUT_FRACTION = .15
                for name, (_, _, pitch) in PROBES.items():
                    sensor.VERTICAL_DEG = np.linspace(pitch-22.5,
                                                      pitch+22.5, 8)
                    rays[name], counts[name] = sensor.range_packet(
                        room, pose, name, seed, v129p.DECISION_S)
                packet_path = OUT / f"seed{seed}-probe-rays.json"
                packet_path.write_text(json.dumps({
                    "capture_time_s": v129p.DECISION_S,
                    "ray_packets": {name: rays[name] for name in PROBES},
                    "counts": counts}, indent=2)+"\n")
                layouts = {}
                for baseline, baseline_names in BASE_LAYOUTS.items():
                    for probe in ("none", *PROBES):
                        voxel_map = head_map(record)
                        for name in (*baseline_names,
                                     *((probe,) if probe != "none" else ())):
                            sensor.add_packet(voxel_map, rays[name],
                                              v129p.DECISION_S)
                        fresh = freshness_filtered(
                            voxel_map, v129p.DECISION_S,
                            ROUTE_FREE_AGE_S)
                        quick = route(robot, fresh, qpos, pose,
                                      {"estimated": (0., 0., 0.)})
                        result = (route(robot, fresh, qpos, pose, POSE_OFFSETS)
                                  if quick["coverage_status"] == "observed_free"
                                  else quick)
                        layouts[f"{baseline}+{probe}"] = result
                        plus = result["pose_posture_samples"][
                            "left_forward:estimated"]
                        print(seed, baseline, probe,
                              "plus", plus["upper"]["unknown"],
                              plus["lower"]["unknown"],
                              "status", result["coverage_status"], flush=True)
                rows.append({"seed": seed,
                             "source_packet_path": record["source_packet_path"],
                             "new_packet_path": str(packet_path.relative_to(ROOT)),
                             "new_packet_sha256": sha(packet_path),
                             "probe_counts": counts,
                             "layouts": layouts})
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
    report = {"schema": "robot-learning-lab.ape-rear-left-up-probe.v1",
              "source_sha256": sha(SOURCE),
              "v129_report_sha256": sha(v129p.OUT / "report.json"),
              "selected_visual_sha256": sha(selected_visual()),
              "probe_mounts_offset_yaw_pitch": PROBES,
              "scope": "Two virtual directed8x8 first-hit range probes at "
              "rear-low and rear-mid historical solid offsets. 45deg "
              "horizontal/vertical field,60cm maximum,12mm Gaussian noise "
              "and15% dropout. Added to saved8.0s head/outward/inward "
              "packets with or without V129p's optimistic three-origin "
              "multi-elevation rings. Both neutral and +0.6rad left-arm "
              "retreat shells require observed free. No actual ToF optical "
              "model, mount packaging, timing or selected body is claimed.",
              "rows": rows}
    (OUT / "report.json").write_text(json.dumps(report, indent=2)+"\n")
    print(OUT / "report.json")


if __name__ == "__main__":
    main()
