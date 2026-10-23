"""Locate V129p's residual raised-arm retreat samples and exact-point LOS.

Exact-point LOS is a diagnosis of mount geometry, not a voxel-free proof.
"""

from __future__ import annotations

import json
import math
from pathlib import Path

import numpy as np

from articulated_person_scene import START_HAND_XYZ_M, add_person, set_person_pose
import audit_blocked_range_mounts_v319 as sensor
from audit_pre_show_unknown_los_v116 import (CANDIDATE_MOUNTS,
                                             actual_point, line_of_sight)
import direct_structural_rooms_v270 as structural
from depth_voxel import DepthVoxelMap
from direct_structural_rooms_v270 import selected_visual
from pre_show_actor_planner import ROUTE_FREE_AGE_S, freshness_filtered
from robot_geometry import RobotGeometry
from run import ROOT, sha
from run_articulated_owner_low_drift_v92 import OfferDriftRoom
from run_moving_low_clutter_v302 import STOOL_SIZE
import screen_early_retreat_observability_v129p as v129p
import validate_early_meeting_v128p as v128p


SOURCE = Path(__file__)
OUT = ROOT / "artifacts/ape-closed-loop-cove50-early-retreat-observability/unknown-cells.json"
LAYOUT = "yaw+1.4-outward_dense_3d_rings"
RAY_NAMES = v129p.LAYOUTS["outward_dense_3d_rings"]
MOUNTS = {name: CANDIDATE_MOUNTS[name] for name in (
    "rear_low_near", "rear_mid_near", "rear_high_mid",
    "left_low", "left_upper", "right_low", "right_upper",
    "rear_low_inward_probe", "left_low_inward_probe",
    "right_upper_inward_probe", "rear_low_omni_bound")}


def unknown_samples(robot: RobotGeometry, voxel_map: DepthVoxelMap,
                    qpos: np.ndarray, pose: np.ndarray,
                    offset_xy: tuple[float, float],
                    z_min: float, z_max: float) -> tuple[list[dict], dict]:
    current = pose.copy()
    future = pose.copy()
    future[:2] += offset_xy
    heading = np.array([*offset_xy, 0.]) / np.linalg.norm(offset_xy)
    visited = set()
    rows = []
    for fraction in (.25, .5, .75, 1.):
        candidate = current + fraction*(future-current)
        robot.set_state(robot.pose_qpos(qpos, candidate))
        solids = robot.primitives()
        for point, normal in robot.surface(solids):
            if point[2] < z_min or point[2] > z_max:
                continue
            if float(normal@heading) < .2:
                continue
            if robot.solid_distance(point, solids) < -.005:
                continue
            sample = point+.05*normal
            cell = voxel_map.cell(sample)
            if cell in visited:
                continue
            visited.add(cell)
            rows.append({"cell": cell, "sample": sample,
                         "state": voxel_map.state(sample, v129p.DECISION_S)})
    reference = robot.swept_exterior_coverage(
        voxel_map, qpos, current, future, v129p.DECISION_S,
        margin_m=.05, z_min_m=z_min, z_max_m=z_max)
    for status in ("free", "occupied", "unknown"):
        assert sum(row["state"] == status for row in rows) == reference[status]
    return [row for row in rows if row["state"] == "unknown"], reference


def main() -> None:
    v129 = json.loads((v129p.OUT / "report.json").read_text())
    v98 = json.loads((v128p.V98 /
                      "live-moving-owner-offer-audit.json").read_text())
    original_model = structural.structural_model
    structural.structural_model = lambda kind, side: add_person(
        original_model(kind, side))
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
                estimated = packet["estimated_pose_xyyaw"].copy()
            room = OfferDriftRoom(case=tuple(prior["case"]),
                                  pose_source="command", branch=OUT.parent)
            try:
                room.layout(seed, absent=False, furnished=True)
                room.data.qpos[:] = qpos
                room.actual[:] = (qpos[0], qpos[1],
                                  2*math.atan2(qpos[6], qpos[3]))
                room.estimated[:] = estimated
                room.head = 1.4
                room.time_s = v129p.DECISION_S
                set_person_pose(room.model, START_HAND_XYZ_M,
                                carried_item="loop_TomatoSauce")
                room.obstacle_xy[3] = (0., .306)
                room.model.body_pos[room.model.body(
                    "loop_stool").id] = (0., .306, STOOL_SIZE[2])
                room._forward()
                head = record["head_packets"]["1.4"]
                assert sha(ROOT / head["path"]) == head["sha256"]
                voxel_map = DepthVoxelMap()
                with np.load(ROOT / head["path"]) as packet:
                    voxel_map.observe(packet["filtered_depth_m"],
                                      (packet["camera_origin_estimated_m"],
                                       packet["camera_rotation_estimated"]),
                                      v129p.DECISION_S,
                                      fovy_deg=head["fovy_deg"], stride=4,
                                      max_depth_m=1.4, self_masked=True)
                rays_path = ROOT / record["range_packet_path"]
                assert sha(rays_path) == record["range_packet_sha256"]
                ray_packets = json.loads(rays_path.read_text())[
                    "ray_packets"]
                for name in RAY_NAMES:
                    sensor.add_packet(voxel_map, ray_packets[name],
                                      v129p.DECISION_S)
                fresh = freshness_filtered(
                    voxel_map, v129p.DECISION_S, ROUTE_FREE_AGE_S)
                robot = RobotGeometry(room.model)
                qpos[room.model.joint("left_shoulder_pitch").qposadr[0]] = -.6
                unknown_by_height = {}
                for height, lo, hi in (("lower", .02, .18),
                                       ("upper", .18, .43)):
                    unknown, coverage = unknown_samples(
                        robot, fresh, qpos, estimated,
                        v128p.STATIONS["back12"], lo, hi)
                    saved = record["alternatives"][LAYOUT]["routes"][
                        "back12"]["pose_posture_samples"][
                        "left_raised:estimated"][height]
                    assert coverage == saved
                    items = []
                    for item in unknown:
                        actual = actual_point(item["sample"], estimated,
                                              room.actual)
                        sight = {name: line_of_sight(
                            room, actual, mount)["status"]
                                 for name, mount in MOUNTS.items()}
                        items.append({
                            "voxel_cell": [int(x) for x in item["cell"]],
                            "actor_sample_xyz_m": np.round(
                                item["sample"], 5).tolist(),
                            "mount_sample_los_status": sight})
                    unknown_by_height[height] = {
                        "coverage": coverage,
                        "unknown_samples": items,
                        "candidate_mount_clear_sample_counts": {
                            name: sum(x["mount_sample_los_status"][name] ==
                                      "sample_clear_los" for x in items)
                            for name in MOUNTS}}
                rows.append({"seed": seed,
                             "source_packet_path": record["source_packet_path"],
                             "unknown_by_height": unknown_by_height})
                print(seed, {height: len(entry["unknown_samples"])
                             for height, entry in unknown_by_height.items()},
                      flush=True)
            finally:
                room.close()
    finally:
        structural.structural_model = original_model
    report = {
        "schema": "robot-learning-lab.ape-early-retreat-unknown-cells.v1",
        "source_sha256": sha(SOURCE),
        "v129_report_sha256": sha(v129p.OUT / "report.json"),
        "selected_visual_sha256": sha(selected_visual()),
        "scope": "Reconstruct V129p's residual raised-left-arm 12cm "
        "retreat shell samples for the richest virtual sensor layout. "
        "First-hit exact-point LOS uses evaluator actual pose in the "
        "frozen start scene; it is not a complete voxel-visibility bound, "
        "sensor model, actor map or route authorization.",
        "rows": rows}
    OUT.write_text(json.dumps(report, indent=2)+"\n")
    print(OUT)


if __name__ == "__main__":
    main()
