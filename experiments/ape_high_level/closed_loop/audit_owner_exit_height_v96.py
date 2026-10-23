"""Audit V96 saved actor rays, paired images and 20ms side-step geometry."""

from __future__ import annotations

import json
import math
from pathlib import Path

import imageio.v2 as imageio
import numpy as np
from PIL import Image

from articulated_person_scene import FAR_HAND_XYZ_M, add_person, set_person_pose
from depth_voxel import DepthVoxelMap
import direct_structural_rooms_v270 as structural
from direct_structural_rooms_v270 import selected_visual
from robot_geometry import RobotGeometry
from run import ROOT, sha
from run_articulated_owner_low_drift_v92 import OfferDriftRoom
from run_owner_exit_height_v96 import (ACTUAL_SLIP, CASES, DECISION_TIME_S,
                                       OUT, RANGE_MOUNTS, SIDE_STEP_M, SOURCE,
                                       V95)
from audit_blocked_range_mounts_v319 import add_packet


AUDIT_SOURCE = Path(__file__)


def main() -> None:
    report = json.loads((OUT / "report.json").read_text())
    assert report["source_sha256"] == sha(SOURCE)
    assert report["selected_visual_sha256"] == sha(selected_visual())
    assert report["v95_report_sha256"] == sha(
        V95 / "live-moving-owner-offer-audit.json")
    assert len(report["rows"]) == 8
    checks = 4
    rows = {(row["seed"], row["case"]): row for row in report["rows"]}
    assert set(rows) == {(seed, name) for seed in (170, 172) for name in CASES}
    checks += 1
    v95_report = json.loads((V95 / "live-moving-owner-offer-audit.json").read_text())
    old_model = structural.structural_model
    structural.structural_model = lambda kind, side: add_person(
        old_model(kind, side))
    results = []
    try:
        for seed in (170, 172):
            original = next(row for row in v95_report["rows"]
                            if row["condition"] == "moving" and
                            row["case"][0] == seed)
            source = (V95 / "moving" / f"seed{seed}-present" /
                      f"seed{seed}-present/owner-guard-04.npz")
            with np.load(source) as packet:
                source_qpos = packet["actor_qpos"].copy()
                source_pose = packet["estimated_pose_xyyaw"].copy()
                assert abs(float(packet["capture_time_s"]) -
                           DECISION_TIME_S) < 1e-5
                checks += 1
            for name, obstacle in CASES.items():
                row = rows[(seed, name)]
                packet_path = ROOT / row["actor_packet_path"]
                assert sha(packet_path) == row["actor_packet_sha256"]
                actor = json.loads(packet_path.read_text())
                assert set(actor["ray_packets"]) == set(RANGE_MOUNTS)
                assert np.allclose(actor["actor_qpos"], source_qpos)
                assert np.allclose(actor["estimated_pose_xyyaw"], source_pose)
                assert abs(actor["capture_time_s"] - DECISION_TIME_S) < 1e-5
                assert "evaluator_gap_m" not in actor
                checks += 6
                room = OfferDriftRoom(case=tuple(original["case"]),
                                      pose_source="command", branch=OUT / "audit")
                try:
                    room.layout(seed, absent=False, furnished=True)
                    if obstacle is not None:
                        position, size = obstacle
                        room.model.body_pos[room.model.body("loop_box").id] = position
                        room.model.geom("loop_box_geom").size[:] = size
                    qpos = source_qpos.copy()
                    estimated = source_pose.copy()
                    actual = np.array((qpos[0], qpos[1],
                                       2*math.atan2(qpos[6], qpos[3])))
                    room.data.qpos[:] = qpos
                    room.actual[:] = actual
                    room.estimated[:] = estimated
                    room.head = float(qpos[room.head_addr])
                    room.time_s = DECISION_TIME_S
                    room._update_moving_obstacle()
                    set_person_pose(room.model, FAR_HAND_XYZ_M,
                                    carried_item="loop_TomatoSauce")
                    room._forward()
                    assert abs(room._current_obstacle_clearance()[0] -
                               row["initial_scene_gap_m"]) < .0001
                    checks += 1
                    robot = RobotGeometry(room.model)
                    right = np.asarray(actor["right_rear_points_xyz_m"],
                                       dtype=float)
                    assert len(right) == row["right_rear_points"]
                    minimum_point = float("inf")
                    for fraction in np.linspace(0., 1., 9):
                        candidate = estimated.copy()
                        candidate[1] -= SIDE_STEP_M*fraction
                        robot.set_state(robot.pose_qpos(qpos, candidate))
                        minimum_point = min(minimum_point, float(np.min(
                            robot.solid_distances(right,
                                                  robot.primitives()))))
                    assert abs(minimum_point -
                               row["right_rear_swept_point_gap_m"]) < .0001
                    assert bool(len(right) >= 8 and minimum_point >= .06) == (
                        row["v95_point_gate_allows"])
                    checks += 3
                    future = estimated.copy()
                    future[1] -= SIDE_STEP_M
                    for mount, payload in actor["ray_packets"].items():
                        rays = payload["rays"]
                        assert all(set(ray) == {
                            "actor_origin_m", "actor_direction",
                            "measured_range_m", "endpoint_hit",
                            "capture_time_s"} for ray in rays)
                        assert all(abs(ray["capture_time_s"] -
                                       DECISION_TIME_S) < 1e-5 for ray in rays)
                        assert len(rays) == payload["counts"]["valid"]
                        near = sum(ray["endpoint_hit"] and
                                   ray["measured_range_m"] < .20
                                   for ray in rays)
                        assert near == payload["near_occupied_hits"] == (
                            row["sensor_summary"][mount]["near_occupied_hits"])
                        voxel = DepthVoxelMap()
                        add_packet(voxel, rays, DECISION_TIME_S)
                        coverage = robot.swept_exterior_coverage(
                            voxel, qpos, estimated, future,
                            DECISION_TIME_S, margin_m=.05)
                        assert coverage == payload["upper_shell_coverage"]
                        checks += len(rays)*2 + 5
                    minimum = float("inf")
                    first_below = None
                    closest = None
                    for tick in range(41):
                        time_s = DECISION_TIME_S + .02*tick
                        fraction = min(1., (time_s - DECISION_TIME_S)/.4)
                        room.actual[:] = actual
                        room.actual[1] -= SIDE_STEP_M*ACTUAL_SLIP*fraction
                        room.time_s = time_s
                        room._update_moving_obstacle()
                        room._forward()
                        gap = room._current_obstacle_clearance()
                        if gap[0] < minimum:
                            minimum, closest = float(gap[0]), list(gap[1:])
                        if first_below is None and gap[0] < .05:
                            first_below = round(time_s, 4)
                        checks += 1
                    actual_result = row["response_evaluator"]
                    assert abs(minimum - actual_result[
                        "minimum_scene_gap_m"]) < .0001
                    assert first_below == actual_result["first_below_5cm_s"]
                    assert closest == actual_result["closest_pair"]
                    checks += 3
                finally:
                    room.close()
                for camera in ("lower_rgb", "upper_rgb"):
                    meta = row["cameras"][camera]
                    rgb_path = ROOT / meta["image_path"]
                    depth_path = ROOT / meta["depth_path"]
                    assert sha(rgb_path) == meta["image_sha256"]
                    assert sha(depth_path) == meta["depth_sha256"]
                    clear_meta = rows[(seed, "clear")]["cameras"][camera]
                    clear_rgb = np.asarray(Image.open(ROOT / clear_meta["image_path"]))
                    rgb = np.asarray(Image.open(rgb_path))
                    with np.load(ROOT / clear_meta["depth_path"]) as data:
                        clear_depth = data["depth_m"].copy()
                    with np.load(depth_path) as data:
                        depth = data["depth_m"].copy()
                    changed_rgb = int(np.count_nonzero(
                        np.any(rgb != clear_rgb, axis=2)))
                    changed_depth = int(np.count_nonzero(~np.isclose(
                        depth, clear_depth, atol=.001, equal_nan=True)))
                    assert changed_rgb == meta["changed_rgb_pixels_vs_clear"]
                    assert changed_depth == meta["changed_depth_pixels_vs_clear"]
                    checks += 4
                if row["video"]:
                    video_path = ROOT / row["video"]["path"]
                    assert sha(video_path) == row["video"]["sha256"]
                    reader = imageio.get_reader(video_path)
                    try:
                        assert reader.count_frames() == 9
                        assert reader.get_data(0).shape == reader.get_data(8).shape
                    finally:
                        reader.close()
                    checks += 3
                results.append({"seed": seed, "case": name,
                                "minimum_gap_m": actual_result[
                                    "minimum_scene_gap_m"],
                                "point_gate_allows": row["v95_point_gate_allows"],
                                "low_near_hits": row["sensor_summary"][
                                    "low_right8"]["near_occupied_hits"],
                                "upper_near_hits": row["sensor_summary"][
                                    "upper_right8"]["near_occupied_hits"]})
    finally:
        structural.structural_model = old_model
    assert all(row["v95_point_gate_allows"] and
               row["response_evaluator"]["minimum_scene_gap_m"] < .05
               and row["sensor_summary"]["upper_right8"][
                   "near_occupied_hits"] >= 50
               and row["sensor_summary"]["low_right8"][
                   "near_occupied_hits"] == 0
               and all(camera["changed_rgb_pixels_vs_clear"] == 0 and
                       camera["changed_depth_pixels_vs_clear"] == 0
                       for camera in row["cameras"].values())
               for row in report["rows"] if row["case"] == "hand_close")
    checks += 1
    result = {
        "schema": "robot-learning-lab.ape-owner-exit-height-audit.v1",
        "passed": True, "checks": checks,
        "source_sha256": sha(AUDIT_SOURCE),
        "results": results,
        "scope": "Independently rebuilds four fixed V95-decision scenes in two seeds, verifies retained actor ray and camera hashes, recomputes point and occupancy gates, replays 20ms geometry, compares paired RGB/depth pixels and decodes response videos. All scene object labels and physical gaps are evaluator-only; the candidate side mounts remain hypothetical.",
    }
    path = OUT / "independent-audit.json"
    path.write_text(json.dumps(result, indent=2) + "\n")
    print(path, checks, "checks passed")


if __name__ == "__main__":
    main()
