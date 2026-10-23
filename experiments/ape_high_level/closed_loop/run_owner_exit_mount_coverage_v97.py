"""V97: can wider side range placements observe V96's swept arm corridor?

Development-only sensor diagnostic at the saved V95 owner decision. Each mount
is a hypothetical first-hit range array; simulator geometry is used only to
generate rays and to score the imposed side step after the actor map is built.
"""

from __future__ import annotations

import json
import math
from pathlib import Path

import numpy as np

import audit_blocked_range_mounts_v319 as sensor
import direct_structural_rooms_v270 as structural
from articulated_person_scene import FAR_HAND_XYZ_M, add_person, set_person_pose
from depth_voxel import DepthVoxelMap
from direct_structural_rooms_v270 import selected_visual
from robot_geometry import RobotGeometry
from run import ROOT, sha
from run_articulated_owner_low_drift_v92 import OfferDriftRoom
from run_owner_exit_height_v96 import (CASES, DECISION_TIME_S,
                                       SIDE_STEP_M, response_path)


SOURCE = Path(__file__)
OUT = ROOT / "artifacts/ape-closed-loop-cove50-v97"
V95 = ROOT / "artifacts/ape-closed-loop-cove50-v95"
MOUNTS = {
    "inner_upper_45": ((0., -.155, .23), -math.pi / 2, 45., 45., 8, 8),
    "inner_upper_110": ((0., -.155, .23), -math.pi / 2, 110., 80., 31, 21),
    "inner_low_110": ((0., -.155, .12), -math.pi / 2, 110., 80., 31, 21),
    "inner_upper_150": ((0., -.155, .23), -math.pi / 2, 150., 110., 51, 31),
    "inner_low_150": ((0., -.155, .12), -math.pi / 2, 150., 110., 51, 31),
}
LAYOUTS = {
    "inner_upper": ("inner_upper_45",),
    "inner_upper_110": ("inner_upper_110",),
    "inner_low_upper_110": ("inner_low_110", "inner_upper_110"),
    "inner_low_upper_150": ("inner_low_150", "inner_upper_150"),
}


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    old_model = structural.structural_model
    old_mounts = sensor.MOUNTS.copy()
    old_horizontal = sensor.HORIZONTAL_DEG.copy()
    old_vertical = sensor.VERTICAL_DEG.copy()
    structural.structural_model = lambda kind, side: add_person(
        old_model(kind, side))
    rows = []
    try:
        for seed in (170, 172):
            v95 = json.loads((V95 / "live-moving-owner-offer-audit.json").read_text())
            prior = next(row for row in v95["rows"]
                         if row["condition"] == "moving" and row["case"][0] == seed)
            source = (V95 / "moving" / f"seed{seed}-present" /
                      f"seed{seed}-present/owner-guard-04.npz")
            with np.load(source) as packet:
                qpos = packet["actor_qpos"].copy()
                pose = packet["estimated_pose_xyyaw"].copy()
            actual = np.array((qpos[0], qpos[1],
                               2 * math.atan2(qpos[6], qpos[3])))
            for case in ("clear", "hand_close", "hand_clear_offset"):
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
                    room.actual[:] = actual
                    room.estimated[:] = pose
                    room.head = float(qpos[room.head_addr])
                    room.time_s = DECISION_TIME_S
                    room._update_moving_obstacle()
                    set_person_pose(room.model, FAR_HAND_XYZ_M,
                                    carried_item="loop_TomatoSauce")
                    room._forward()
                    robot = RobotGeometry(room.model)
                    future = pose.copy()
                    future[1] -= SIDE_STEP_M
                    maps = {name: DepthVoxelMap() for name in LAYOUTS}
                    summaries = {}
                    retained_rays = {}
                    for mount, (offset, bearing, hfov, vfov, width, height) in MOUNTS.items():
                        origin = np.array([*actual[:2], 0.]) + sensor.rotate(
                            float(actual[2]), np.asarray(offset))
                        origin_inside_block = None
                        if obstacle is not None:
                            position, size = obstacle
                            origin_inside_block = bool(np.all(
                                np.abs(origin - np.asarray(position)) <
                                np.asarray(size)))
                            if origin_inside_block:
                                raise AssertionError(
                                    f"{mount} origin intersects changed block")
                        sensor.MOUNTS.clear()
                        sensor.MOUNTS[mount] = (offset, bearing)
                        sensor.HORIZONTAL_DEG = np.linspace(-hfov / 2, hfov / 2, width)
                        sensor.VERTICAL_DEG = np.linspace(-vfov / 2, vfov / 2, height)
                        rays, counts = sensor.range_packet(room, pose, mount,
                                                           seed, DECISION_TIME_S)
                        retained_rays[mount] = rays
                        for layout, names in LAYOUTS.items():
                            if mount in names:
                                sensor.add_packet(maps[layout], rays,
                                                  DECISION_TIME_S)
                        summaries[mount] = {
                            "counts": counts,
                            "mount_origin_inside_changed_block":
                                origin_inside_block,
                            "near_hits_under_20cm": sum(
                                ray["endpoint_hit"] and
                                ray["measured_range_m"] < .2 for ray in rays),
                            "shortest_hit_m": (min((ray["measured_range_m"]
                                for ray in rays if ray["endpoint_hit"]),
                                default=None)),
                        }
                    coverage = {name: robot.swept_exterior_coverage(
                        voxel_map, qpos, pose, future, DECISION_TIME_S,
                        margin_m=.05) for name, voxel_map in maps.items()}
                    branch = OUT / f"seed{seed}-{case}"
                    branch.mkdir(parents=True, exist_ok=True)
                    packet_path = branch / "actor-side-rays.json"
                    packet_path.write_text(json.dumps({
                        "capture_time_s": DECISION_TIME_S,
                        "estimated_pose_xyyaw": pose.tolist(),
                        "actor_qpos": qpos.tolist(),
                        "mounts": MOUNTS,
                        "rays": retained_rays,
                    }, indent=2) + "\n")
                    evaluation, _ = response_path(room, actual)
                    rows.append({"seed": seed, "case": case,
                                 "mounts": summaries, "layouts": coverage,
                                 "actor_packet_path": str(packet_path.relative_to(ROOT)),
                                 "actor_packet_sha256": sha(packet_path),
                                 "evaluator_minimum_scene_gap_m":
                                     evaluation["minimum_scene_gap_m"]})
                finally:
                    room.close()
    finally:
        structural.structural_model = old_model
        sensor.MOUNTS.clear()
        sensor.MOUNTS.update(old_mounts)
        sensor.HORIZONTAL_DEG = old_horizontal
        sensor.VERTICAL_DEG = old_vertical
    report = {
        "schema": "robot-learning-lab.ape-owner-exit-mount-coverage.v1",
        "source_sha256": sha(SOURCE),
        "v95_report_sha256": sha(V95 / "live-moving-owner-offer-audit.json"),
        "selected_visual_sha256": sha(selected_visual()),
        "mounts": MOUNTS,
        "layouts": LAYOUTS,
        "sensor_model": {"max_range_m": sensor.RANGE_MAX_M,
                         "noise_std_m": sensor.RANGE_NOISE_STD_M,
                         "dropout_fraction": sensor.DROPOUT_FRACTION,
                         "voxel_m": .04},
        "scope": "Fixed V96 hand-height development scenes at the V95 16.6s decision. Hypothetical first-hit side range arrays only; no packaging or physical-body qualification. Maps receive rays, not evaluator object identity or free-space truth. 5cm buffered sampled upper-shell coverage is diagnostic; evaluator geometry scores the imposed 8cm/0.55 step separately.",
        "rows": rows,
    }
    path = OUT / "report.json"
    path.write_text(json.dumps(report, indent=2) + "\n")
    for row in rows:
        print(row["seed"], row["case"],
              [(name, c["status"], c["free"], c["occupied"], c["unknown"])
               for name, c in row["layouts"].items()])
    print(path)


if __name__ == "__main__":
    main()
