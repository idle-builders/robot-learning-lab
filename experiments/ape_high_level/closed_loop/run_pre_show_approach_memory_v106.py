"""V106: can existing approach-time ape views support the pre-show route?

Replay timestamped, self-masked side looks captured before V98's owner show
into the same actor voxel representation as V102. Compare first-step swept
coverage with and without that memory; changed obstacle cases reveal whether
old free-space evidence remains safe to reuse. This is an observability audit,
not a motion controller or proof of dynamic-scene clearance.
"""

from __future__ import annotations

import json
import math
from pathlib import Path

import numpy as np

from articulated_person_scene import START_HAND_XYZ_M, add_person, set_person_pose
import audit_blocked_range_mounts_v319 as sensor
import direct_structural_rooms_v270 as structural
from depth_voxel import DepthVoxelMap, MAX_AGE_S
from direct_structural_rooms_v270 import selected_visual
from robot_geometry import RobotGeometry
from run import H, ROOT, W, sha
from run_articulated_owner_low_drift_v92 import OfferDriftRoom
from run_moving_low_clutter_v302 import STOOL_SIZE
from run_owner_exit_height_v96 import CASES
from run_pre_show_reserve_screen_v100 import (COMMAND_STEP_M, DIRECTIONS,
                                              SEQUENCES, START_S, V98)
from run_pre_show_sensor_coverage_v102 import LAYOUTS
from self_depth_mask import SelfDepthMask


SOURCE = Path(__file__)
OUT = ROOT / "artifacts/ape-closed-loop-cove50-v106"
V102 = ROOT / "artifacts/ape-closed-loop-cove50-v102"
SIDEPROBES = (2, 3, 4, 5)
LAYOUT_NAMES = ("head_only", "head_rear_low_mid_near",
                "head_full_ring_low_upper")


def coverage(robot: RobotGeometry, qpos: np.ndarray, pose: np.ndarray,
             voxel_map: DepthVoxelMap) -> list[dict]:
    current = pose.copy()
    steps = []
    for command in SEQUENCES["back2_diag2"]:
        future = current.copy()
        future[:2] += COMMAND_STEP_M * np.asarray(DIRECTIONS[command])
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
    structural.structural_model = lambda kind, side: add_person(old_model(kind, side))
    rows = []
    try:
        for seed in (170, 172):
            prior = next(row for row in v98["rows"]
                         if row["condition"] == "moving" and row["case"][0] == seed)
            source_dir = (V98 / "moving" / f"seed{seed}-present" /
                          f"seed{seed}-present")
            side_sources = []
            for index in SIDEPROBES:
                path = source_dir / f"side-probe-{index:02d}.npz"
                with np.load(path) as packet:
                    stamp = float(packet["capture_time_s"])
                if not 0 <= START_S - stamp <= MAX_AGE_S:
                    raise ValueError(f"Side view has unsupported age: {path}")
                side_sources.append((path, stamp))
            for case in ("clear", "hand_close"):
                row102 = next(row for row in v102["rows"]
                              if row["seed"] == seed and row["case"] == case)
                packet_path = ROOT / row102["actor_packet_path"]
                if sha(packet_path) != row102["actor_packet_sha256"]:
                    raise ValueError(f"Changed V102 actor packet: {packet_path}")
                actor = json.loads(packet_path.read_text())
                qpos = np.asarray(actor["actor_qpos"], dtype=float)
                pose = np.asarray(actor["estimated_pose_xyyaw"], dtype=float)
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
                    room.actual[:] = (qpos[0], qpos[1],
                                      2*math.atan2(qpos[6], qpos[3]))
                    room.estimated[:] = pose
                    room.head = float(qpos[room.head_addr])
                    room.time_s = START_S
                    set_person_pose(room.model, START_HAND_XYZ_M,
                                    carried_item="loop_TomatoSauce")
                    room.obstacle_xy[3] = (0., .306)
                    room.model.body_pos[room.model.body("loop_stool").id] = (
                        0., .306, STOOL_SIZE[2])
                    room._forward()
                    fovy = float(room.model.cam_fovy[
                        room.model.camera("lower_rgb").id])
                    _, depth = room.capture("lower_rgb", depth=True)
                    mask = SelfDepthMask(room.model, camera="lower_rgb",
                                         width=W, height=H, agreement_m=.04)
                    try:
                        filtered, mask_stats = mask.apply(
                            depth, room.data.qpos.copy(), max_depth_m=1.4)
                    finally:
                        mask.close()
                    robot = RobotGeometry(room.model)
                    variants = {}
                    for retain_side in (False, True):
                        base = DepthVoxelMap()
                        observed = []
                        if retain_side:
                            for path, stamp in side_sources:
                                with np.load(path) as packet:
                                    stats = base.observe(
                                        packet["filtered_depth_m"],
                                        (packet["camera_origin_estimated_m"],
                                         packet["camera_rotation_estimated"]),
                                        stamp, fovy_deg=fovy, stride=4,
                                        max_depth_m=1.4, self_masked=True)
                                observed.append({
                                    "path": str(path.relative_to(ROOT)),
                                    "sha256": sha(path),
                                    "capture_time_s": round(stamp, 4),
                                    "age_at_decision_s": round(START_S-stamp, 4),
                                    "map_stats_after": stats,
                                })
                        fresh_stats = base.observe(
                            filtered, room.camera_pose_estimate("lower_rgb"),
                            START_S, fovy_deg=fovy, stride=4,
                            max_depth_m=1.4, self_masked=True)
                        layouts = {}
                        for layout_name in LAYOUT_NAMES:
                            voxel_map = DepthVoxelMap()
                            voxel_map.free = base.free.copy()
                            voxel_map.occupied = base.occupied.copy()
                            voxel_map.last_time_s = START_S
                            for mount in LAYOUTS[layout_name]:
                                sensor.add_packet(voxel_map,
                                                  actor["ray_packets"][mount],
                                                  START_S)
                            layouts[layout_name] = {
                                "steps": coverage(robot, qpos, pose, voxel_map),
                                "free_cells": len(voxel_map.free),
                                "occupied_cells": len(voxel_map.occupied),
                            }
                        variants["side_history" if retain_side else
                                 "fresh_only"] = {
                            "historical_views": observed,
                            "current_head_stats": fresh_stats,
                            "layouts": layouts,
                        }
                    rows.append({
                        "seed": seed, "case": case,
                        "v102_actor_packet_path": row102["actor_packet_path"],
                        "v102_actor_packet_sha256": row102["actor_packet_sha256"],
                        "head_mask_stats": mask_stats,
                        "variants": variants,
                    })
                finally:
                    room.close()
    finally:
        structural.structural_model = old_model
    report = {
        "schema": "robot-learning-lab.ape-pre-show-approach-memory.v1",
        "source_sha256": sha(SOURCE),
        "v98_report_sha256": sha(V98 / "live-moving-owner-offer-audit.json"),
        "v102_report_sha256": sha(V102 / "report.json"),
        "selected_visual_sha256": sha(selected_visual()),
        "max_map_age_s": MAX_AGE_S,
        "side_probe_indices": SIDEPROBES,
        "layouts": {name: LAYOUTS[name] for name in LAYOUT_NAMES},
        "scope": "Actor-side replay of four existing V98 self-masked lower RGB-D side views (12.4-14.8s) plus current16.2s head depth and retained V102 range packets at historical-body pre-show pose. Compare time-stamped 4cm voxel free/occupied/unknown swept-shell coverage with and without approach memory in clear and newly added close hand-height block scenes, two seeds. The historical views never saw the added block; no semantic static/dynamic separation, active replan, physical mount, selected-body action or route authorization is implied.",
        "rows": rows,
    }
    path = OUT / "report.json"
    path.write_text(json.dumps(report, indent=2) + "\n")
    for row in rows:
        print(row["seed"], row["case"], {
            variant: {
                name: (data["steps"][0]["upper"]["free"],
                       data["steps"][0]["upper"]["occupied"],
                       data["steps"][0]["upper"]["unknown"],
                       data["steps"][0]["lower"]["free"],
                       data["steps"][0]["lower"]["occupied"],
                       data["steps"][0]["lower"]["unknown"])
                for name, data in details["layouts"].items()}
            for variant, details in row["variants"].items()})
    print(path)


if __name__ == "__main__":
    main()
