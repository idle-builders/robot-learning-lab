"""V109: actor evidence gate for the V100 pre-show meeting candidates.

The actor replays its timestamped approach views, takes a current RGB-D look
and source-tagged near-body ranges, and screens fixed candidate moves across
pose perturbations. Evaluator V100 outcomes are joined only after its choice.
This is a pre-show planner decision, not a full new owner episode or a
selected-body movement implementation.
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
                                    freshness_filtered, route_screen)
from robot_geometry import RobotGeometry
from run import H, ROOT, W, sha
from run_articulated_owner_low_drift_v92 import OfferDriftRoom
from run_moving_low_clutter_v302 import STOOL_SIZE
from run_owner_exit_height_v96 import CASES
from run_pre_show_reserve_screen_v100 import SEQUENCES, START_S, V98
from self_depth_mask import SelfDepthMask


SOURCE = Path(__file__)
OUT = ROOT / "artifacts/ape-closed-loop-cove50-v109"
V100 = ROOT / "artifacts/ape-closed-loop-cove50-v100"
V105 = ROOT / "artifacts/ape-closed-loop-cove50-v105"
SIDEPROBES = (2, 3, 4, 5)
LAYOUTS = {
    "head_only": (),
    "head_rear_pair": ("rear_mid_near", "rear_low_near"),
    "head_rear_pair_right_upper": ("rear_mid_near", "rear_low_near",
                                   "right_upper"),
}
CANDIDATES = {name: commands for name, commands in SEQUENCES.items()
              if commands}


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    v98 = json.loads((V98 / "live-moving-owner-offer-audit.json").read_text())
    v100 = json.loads((V100 / "report.json").read_text())
    v105 = json.loads((V105 / "report.json").read_text())
    current_design = json.loads((ROOT / "robot/ape-current-design.json").read_text())
    old_model = structural.structural_model
    structural.structural_model = lambda kind, side: add_person(old_model(kind, side))
    rows = []
    try:
        for seed in (170, 172):
            prior = next(row for row in v98["rows"]
                         if row["condition"] == "moving" and row["case"][0] == seed)
            source_dir = (V98 / "moving" / f"seed{seed}-present" /
                          f"seed{seed}-present")
            source_packet = source_dir / "owner-guard-00.npz"
            with np.load(source_packet) as packet:
                qpos = packet["actor_qpos"].copy()
                estimated = packet["estimated_pose_xyyaw"].copy()
            side_sources = []
            for index in SIDEPROBES:
                path = source_dir / f"side-probe-{index:02d}.npz"
                with np.load(path) as packet:
                    stamp = float(packet["capture_time_s"])
                if not 0 <= START_S - stamp <= 4.:
                    raise ValueError("approach view is stale or from the future")
                side_sources.append((path, stamp))
            for case, obstacle in CASES.items():
                range_row = next(row for row in v105["rows"]
                                 if row["seed"] == seed and row["case"] == case
                                 and row["perturbation"] == "nominal")
                range_path = ROOT / range_row["actor_packet_path"]
                if sha(range_path) != range_row["actor_packet_sha256"]:
                    raise ValueError("V105 range packet changed")
                range_actor = json.loads(range_path.read_text())
                if not np.allclose(range_actor["estimated_pose_xyyaw"], estimated):
                    raise ValueError("actor pose does not match V98 source")
                room = OfferDriftRoom(case=tuple(prior["case"]),
                                      pose_source="command", branch=OUT / "scene")
                try:
                    room.layout(seed, absent=False, furnished=True)
                    if obstacle is not None:
                        position, size = obstacle
                        room.model.body_pos[room.model.body("loop_box").id] = position
                        room.model.geom("loop_box_geom").size[:] = size
                    room.data.qpos[:] = qpos
                    room.actual[:] = (qpos[0], qpos[1],
                                      2*math.atan2(qpos[6], qpos[3]))
                    room.estimated[:] = estimated
                    room.head = float(qpos[room.head_addr])
                    room.time_s = START_S
                    set_person_pose(room.model, START_HAND_XYZ_M,
                                    carried_item="loop_TomatoSauce")
                    room.obstacle_xy[3] = (0., .306)
                    room.model.body_pos[room.model.body("loop_stool").id] = (
                        0., .306, STOOL_SIZE[2])
                    room._forward()
                    rgb, depth = room.capture("lower_rgb", depth=True)
                    mask = SelfDepthMask(room.model, camera="lower_rgb",
                                         width=W, height=H, agreement_m=.04)
                    try:
                        filtered, mask_stats = mask.apply(
                            depth, room.data.qpos.copy(), max_depth_m=1.4)
                    finally:
                        mask.close()
                    camera_pose = room.camera_pose_estimate("lower_rgb")
                    fovy = float(room.model.cam_fovy[
                        room.model.camera("lower_rgb").id])
                    branch = OUT / f"seed{seed}-{case}"
                    branch.mkdir(parents=True, exist_ok=True)
                    head_path = branch / "current-head-rgbd.npz"
                    np.savez_compressed(
                        head_path, rgb=np.asarray(rgb),
                        filtered_depth_m=filtered,
                        camera_origin_estimated_m=camera_pose[0],
                        camera_rotation_estimated=camera_pose[1],
                        actor_qpos=qpos,
                        estimated_pose_xyyaw=estimated,
                        capture_time_s=START_S)
                    actor_sources = [{
                        "path": str(path.relative_to(ROOT)),
                        "sha256": sha(path),
                        "capture_time_s": round(stamp, 4),
                        "age_s": round(START_S-stamp, 4),
                        "kind": "historical_side_rgbd",
                    } for path, stamp in side_sources]
                    actor_sources += [{
                        "path": str(head_path.relative_to(ROOT)),
                        "sha256": sha(head_path),
                        "capture_time_s": START_S,
                        "age_s": 0., "kind": "current_head_rgbd",
                    }, {
                        "path": str(range_path.relative_to(ROOT)),
                        "sha256": sha(range_path),
                        "capture_time_s": START_S,
                        "age_s": 0., "kind": "current_near_body_range",
                    }]
                    historical = DepthVoxelMap()
                    for path, stamp in side_sources:
                        with np.load(path) as packet:
                            historical.observe(
                                packet["filtered_depth_m"],
                                (packet["camera_origin_estimated_m"],
                                 packet["camera_rotation_estimated"]),
                                stamp, fovy_deg=fovy, stride=4,
                                max_depth_m=1.4, self_masked=True)
                    historical.observe(
                        filtered, camera_pose, START_S,
                        fovy_deg=fovy, stride=4,
                        max_depth_m=1.4, self_masked=True)
                    robot = RobotGeometry(room.model)
                    layouts = {}
                    for name, mounts in LAYOUTS.items():
                        voxel_map = DepthVoxelMap()
                        voxel_map.free = historical.free.copy()
                        voxel_map.occupied = historical.occupied.copy()
                        voxel_map.last_time_s = START_S
                        for mount in mounts:
                            sensor.add_packet(
                                voxel_map,
                                range_actor["ray_packets"][mount], START_S)
                        fresh = freshness_filtered(
                            voxel_map, START_S, ROUTE_FREE_AGE_S)
                        layouts[name] = {
                            "raw_free_cells": len(voxel_map.free),
                            "motion_fresh_free_cells": len(fresh.free),
                            "occupied_cells": len(fresh.occupied),
                            "candidate_routes": route_screen(
                                robot, fresh, qpos, estimated,
                                CANDIDATES, START_S),
                        }
                    preferred = layouts["head_rear_pair_right_upper"]
                    supported = [name for name, result in preferred[
                        "candidate_routes"].items()
                        if result["coverage_status"] == "observed_free"]
                    evaluation = {name: next(row for row in v100["rows"]
                                             if row["seed"] == seed and
                                             row["case"] == case and
                                             row["sequence"] == name)[
                                  "minimum_scene_gap_m"]
                                  for name in SEQUENCES}
                    rows.append({
                        "seed": seed, "case": case,
                        "estimated_pose_xyyaw": estimated.round(5).tolist(),
                        "source_packet_path": str(source_packet.relative_to(ROOT)),
                        "source_packet_sha256": sha(source_packet),
                        "actor_sources": actor_sources,
                        "head_mask_stats": mask_stats,
                        "right_upper_under_10cm_fraction": range_row[
                            "actor_range_metrics"]["right_upper"][
                                "under_10cm_fraction_of_valid"],
                        "layouts": layouts,
                        "actor_choice": {
                            "status": ("no_observed_free_meeting_route"
                                       if not supported else
                                       "map_covered_but_unqualified"),
                            "map_covered_routes": supported,
                            "owner_turn_request":
                                "defer_show_and_request_clear_space",
                            "pose_error_bound_qualified": False,
                            "dynamic_future_occupancy_qualified": False,
                            "selected_body_motion_skill_qualified": False,
                            "motion_authorized": False,
                        },
                        "evaluator_only_minimum_gap_by_sequence_m": evaluation,
                    })
                finally:
                    room.close()
    finally:
        structural.structural_model = old_model
    report = {
        "schema": "robot-learning-lab.ape-pre-show-actor-choice.v1",
        "source_sha256": sha(SOURCE),
        "planner_source_sha256": sha(Path(__file__).with_name(
            "pre_show_actor_planner.py")),
        "v98_report_sha256": sha(V98 / "live-moving-owner-offer-audit.json"),
        "v100_report_sha256": sha(V100 / "report.json"),
        "v105_report_sha256": sha(V105 / "report.json"),
        "current_design_pointer_sha256": sha(ROOT / "robot/ape-current-design.json"),
        "current_design_status": current_design["status"],
        "route_free_age_s": ROUTE_FREE_AGE_S,
        "pose_samples_xyyaw": POSE_OFFSETS,
        "candidate_sequences": CANDIDATES,
        "layouts": LAYOUTS,
        "scope": "Actor-side pre-show route coverage decision at V98 16.2s state in four V96 development obstacle cases and two lighting seeds. Uses saved timestamped self-masked side views, newly captured head RGB-D, current hypothetical rear/upper range packets, actor estimated pose and joint geometry. Free-space evidence older than0.4s cannot authorize motion; occupied evidence remains. Every candidate step needs free upper/lower 5cm-buffered shell in five illustrative pose samples. Evaluator V100 outcome added only after decision. No calibrated pose bound, dynamic forecast, selected-body policy, physical sensor mount, live owner request or new complete episode is claimed.",
        "rows": rows,
    }
    path = OUT / "report.json"
    path.write_text(json.dumps(report, indent=2) + "\n")
    for row in rows:
        print(row["seed"], row["case"], row["actor_choice"]["status"],
              {name: {route: value["coverage_status"]
                      for route, value in layout["candidate_routes"].items()}
               for name, layout in row["layouts"].items()})
    print(path)


if __name__ == "__main__":
    main()
