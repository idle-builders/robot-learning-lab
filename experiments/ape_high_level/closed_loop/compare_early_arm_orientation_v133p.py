"""V133p: compare alternate left shoulder sign at early meeting station.

V126p showed both signs clear the low prop in the original pose. Here the
same owner-show geometry and V129p actor map test +0.6 rad at back12.
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
from pre_show_actor_planner import ROUTE_FREE_AGE_S, freshness_filtered
from robot_geometry import RobotGeometry
from run import ROOT, sha
from run_articulated_owner_low_drift_v92 import OfferDriftRoom
from run_moving_low_clutter_v302 import STOOL_SIZE
import screen_early_show_posture_v126p as v126p
import screen_early_retreat_observability_v129p as v129p
import validate_early_meeting_v128p as v128p


SOURCE = Path(__file__)
OUT = ROOT / "artifacts/ape-closed-loop-cove50-early-arm-orientation"
POSTURE = (.6, 0.)
STATION = v128p.STATIONS["back12"]
LAYOUT = "yaw+1.4-outward_dense_3d_rings"


def map_at_eight(record: dict) -> DepthVoxelMap:
    head = record["head_packets"]["1.4"]
    path = ROOT / head["path"]
    assert sha(path) == head["sha256"]
    voxel_map = DepthVoxelMap()
    with np.load(path) as packet:
        voxel_map.observe(packet["filtered_depth_m"],
                          (packet["camera_origin_estimated_m"],
                           packet["camera_rotation_estimated"]),
                          v129p.DECISION_S, fovy_deg=head["fovy_deg"],
                          stride=4, max_depth_m=1.4, self_masked=True)
    ray_path = ROOT / record["range_packet_path"]
    assert sha(ray_path) == record["range_packet_sha256"]
    rays = json.loads(ray_path.read_text())["ray_packets"]
    for name in v129p.LAYOUTS["outward_dense_3d_rings"]:
        sensor.add_packet(voxel_map, rays[name], v129p.DECISION_S)
    return freshness_filtered(voxel_map, v129p.DECISION_S,
                              ROUTE_FREE_AGE_S)


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    v98 = json.loads((v128p.V98 /
                      "live-moving-owner-offer-audit.json").read_text())
    v129 = json.loads((v129p.OUT / "report.json").read_text())
    original_model = structural.structural_model
    structural.structural_model = lambda kind, side: add_person(
        original_model(kind, side))
    geometry = []
    coverage = []
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
                                  pose_source="command", branch=OUT)
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
                fresh = map_at_eight(record)
                robot = RobotGeometry(room.model)
                plus_qpos = qpos.copy()
                plus_qpos[room.model.joint(
                    "left_shoulder_pitch").qposadr[0]] = POSTURE[0]
                screened = v129p.screen(
                    robot, fresh, plus_qpos, estimated, STATION,
                    {"estimated": (0., 0., 0.)})
                plus = screened["pose_posture_samples"]["neutral:estimated"]
                minus = record["alternatives"][LAYOUT]["routes"][
                    "back12"]["pose_posture_samples"]["left_raised:estimated"]
                coverage.append({"seed": seed, "plus_rad": POSTURE[0],
                                 "plus_swept": plus,
                                 "minus_rad": -.6,
                                 "minus_swept": minus})
                print(seed, "plus", {h: (plus[h]["status"],
                    plus[h]["unknown"]) for h in ("upper", "lower")},
                    flush=True)
                if seed == v126p.SEED:
                    station_qpos = qpos.copy()
                    station_qpos[:2] += STATION
                    station_estimate = estimated.copy()
                    station_estimate[:2] += STATION
                    for case in v128p.CASES_TO_VALIDATE:
                        row = v126p.one(room, station_qpos,
                                        station_estimate, case,
                                        "left_p06", POSTURE)
                        geometry.append(row)
                        print("show", case, row["minimum_scene_gap_m"],
                              row["closest_pair"], flush=True)
            finally:
                room.close()
    finally:
        structural.structural_model = original_model
    report = {"schema": "robot-learning-lab.ape-early-arm-orientation.v1",
              "source_sha256": sha(SOURCE),
              "v129_report_sha256": sha(v129p.OUT / "report.json"),
              "selected_visual_sha256": sha(selected_visual()),
              "scope": "Evaluator geometry and actor-map sensitivity comparison "
              "for +0.6 versus -0.6 rad left shoulder on the 12cm early "
              "retreat. Uses V129p's richest virtual sensor layout at the "
              "estimated pose only. Plus-arm endpoint show geometry is "
              "tested in four scene cases; full arm self-clearance, RGB-D "
              "identity, pose perturbations and skill dynamics are not.",
              "coverage": coverage, "geometry": geometry}
    (OUT / "report.json").write_text(json.dumps(report, indent=2)+"\n")
    print(OUT / "report.json")


if __name__ == "__main__":
    main()
