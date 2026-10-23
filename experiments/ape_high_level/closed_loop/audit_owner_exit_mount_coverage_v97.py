"""Independent retained-ray and geometry audit for the V97 side-view sweep."""

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
from run_owner_exit_height_v96 import CASES, DECISION_TIME_S, SIDE_STEP_M
from run_owner_exit_mount_coverage_v97 import LAYOUTS, MOUNTS, OUT, V95


SOURCE = Path(__file__)


def main() -> None:
    report = json.loads((OUT / "report.json").read_text())
    checks = 0

    def check(condition: bool, message: str) -> None:
        nonlocal checks
        if not condition:
            raise AssertionError(message)
        checks += 1

    check(report["source_sha256"] == sha(Path(__file__).with_name(
        "run_owner_exit_mount_coverage_v97.py")), "source hash")
    check(report["v95_report_sha256"] == sha(
        V95 / "live-moving-owner-offer-audit.json"), "V95 hash")
    check(report["selected_visual_sha256"] == sha(selected_visual()),
          "visual hash")
    check(report["mounts"] == json.loads(json.dumps(MOUNTS)), "mounts")
    check(report["layouts"] == json.loads(json.dumps(LAYOUTS)), "layouts")
    check(len(report["rows"]) == 6, "six cases")
    original_model = structural.structural_model
    structural.structural_model = lambda kind, side: add_person(
        original_model(kind, side))
    rows = []
    try:
        for row in report["rows"]:
            seed, case = row["seed"], row["case"]
            check(seed in (170, 172) and case in (
                "clear", "hand_close", "hand_clear_offset"), "case key")
            path = ROOT / row["actor_packet_path"]
            check(sha(path) == row["actor_packet_sha256"], "packet hash")
            packet = json.loads(path.read_text())
            check(packet["capture_time_s"] == DECISION_TIME_S,
                  "capture time")
            check(packet["mounts"] == report["mounts"], "packet mounts")
            check(set(packet["rays"]) == set(MOUNTS), "ray sources")
            pose = np.asarray(packet["estimated_pose_xyyaw"], dtype=float)
            qpos = np.asarray(packet["actor_qpos"], dtype=float)
            prior = json.loads((V95 / "live-moving-owner-offer-audit.json").read_text())
            v95 = next(item for item in prior["rows"]
                       if item["condition"] == "moving" and
                       item["case"][0] == seed)
            source = (V95 / "moving" / f"seed{seed}-present" /
                      f"seed{seed}-present/owner-guard-04.npz")
            with np.load(source) as old:
                check(np.array_equal(qpos, old["actor_qpos"]), "V95 qpos")
                check(np.array_equal(pose, old["estimated_pose_xyyaw"]),
                      "V95 pose")
            maps = {name: DepthVoxelMap() for name in LAYOUTS}
            for mount, (offset, _, _, _, width, height) in MOUNTS.items():
                rays = packet["rays"][mount]
                summary = row["mounts"][mount]
                counts = summary["counts"]
                check(len(rays) == counts["valid"], "valid count")
                check(counts["valid"] + counts["dropout"] +
                      counts["self_first"] == width * height,
                      "ray population")
                expected_origin = np.array([*pose[:2], 0.]) + sensor.rotate(
                    float(pose[2]), np.asarray(offset))
                near = 0
                hits = []
                for ray in rays:
                    check(ray["capture_time_s"] == DECISION_TIME_S,
                          "ray time")
                    check(np.allclose(ray["actor_origin_m"], expected_origin),
                          "ray origin")
                    check(abs(np.linalg.norm(ray["actor_direction"])-1) < 1e-8,
                          "ray direction")
                    check(.04 <= ray["measured_range_m"] <= .65,
                          "ray range")
                    if ray["endpoint_hit"]:
                        hits.append(ray["measured_range_m"])
                        near += ray["measured_range_m"] < .2
                check(len(hits) == counts["environment_hit"], "hit count")
                check(len(rays)-len(hits) == counts["max_range"],
                      "max-range count")
                check(near == summary["near_hits_under_20cm"], "near hits")
                check((min(hits) if hits else None) ==
                      summary["shortest_hit_m"], "nearest hit")
                for layout, names in LAYOUTS.items():
                    if mount in names:
                        sensor.add_packet(maps[layout], rays,
                                          DECISION_TIME_S)
            room = OfferDriftRoom(case=tuple(v95["case"]),
                                  pose_source="command", branch=OUT / "audit")
            try:
                room.layout(seed, absent=False, furnished=True)
                obstacle = CASES[case]
                if obstacle is not None:
                    position, size = obstacle
                    room.model.body_pos[room.model.body("loop_box").id] = position
                    room.model.geom("loop_box_geom").size[:] = size
                room.data.qpos[:] = qpos
                actual = np.array((qpos[0], qpos[1],
                                   2*math.atan2(qpos[6], qpos[3])))
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
                for layout, voxel_map in maps.items():
                    coverage = robot.swept_exterior_coverage(
                        voxel_map, qpos, pose, future, DECISION_TIME_S,
                        margin_m=.05)
                    check(coverage == row["layouts"][layout],
                          "retained-ray map coverage")
                    check(coverage["status"] != "free",
                          "no unsupported free corridor")
                minimum = float("inf")
                for tick in range(41):
                    room.actual[:] = actual
                    room.actual[1] -= SIDE_STEP_M * .55 * min(1., tick*.02/.4)
                    room.time_s = DECISION_TIME_S + tick*.02
                    room._update_moving_obstacle()
                    room._forward()
                    minimum = min(minimum,
                                  float(room._current_obstacle_clearance()[0]))
                check(round(minimum, 5) ==
                      row["evaluator_minimum_scene_gap_m"],
                      "response geometry")
            finally:
                room.close()
            rows.append({"seed": seed, "case": case,
                         "minimum_gap_m": round(minimum, 5),
                         "wide_clear_free_cells":
                             row["layouts"]["inner_low_upper_150"]["free"]})
    finally:
        structural.structural_model = original_model
    output = {"schema": "robot-learning-lab.ape-owner-exit-mount-audit.v1",
              "passed": True, "checks": checks,
              "source_sha256": sha(SOURCE), "results": rows,
              "scope": "Rebuilds six V97 rooms from retained V95 body states and V97 actor rays. Independently checks source identity, timestamps, ray geometry/counts, low/upper map coverage and 20ms response clearance. The side arrays and kinematic step remain hypothetical."}
    path = OUT / "independent-audit.json"
    path.write_text(json.dumps(output, indent=2) + "\n")
    print(path, checks, "checks passed")


if __name__ == "__main__":
    main()
