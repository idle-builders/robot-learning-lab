"""V190: replay V70 rotated low-crate actions with articulated shin range views.

Actor clouds come from first-hit ray returns and selected joint geometry.
V70's full-body labels are read only after actor gap decisions are computed.
"""

from __future__ import annotations

from collections import Counter, defaultdict
import hashlib
import json
import math
from pathlib import Path

import mujoco
import numpy as np

from audit_low_range_height_transfer_v70 import FRACTIONS, SCENES
from robot_geometry import RobotGeometry
from run import ROOT, Room, sha


SOURCE = Path(__file__)
OUT = ROOT / "artifacts/ape-closed-loop-cove50-v190"
V70 = ROOT / "artifacts/ape-closed-loop-cove50-v70/low-range-height-transfer.json"
BEAMS = 8
FOV_RAD = math.radians(45)
PITCH_RAD = math.radians(-12)
MAX_RANGE_M = 1.2
POINT_MARGIN_M = .065
ROOM_GROUP = np.array([1, 0, 0, 0, 0, 0], dtype=np.uint8)
SELF_GROUP = np.array([0, 1, 0, 0, 1, 0], dtype=np.uint8)


def scan(room: Room, robot: RobotGeometry, case: dict) -> tuple[dict, dict]:
    """World points from first room/self returns; no object identity is exposed."""
    seed = int.from_bytes(hashlib.sha256(case["case"].encode()).digest()[:8], "little")
    rng = np.random.default_rng(seed)
    clouds = {"two": [], "four": []}
    counts = Counter()
    robot.set_state(room.data.qpos.copy())
    solids = robot.primitives()
    for side, body_name in ((1, "left_shin"), (-1, "right_shin")):
        bid = room.model.body(body_name).id
        rotation = room.data.xmat[bid].reshape(3, 3)
        origin = room.data.xpos[bid] + rotation @ np.array((.02, side * .04, -.05))
        counts[f"{body_name}_origin_z_cm"] = round(100 * float(origin[2]), 2)
        counts[f"{body_name}_origin_body_gap_cm"] = round(
            100 * RobotGeometry.solid_distance(origin, solids), 2)
        for yaw_deg in (side * 22.5, side * 67.5):
            layout = "two" if abs(yaw_deg) < 30 else "four"
            for row in range(BEAMS):
                elevation = PITCH_RAD + (row + .5 - BEAMS / 2) * FOV_RAD / BEAMS
                for column in range(BEAMS):
                    azimuth = math.radians(yaw_deg) + (column + .5 - BEAMS / 2) * FOV_RAD / BEAMS
                    local_ray = np.array((math.cos(elevation) * math.cos(azimuth),
                                          math.cos(elevation) * math.sin(azimuth),
                                          math.sin(elevation)))
                    direction = rotation @ local_ray
                    room_hit = np.array([-1], dtype=np.int32)
                    self_hit = np.array([-1], dtype=np.int32)
                    distance = mujoco.mj_ray(room.model, room.data, origin, direction,
                                             ROOM_GROUP, True, -1, room_hit)
                    self_distance = mujoco.mj_ray(room.model, room.data, origin, direction,
                                                  SELF_GROUP, True, -1, self_hit)
                    counts[f"{layout}_rays"] += 1
                    if self_distance >= 0 and (distance < 0 or self_distance <= distance):
                        counts[f"{layout}_self_first"] += 1
                        continue
                    if distance < 0 or distance > MAX_RANGE_M:
                        continue
                    point = origin + distance * direction
                    if point[2] < .015:
                        counts[f"{layout}_floor_first"] += 1
                        continue
                    if rng.random() < .025:
                        counts[f"{layout}_dropout"] += 1
                        continue
                    noisy_point = origin + max(.02, distance + rng.normal(0, .004)) * direction
                    clouds[layout].append(noisy_point)
                    counts[f"{layout}_environment_points"] += 1
                    if noisy_point[2] <= .08:
                        counts[f"{layout}_low_points"] += 1
    two = np.asarray(clouds["two"], dtype=float).reshape(-1, 3)
    four = np.concatenate((two, np.asarray(clouds["four"], dtype=float).reshape(-1, 3)))
    return {"two": two, "four": four}, dict(counts)


def main() -> None:
    prior = json.loads(V70.read_text())
    assert prior["transfer_interpretable"] and len(prior["rows"]) == 592
    cases = {case["case"]: case for case in prior["cases"] if case["valid_start"]}
    by_case: dict[str, list[dict]] = defaultdict(list)
    for row in prior["rows"]:
        by_case[row["case"]].append(row)
    assert set(cases) == set(by_case) and len(cases) == 74
    scene_specs = {spec[0]: spec for spec in SCENES}
    room = Room()
    robot = RobotGeometry(room.model)
    rows = []
    observations = []
    try:
        for case_id, case in cases.items():
            seed = case["seed"]
            _, crate_x, crate_y_rel, half_x, half_y, half_z = scene_specs[seed]
            side = 1 if seed % 2 else -1
            room.reset()
            room.layout(seed, absent=False, furnished=True)
            crate_id = room.model.geom("loop_crate_geom").id
            room.model.geom_size[crate_id] = (half_x, half_y, half_z)
            room.model.body_pos[room.model.body("loop_crate").id] = (
                crate_x, side * crate_y_rel, half_z)
            pose = np.asarray(case["pose_xyyaw"], dtype=float)
            room.actual[:] = pose
            room.estimated[:] = pose
            room.head = 0.
            room._forward()
            points, counts = scan(room, robot, case)
            observations.append({"case": case_id, "pose_xyyaw": pose.tolist(),
                                 "point_counts": {name: len(cloud) for name, cloud in points.items()},
                                 **counts})
            actor_qpos = room.data.qpos.copy()
            for prior_row in by_case[case_id]:
                future_xy = np.asarray(prior_row["candidate_xy_m"], dtype=float)
                delta = future_xy - pose[:2]
                gaps = {name: math.inf for name in points}
                for fraction in FRACTIONS:
                    future = pose.copy()
                    future[:2] += fraction * delta
                    robot.set_state(robot.pose_qpos(actor_qpos, future))
                    solids = robot.primitives()
                    for name, cloud in points.items():
                        if len(cloud):
                            gaps[name] = min(gaps[name], float(np.min(
                                RobotGeometry.solid_distances(cloud, solids))))
                head_gap = prior_row["point_gaps_m"]["head_upper"]
                upper_virtual_gap = prior_row["point_gaps_m"]["range_11"]
                row = {"case": case_id, "seed": seed,
                       "direction_cell": prior_row["direction_cell"],
                       "body_relative_move_rad": prior_row["body_relative_move_rad"],
                       "outside_forward_90": prior_row["move_outside_forward_90"],
                       "crate_height_m": prior_row["crate_height_m"],
                       "evaluator_safe_5cm": prior_row["evaluator_safe_5cm"],
                       "evaluator_model_clearance_m": prior_row["evaluator_model_clearance_m"],
                       "evaluator_closest_geom_pair": prior_row["evaluator_closest_geom_pair"],
                       "point_gaps_m": {name: None if math.isinf(gap) else round(gap, 5)
                                        for name, gap in gaps.items()},
                       "ideal_v70_pass": prior_row["point_4_8_11_pass"]}
                for name in ("two", "four"):
                    row[f"{name}_head_pass"] = min(gaps[name], head_gap) >= POINT_MARGIN_M
                    row[f"{name}_head_virtual11_pass"] = min(
                        gaps[name], head_gap, upper_virtual_gap) >= POINT_MARGIN_M
                rows.append(row)
            print(case_id, "two", len(points["two"]), "four", len(points["four"]),
                  flush=True)
    finally:
        room.close()
    def score(key: str) -> dict:
        return {"passes": sum(row[key] for row in rows),
                "false_safe": sum(row[key] and not row["evaluator_safe_5cm"] for row in rows),
                "false_block": sum(not row[key] and row["evaluator_safe_5cm"] for row in rows),
                "unsafe_outside_forward90_pass": sum(
                    row[key] and not row["evaluator_safe_5cm"] and row["outside_forward_90"]
                    for row in rows),
                "unsafe_sub8cm_crate_pass": sum(
                    row[key] and not row["evaluator_safe_5cm"] and
                    row["crate_height_m"] < .08 and
                    row["evaluator_closest_geom_pair"][1] == "loop_crate_geom"
                    for row in rows)}
    scores = {key: score(key) for key in (
        "ideal_v70_pass", "two_head_pass", "four_head_pass",
        "two_head_virtual11_pass", "four_head_virtual11_pass")}
    result = {"scope": "V190 preregistered same-scene optical screen of 2 and 4 joint-attached 8x8 shin range imagers against 74 V70 rotated poses/592 actions. First room/self hits, floor reject, 2.5% dropout, 4mm noise; no package/dynamic/reflectance qualification. V70 evaluator label never enters actor gaps.",
              "source_sha256": sha(SOURCE), "v70_sha256": sha(V70),
              "selected_design_sha256": sha(ROOT / "robot/ape-current-design.json"),
              "scores": scores,
              "layout_screen_pass": (scores["four_head_pass"]["false_safe"] == 0 and
                                     scores["four_head_pass"]["false_block"] <= 36),
              "observations": observations, "rows": rows}
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "shin-range-audit.json").write_text(json.dumps(result, indent=2) + "\n")
    (OUT / SOURCE.name).write_bytes(SOURCE.read_bytes())
    print(json.dumps({"scores": scores, "screen_pass": result["layout_screen_pass"],
                      "total_observations": len(observations)}, indent=2))


if __name__ == "__main__":
    main()
