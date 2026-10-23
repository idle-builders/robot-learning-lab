"""V216: do four added rear shin ToF views repair V190's side/back blind actions?"""

from __future__ import annotations

from collections import Counter, defaultdict
import hashlib
import json
import math
from pathlib import Path

import mujoco
import numpy as np

from audit_low_range_height_transfer_v70 import FRACTIONS, SCENES
from audit_shin_range_v190 import (BEAMS, FOV_RAD, MAX_RANGE_M, PITCH_RAD,
                                   POINT_MARGIN_M, ROOM_GROUP, SELF_GROUP)
from robot_geometry import RobotGeometry
from run import ROOT, Room, sha


SOURCE = Path(__file__)
OUT = ROOT / "artifacts/ape-closed-loop-cove50-v216"
V70 = ROOT / "artifacts/ape-closed-loop-cove50-v70/low-range-height-transfer.json"
V190 = ROOT / "artifacts/ape-closed-loop-cove50-v190/shin-range-audit.json"


def rear_cloud(room: Room, robot: RobotGeometry, case_id: str) -> tuple[np.ndarray, dict]:
    seed = int.from_bytes(hashlib.sha256((case_id + "-rear").encode()).digest()[:8], "little")
    rng = np.random.default_rng(seed)
    cloud = []
    counts = Counter()
    robot.set_state(room.data.qpos.copy())
    solids = robot.primitives()
    for side, name in ((1, "left_shin"), (-1, "right_shin")):
        bid = room.model.body(name).id
        rotation = room.data.xmat[bid].reshape(3, 3)
        origin = room.data.xpos[bid] + rotation @ np.array((-.02, side * .04, -.05))
        counts[f"{name}_rear_origin_z_cm"] = round(100 * float(origin[2]), 2)
        counts[f"{name}_rear_origin_body_gap_cm"] = round(
            100 * RobotGeometry.solid_distance(origin, solids), 2)
        for yaw_deg in (side * 112.5, side * 157.5):
            for row in range(BEAMS):
                elevation = PITCH_RAD + (row + .5 - BEAMS / 2) * FOV_RAD / BEAMS
                for col in range(BEAMS):
                    azimuth = math.radians(yaw_deg) + (col + .5 - BEAMS / 2) * FOV_RAD / BEAMS
                    local = np.array((math.cos(elevation) * math.cos(azimuth),
                                      math.cos(elevation) * math.sin(azimuth),
                                      math.sin(elevation)))
                    direction = rotation @ local
                    room_geom = np.array([-1], dtype=np.int32)
                    self_geom = np.array([-1], dtype=np.int32)
                    distance = mujoco.mj_ray(room.model, room.data, origin, direction,
                                             ROOM_GROUP, True, -1, room_geom)
                    self_distance = mujoco.mj_ray(room.model, room.data, origin, direction,
                                                  SELF_GROUP, True, -1, self_geom)
                    counts["rays"] += 1
                    if self_distance >= 0 and (distance < 0 or self_distance <= distance):
                        counts["self_first"] += 1
                        continue
                    if distance < 0 or distance > MAX_RANGE_M:
                        continue
                    point = origin + distance * direction
                    if point[2] < .015:
                        counts["floor_first"] += 1
                        continue
                    if rng.random() < .025:
                        counts["dropout"] += 1
                        continue
                    noisy = origin + max(.02, distance + rng.normal(0., .004)) * direction
                    cloud.append(noisy)
                    counts["environment_points"] += 1
                    if noisy[2] <= .08:
                        counts["low_points"] += 1
    return np.asarray(cloud, dtype=float).reshape(-1, 3), dict(counts)


def main() -> None:
    prior = json.loads(V70.read_text())
    old = json.loads(V190.read_text())
    assert prior["transfer_interpretable"] and len(prior["rows"]) == 592
    by_case = defaultdict(list)
    for row in prior["rows"]:
        by_case[row["case"]].append(row)
    cases = {case["case"]: case for case in prior["cases"] if case["valid_start"]}
    assert len(cases) == 74
    scene_specs = {spec[0]: spec for spec in SCENES}
    room = Room()
    robot = RobotGeometry(room.model)
    rows = []
    origins = []
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
            rear, counts = rear_cloud(room, robot, case_id)
            origins.append({"case": case_id, **counts})
            actor_qpos = room.data.qpos.copy()
            for prior_row in by_case[case_id]:
                delta = np.asarray(prior_row["candidate_xy_m"]) - pose[:2]
                gap = math.inf
                for fraction in FRACTIONS:
                    future = pose.copy()
                    future[:2] += fraction * delta
                    robot.set_state(robot.pose_qpos(actor_qpos, future))
                    if len(rear):
                        gap = min(gap, float(np.min(
                            RobotGeometry.solid_distances(rear, robot.primitives()))))
                old_row = next(r for r in old["rows"] if r["case"] == case_id and
                               r["direction_cell"] == prior_row["direction_cell"])
                front = old_row["point_gaps_m"]["four"]
                head = prior_row["point_gaps_m"]["head_upper"]
                combined = min(gap, math.inf if front is None else front, head)
                rows.append({"case": case_id,
                             "direction_cell": prior_row["direction_cell"],
                             "outside_forward_90": prior_row["move_outside_forward_90"],
                             "evaluator_safe_5cm": prior_row["evaluator_safe_5cm"],
                             "evaluator_model_clearance_m": prior_row["evaluator_model_clearance_m"],
                             "rear_environment_points": len(rear),
                             "rear_point_gap_m": None if math.isinf(gap) else round(gap, 5),
                             "front_four_head_pass": old_row["four_head_pass"],
                             "eight_head_pass": combined >= POINT_MARGIN_M})
            print(case_id, "rear points", len(rear), flush=True)
    finally:
        room.close()
    def score(key):
        return {"false_safe": sum(r[key] and not r["evaluator_safe_5cm"] for r in rows),
                "false_block": sum(not r[key] and r["evaluator_safe_5cm"] for r in rows),
                "unsafe_outside_forward90_pass": sum(
                    r[key] and not r["evaluator_safe_5cm"] and r["outside_forward_90"]
                    for r in rows)}
    result = {"scope": "V216 revealed extension of V190 same 74 selected-body poses and 592 rotated low-crate actions. Add four 45deg 8x8 rear-facing shin ToF views at rear-offset optical origins to the prior four forward views, with 4mm range noise and 2.5% dropout, room/self/floor first hits and unchanged 6.5cm actor point rule plus ideal upper-head depth. Reuse V190 front results; evaluator full-body labels only score after actor decisions. Optical upper bound only: no PCBs, swing/package, reflectance, interference, timing or native stop.",
              "source_sha256": sha(SOURCE), "v70_sha256": sha(V70),
              "v190_sha256": sha(V190),
              "scores": {"front_four_head": score("front_four_head_pass"),
                         "front_plus_rear_eight_head": score("eight_head_pass")},
              "origins": origins, "rows": rows}
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "rear-shin-range-audit.json").write_text(json.dumps(result, indent=2) + "\n")
    (OUT / SOURCE.name).write_bytes(SOURCE.read_bytes())
    print(json.dumps(result["scores"], indent=2))


if __name__ == "__main__":
    main()
