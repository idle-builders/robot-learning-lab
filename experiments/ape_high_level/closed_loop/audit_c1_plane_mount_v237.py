"""V237: first-hit C1-class single-plane mount screen on frozen V70 actions.

The model tests geometry and scan-plane role, not a fitted or qualified C1.
Evaluator labels are consulted only after actor point-gap decisions.
"""

from __future__ import annotations

from collections import defaultdict
import json
import math
from pathlib import Path

import mujoco
import numpy as np

from audit_low_range_height_transfer_v70 import FRACTIONS, SCENES
from robot_geometry import RobotGeometry
from run import ROOT, Room, sha


SOURCE = Path(__file__)
OUT = ROOT / "artifacts/ape-closed-loop-cove50-v237"
V70 = ROOT / "artifacts/ape-closed-loop-cove50-v70/low-range-height-transfer.json"
ROOM_GROUP = np.array([1, 0, 0, 0, 0, 0], dtype=np.uint8)
SELF_GROUP = np.array([0, 1, 0, 0, 1, 0], dtype=np.uint8)
MOUNTS = {"torso_30cm": (.00, .00, .30),
          "front_30cm": (.14, .00, .30),
          "head_top_50cm": (.00, .00, .50),
          "front_11cm": (.14, .00, .11),
          "front_4cm": (.14, .00, .04)}
MIN_RANGE_M = .05
MAX_RANGE_M = 12.
POINT_MARGIN_M = .065
BEAMS = 500  # 0.72 degrees, vendor nominal at 10 Hz


def scan(room: Room, pose: np.ndarray, mount: tuple[float, float, float]
         ) -> tuple[np.ndarray, dict]:
    c, s = math.cos(pose[2]), math.sin(pose[2])
    x, y, z = mount
    origin = np.array([pose[0] + c * x - s * y,
                       pose[1] + s * x + c * y, z])
    points = []
    count = {"self_first": 0, "room_hits": 0, "crate_hits": 0,
             "blind_hits": 0, "no_hit": 0}
    for local_angle in np.linspace(-math.pi, math.pi, BEAMS, endpoint=False):
        a = pose[2] + local_angle
        direction = np.array([math.cos(a), math.sin(a), 0.])
        room_id = np.array([-1], dtype=np.int32)
        self_id = np.array([-1], dtype=np.int32)
        distance = mujoco.mj_ray(room.model, room.data, origin, direction,
                                 ROOM_GROUP, True, -1, room_id)
        self_distance = mujoco.mj_ray(room.model, room.data, origin, direction,
                                      SELF_GROUP, True, -1, self_id)
        if self_distance >= 0 and (distance < 0 or self_distance <= distance):
            count["self_first"] += 1
        elif distance < 0 or distance > MAX_RANGE_M:
            count["no_hit"] += 1
        elif distance < MIN_RANGE_M:
            count["blind_hits"] += 1
        else:
            count["room_hits"] += 1
            if room_id[0] == room.model.geom("loop_crate_geom").id:
                count["crate_hits"] += 1
            points.append(origin + distance * direction)
    return np.asarray(points, dtype=float).reshape(-1, 3), {
        **count, "origin_xyz_m": origin.tolist()}


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
            actor_qpos = room.data.qpos.copy()
            clouds = {}
            observation = {"case": case_id, "seed": seed}
            for name, mount in MOUNTS.items():
                clouds[name], observation[name] = scan(room, pose, mount)
            observations.append(observation)
            for old in by_case[case_id]:
                delta = np.asarray(old["candidate_xy_m"]) - pose[:2]
                gaps = {name: math.inf for name in MOUNTS}
                for fraction in FRACTIONS:
                    future = pose.copy()
                    future[:2] += fraction * delta
                    robot.set_state(robot.pose_qpos(actor_qpos, future))
                    solids = robot.primitives()
                    for name, cloud in clouds.items():
                        if len(cloud):
                            gaps[name] = min(gaps[name], float(np.min(
                                RobotGeometry.solid_distances(cloud, solids))))
                upper = old["point_gaps_m"]["head_upper"]
                rows.append({"case": case_id, "seed": seed,
                             "direction_cell": old["direction_cell"],
                             "crate_height_m": old["crate_height_m"],
                             "evaluator_safe_5cm": old["evaluator_safe_5cm"],
                             "evaluator_closest_geom_pair":
                                 old["evaluator_closest_geom_pair"],
                             "virtual_three_plane_pass": old["point_4_8_11_pass"],
                             "point_gaps_m": {name: None if not math.isfinite(gap)
                                              else round(gap, 4)
                                              for name, gap in gaps.items()},
                             **{name + "_plus_head_pass":
                                min(gaps[name], upper) >= POINT_MARGIN_M
                                for name in MOUNTS}})
    finally:
        room.close()

    def score(key: str) -> dict:
        return {"false_safe": sum(r[key] and not r["evaluator_safe_5cm"]
                                  for r in rows),
                "false_block": sum(not r[key] and r["evaluator_safe_5cm"]
                                   for r in rows),
                "passed": sum(r[key] for r in rows),
                "unsafe_low_crate_pass": sum(
                    r[key] and not r["evaluator_safe_5cm"] and
                    r["crate_height_m"] < .08 and
                    r["evaluator_closest_geom_pair"][1] == "loop_crate_geom"
                    for r in rows)}

    scores = {key: score(key) for key in
              ("virtual_three_plane_pass",) +
              tuple(name + "_plus_head_pass" for name in MOUNTS)}
    output = {"scope": "V237 revealed static same-pose geometric screen on 74 V70 rotated poses/592 actions. Nominal C1-class 500-ray horizontal first-hit returns at three candidate optical centers, 5cm near blind and 12m max, self first hits. Same retained ideal head-upper points and selected-body 6.5cm point gap; evaluator labels only after actor decisions. No fitted package, reflectance, range error, scan age/motion smear, native stop or live task authority. Vendor specification is a parameter reference, not an emulated device.",
              "source_sha256": sha(SOURCE), "v70_sha256": sha(V70),
              "vendor_sources": ["https://www.slamtec.com/en/c1/spec",
                                 "https://wiki.slamtec.com/download/attachments/83066883/SLAMTEC_rplidar_datasheet_C1_v1.0_en.pdf?api=v2&modificationDate=1700531479533&version=1"],
              "mounts_body_xyz_m": MOUNTS,
              "optical_height_above_base_m": .0298,
              "package_xyz_m": [.0556, .0556, .0413],
              "package_mass_kg": .110,
              "scores": scores, "observations": observations, "rows": rows}
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "c1-plane-mount-audit.json").write_text(
        json.dumps(output, indent=2) + "\n")
    (OUT / SOURCE.name).write_bytes(SOURCE.read_bytes())
    print(json.dumps({"scores": scores,
                      "mean_hits": {name: round(float(np.mean([
                          o[name]["room_hits"] for o in observations])), 1)
                          for name in MOUNTS},
                      "mean_self_first": {name: round(float(np.mean([
                          o[name]["self_first"] for o in observations])), 1)
                          for name in MOUNTS}}, indent=2), flush=True)


if __name__ == "__main__":
    main()
