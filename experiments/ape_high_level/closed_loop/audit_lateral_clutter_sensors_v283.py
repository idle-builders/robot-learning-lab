"""V283: same-action range layouts on the frozen Direct Frame low-clutter fork.

Actor decisions use only first-hit range points and encoder-derived body solids.
The full-body room clearance and geom labels are added after each decision.
"""

from __future__ import annotations

import json
import math
from pathlib import Path

import numpy as np

from audit_c1_plane_mount_v237 import MOUNTS, scan as c1_scan
from audit_shin_range_v190 import scan as shin_scan
from build_scene_pairs_v133 import body_geoms, full_body_clearance, scene_geoms
from preflight_lateral_clutter_v282 import SEED, make_room
from robot_geometry import RobotGeometry
from run import ROOT, sha


SOURCE = Path(__file__)
OUT = ROOT / "artifacts/ape-closed-loop-cove50-v283"
POSE_Y = (0., -.0704, -.1408)
DIRECTIONS = (-1, 1)
STEP_M = .08
SLIP = .88
ACTOR_POINT_MARGIN_M = .065
EVALUATOR_MARGIN_M = .05


def virtual_cloud(room, height: float) -> np.ndarray:
    points = []
    for angle, distance in zip(np.linspace(-math.pi, math.pi, 72, endpoint=False),
                               room.range_scan(height)):
        if distance is None:
            continue
        yaw = float(room.actual[2]) + float(angle)
        points.append((float(room.actual[0]) + distance * math.cos(yaw),
                       float(room.actual[1]) + distance * math.sin(yaw), height))
    return np.asarray(points, dtype=float).reshape(-1, 3)


def actor_gap(robot: RobotGeometry, actor_qpos: np.ndarray,
              pose: np.ndarray, cloud: np.ndarray, direction: int) -> float:
    gap = math.inf
    for fraction in np.linspace(0., 1., 21):
        future = pose.copy()
        future[1] += direction * STEP_M * fraction
        robot.set_state(robot.pose_qpos(actor_qpos, future))
        if len(cloud):
            gap = min(gap, float(np.min(RobotGeometry.solid_distances(
                cloud, robot.primitives()))))
    return gap


def evaluator_gap(room, obstacles: list[int], bodies: list[int],
                  pose: np.ndarray, direction: int) -> tuple[float, str, str]:
    closest = (math.inf, "", "")
    for fraction in np.linspace(0., 1., 21):
        room.actual[:] = pose
        room.actual[1] += direction * STEP_M * SLIP * fraction
        room._forward()
        row = full_body_clearance(room, obstacles, bodies)
        if row[0] < closest[0]:
            closest = row
    return closest


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    room = make_room(SEED, "present")
    robot = RobotGeometry(room.model)
    obstacles = scene_geoms(room)
    bodies = body_geoms(room, obstacles)
    rows = []
    observations = []
    try:
        for index, y in enumerate(POSE_Y):
            pose = np.array((0., y, 0.))
            room.actual[:] = pose
            room.estimated[:] = pose
            room._forward()
            qpos = room.data.qpos.copy()
            start_clearance = float(full_body_clearance(room, obstacles, bodies)[0])
            clouds = {
                "virtual_4_11": np.concatenate((virtual_cloud(room, .04),
                                                   virtual_cloud(room, .11))),
            }
            counts = {}
            for name in ("front_30cm", "front_11cm", "front_4cm"):
                clouds[name], counts[name] = c1_scan(room, pose, MOUNTS[name])
            shin, shin_counts = shin_scan(room, robot,
                                          {"case": f"v283-seed{SEED}-pose{index}"})
            clouds["shin_two"] = shin["two"]
            clouds["shin_four"] = shin["four"]
            counts["shin"] = shin_counts
            observations.append({"pose_y_m": y, "counts": counts,
                                 "point_counts": {name: len(cloud)
                                                  for name, cloud in clouds.items()},
                                 "low_points_in_negative_lateral_halfspace": {
                                     name: int(np.sum((cloud[:, 2] < .09) &
                                                      (cloud[:, 1] < y - .08)))
                                     for name, cloud in clouds.items()}})
            for direction in DIRECTIONS:
                # Actor calculation is complete before the evaluator query.
                gaps = {name: actor_gap(robot, qpos, pose, cloud, direction)
                        for name, cloud in clouds.items()}
                decisions = {name: gap >= ACTOR_POINT_MARGIN_M
                             for name, gap in gaps.items()}
                closest = evaluator_gap(room, obstacles, bodies, pose, direction)
                rows.append({"pose_y_m": y, "direction": direction,
                             "evaluator_start_clearance_m": round(start_clearance, 5),
                             "evaluator_valid_start_5cm": start_clearance >= EVALUATOR_MARGIN_M,
                             "actor_point_gap_m": {
                                 name: None if not math.isfinite(gap) else round(gap, 5)
                                 for name, gap in gaps.items()},
                             "actor_pass": decisions,
                             "evaluator_safe_5cm": closest[0] >= EVALUATOR_MARGIN_M,
                             "evaluator_min_clearance_m": round(closest[0], 5),
                             "evaluator_limiting_robot_geom": closest[1],
                             "evaluator_limiting_scene_geom": closest[2]})
            room.actual[:] = pose
            room._forward()
            print("pose", index, "y", y, "hits",
                  {key: counts[key]["crate_hits"] for key in
                   ("front_30cm", "front_11cm", "front_4cm")}, flush=True)
    finally:
        room.close()
    keys = tuple(rows[0]["actor_pass"])
    def score(pool: list[dict]) -> dict:
        return {key: {"false_safe": sum(row["actor_pass"][key] and
                                        not row["evaluator_safe_5cm"] for row in pool),
                      "false_block": sum(not row["actor_pass"][key] and
                                         row["evaluator_safe_5cm"] for row in pool),
                      "passed": sum(row["actor_pass"][key] for row in pool)}
                for key in keys}
    scores = score(rows)
    valid_start_scores = score([row for row in rows if row["evaluator_valid_start_5cm"]])
    report = {"scope": "V283 frozen V282 seed138 Direct Frame low-clutter fork, same six 8cm lateral actions from 0/-7.04/-14.08cm and both sides. The -14.08cm start is already inside the evaluator 5cm margin; report all-six diagnostics and valid-start-only scores separately, and do not treat those invalid starts as reachable planner outcomes. Actor uses first-hit sensor returns and encoder-derived robot solids with a 6.5cm swept point margin. Virtual 4/11cm central 360 scan is the optimistic control; C1-class front 30/11/4cm single planes have self-first and 5cm blind; shin 8x8 two/four views have independent 2.5% dropout and 4mm noise. Evaluator uses full-body scene geometry with 5cm margin only after decision. Point-only pass does not prove unobserved free space, and no mount is package/dynamic qualified; 4cm C1 is mechanically implausible. Kinematic base and stationary clutter; no live planner or native stop bound.",
              "source_sha256": sha(SOURCE),
              "v282_sha256": sha(ROOT / "artifacts/ape-closed-loop-cove50-v282/lateral-clutter-preflight.json"),
              "rows": rows, "observations": observations, "scores": scores,
              "valid_start_scores": valid_start_scores,
              "valid_start_action_count": sum(row["evaluator_valid_start_5cm"]
                                               for row in rows)}
    (OUT / "lateral-clutter-sensor-audit.json").write_text(
        json.dumps(report, indent=2) + "\n")
    (OUT / SOURCE.name).write_bytes(SOURCE.read_bytes())
    print(json.dumps(scores, indent=2), flush=True)
    print("valid-start-only", json.dumps(valid_start_scores, indent=2), flush=True)


if __name__ == "__main__":
    main()
