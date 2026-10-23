"""V287: ask whether shin first-hit rays observe a whole proposed body sweep.

This is a nominal angular-coverage diagnostic, not a certified range model.
Room geometry is accessed by the sensor adapter and evaluator, never by the
actor coverage calculation.
"""

from __future__ import annotations

from collections import Counter
import hashlib
import json
import math
from pathlib import Path

import mujoco
import numpy as np

from audit_lateral_clutter_sensors_v283 import (ACTOR_POINT_MARGIN_M,
                                                DIRECTIONS, POSE_Y, STEP_M,
                                                actor_gap)
from audit_shin_range_v190 import (BEAMS, FOV_RAD, MAX_RANGE_M, PITCH_RAD,
                                   ROOM_GROUP, SELF_GROUP)
from preflight_lateral_clutter_v282 import SEED, make_room
from robot_geometry import RobotGeometry
from run import ROOT, sha


SOURCE = Path(__file__)
OUT = ROOT / "artifacts/ape-closed-loop-cove50-v287"
BASE = ROOT / "artifacts/ape-closed-loop-cove50-v283/lateral-clutter-sensor-audit.json"
SURFACE_MARGIN_M = .05
RANGE_HEADROOM_M = .02
BEAM_HALF_CELL_RAD = FOV_RAD / BEAMS / 2
BEAM_POSITION_ALLOWANCE_M = .005
BANDS = ((.015, .16), (.16, .32), (.32, .55))


def observe(room, case: str) -> tuple[dict, dict]:
    """Return first-hit beam bundles and obstacle points in world sensor frame."""
    seed = int.from_bytes(hashlib.sha256(case.encode()).digest()[:8], "little")
    rng = np.random.default_rng(seed)
    records = {"two": [], "four": []}
    points = {"two": [], "four": []}
    counts = Counter()
    for side, body_name in ((1, "left_shin"), (-1, "right_shin")):
        bid = room.model.body(body_name).id
        rotation = room.data.xmat[bid].reshape(3, 3)
        origin = room.data.xpos[bid] + rotation @ np.array((.02, side * .04, -.05))
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
                    room_id, self_id = np.array([-1], dtype=np.int32), np.array([-1], dtype=np.int32)
                    distance = mujoco.mj_ray(room.model, room.data, origin,
                                             direction, ROOM_GROUP, True, -1, room_id)
                    self_distance = mujoco.mj_ray(room.model, room.data, origin,
                                                  direction, SELF_GROUP, True, -1, self_id)
                    counts[f"{layout}_rays"] += 1
                    if self_distance >= 0 and (distance < 0 or self_distance <= distance):
                        counts[f"{layout}_self_first"] += 1
                        continue
                    if distance < 0 or distance > MAX_RANGE_M:
                        counts[f"{layout}_no_valid_return"] += 1
                        continue
                    endpoint = origin + distance * direction
                    floor = endpoint[2] < .015
                    if floor:
                        counts[f"{layout}_floor_first"] += 1
                    elif rng.random() < .025:
                        counts[f"{layout}_dropout"] += 1
                        continue
                    measured = distance if floor else max(.02, distance + rng.normal(0., .004))
                    records[layout].append((origin.copy(), direction.copy(), measured))
                    if not floor:
                        points[layout].append(origin + measured * direction)
    result = {}
    for name in ("two", "four"):
        layouts = ("two",) if name == "two" else ("two", "four")
        rec = [item for key in layouts for item in records[key]]
        pts = [item for key in layouts for item in points[key]]
        result[name] = {
            "origins": np.asarray([x[0] for x in rec], dtype=float).reshape(-1, 3),
            "directions": np.asarray([x[1] for x in rec], dtype=float).reshape(-1, 3),
            "ranges": np.asarray([x[2] for x in rec], dtype=float),
            "points": np.asarray(pts, dtype=float).reshape(-1, 3),
        }
    return result, dict(counts)


def proposed_exterior(robot: RobotGeometry, qpos: np.ndarray,
                      pose: np.ndarray, direction: int) -> np.ndarray:
    robot.set_state(robot.pose_qpos(qpos, pose))
    current = robot.primitives()
    heading = np.array((0., float(direction), 0.))
    queried = set()
    samples = []
    for fraction in (.25, .5, .75, 1.):
        future = pose.copy()
        future[1] += direction * STEP_M * fraction
        robot.set_state(robot.pose_qpos(qpos, future))
        solids = robot.primitives()
        for point, normal in robot.surface(solids):
            if point[2] < BANDS[0][0] or point[2] >= BANDS[-1][1]:
                continue
            if float(normal @ heading) < .2:
                continue
            if RobotGeometry.solid_distance(point, solids) < -.005:
                continue
            sample = point + SURFACE_MARGIN_M * normal
            if RobotGeometry.solid_distance(sample, solids) < -.005:
                continue
            if RobotGeometry.solid_distance(sample, current) < -.005:
                continue
            key = tuple(np.floor(sample / .025).astype(int))
            if key not in queried:
                queried.add(key)
                samples.append(sample)
    return np.asarray(samples, dtype=float).reshape(-1, 3)


def nominal_free(samples: np.ndarray, bundle: dict) -> np.ndarray:
    if not len(samples) or not len(bundle["ranges"]):
        return np.zeros(len(samples), dtype=bool)
    # Each 8x8 return supports only its own nominal angular cell and the
    # segment before the first hit. Missing, self-first and dropout cells do
    # not clear space. This does not certify the carrier's true beam footprint.
    delta = samples[:, None, :] - bundle["origins"][None, :, :]
    along = np.einsum("ijk,jk->ij", delta, bundle["directions"])
    radial = np.linalg.norm(delta - along[:, :, None] *
                            bundle["directions"][None, :, :], axis=2)
    beam_radius = BEAM_POSITION_ALLOWANCE_M + np.maximum(along, 0.) * math.tan(BEAM_HALF_CELL_RAD)
    free = ((along >= .02) &
            (along <= bundle["ranges"][None, :] - RANGE_HEADROOM_M) &
            (radial <= beam_radius))
    return np.any(free, axis=1)


def main() -> None:
    prior = json.loads(BASE.read_text())
    OUT.mkdir(parents=True, exist_ok=True)
    room = make_room(SEED, "present")
    robot = RobotGeometry(room.model)
    rows = []
    observations = []
    try:
        for index, y in enumerate(POSE_Y):
            pose = np.array((0., y, 0.))
            room.actual[:] = pose
            room.estimated[:] = pose
            room._forward()
            qpos = room.data.qpos.copy()
            bundles, counts = observe(room, f"v287-seed{SEED}-pose{index}")
            observations.append({"pose_y_m": y, "ray_counts": counts,
                                 "valid_first_hit_rays": {name: len(bundle["ranges"])
                                                          for name, bundle in bundles.items()},
                                 "obstacle_points": {name: len(bundle["points"])
                                                     for name, bundle in bundles.items()}})
            for direction in DIRECTIONS:
                samples = proposed_exterior(robot, qpos, pose, direction)
                baseline = next(row for row in prior["rows"]
                                if row["pose_y_m"] == y and row["direction"] == direction)
                row = {"pose_y_m": y, "direction": direction,
                       "evaluator_valid_start_5cm": baseline[
                           "evaluator_valid_start_5cm"],
                       "evaluator_safe_5cm": baseline["evaluator_safe_5cm"],
                       "evaluator_min_clearance_m": baseline[
                           "evaluator_min_clearance_m"],
                       "surface_samples": len(samples), "layouts": {}}
                for name, bundle in bundles.items():
                    gap = actor_gap(robot, qpos, pose, bundle["points"], direction)
                    free = nominal_free(samples, bundle)
                    band_counts = {}
                    for low, high in BANDS:
                        mask = (samples[:, 2] >= low) & (samples[:, 2] < high)
                        band_counts[f"{low:.3f}-{high:.3f}m"] = {
                            "samples": int(mask.sum()), "nominal_free": int(np.sum(free & mask)),
                            "unknown": int(np.sum(~free & mask))}
                    row["layouts"][name] = {
                        "actor_point_gap_m": None if not math.isfinite(gap) else round(gap, 5),
                        "point_pass": bool(gap >= ACTOR_POINT_MARGIN_M),
                        "nominal_free_surface": int(free.sum()),
                        "unknown_surface": int((~free).sum()),
                        "full_coverage_pass": bool(len(samples) > 0 and
                                                   free.all() and
                                                   gap >= ACTOR_POINT_MARGIN_M),
                        "bands": band_counts}
                rows.append(row)
                print("pose", index, "direction", direction,
                      "samples", len(samples),
                      {name: (part["nominal_free_surface"], part["unknown_surface"])
                       for name, part in row["layouts"].items()}, flush=True)
    finally:
        room.close()
    def score(pool: list[dict]) -> dict:
        result = {}
        for name in ("two", "four"):
            for key in ("point_pass", "full_coverage_pass"):
                result[f"{name}_{key}"] = {
                    "false_safe": sum(row["layouts"][name][key] and
                                      not row["evaluator_safe_5cm"] for row in pool),
                    "false_block": sum(not row["layouts"][name][key] and
                                       row["evaluator_safe_5cm"] for row in pool),
                    "passed": sum(row["layouts"][name][key] for row in pool)}
        return result
    scores = score(rows)
    valid_start_scores = score([row for row in rows
                                if row["evaluator_valid_start_5cm"]])
    report = {"scope": "V287 actor-only nominal observed-free screen for V283 six same-pose side actions on selected Direct Frame. Recreate articulated two/four shin 8x8 first-hit rays with self-first, floor first, 2.5% environment dropout and 4mm noise. Missing/self/dropout rays never establish free space. Query newly entered leading selected-body surface at 5cm outward margin over four 8cm-step fractions, across z1.5-55cm, and require all samples supported by a first-hit beam's nominal 5.625deg angular cell with 2cm range headroom. Compare prior 6.5cm obstacle point gap, then score against independent V283 full-body 5cm clearance labels. Angular-cell footprint, specular behavior and package fit are not calibrated; this diagnostic can reject unsupported movement but cannot certify safety. Static poses only, no native stop or live task.",
              "source_sha256": sha(SOURCE), "v283_sha256": sha(BASE),
              "parameters": {"surface_margin_m": SURFACE_MARGIN_M,
                             "range_headroom_m": RANGE_HEADROOM_M,
                             "beam_half_cell_deg": math.degrees(BEAM_HALF_CELL_RAD),
                             "beam_position_allowance_m": BEAM_POSITION_ALLOWANCE_M},
              "observations": observations, "rows": rows, "scores": scores,
              "valid_start_scores": valid_start_scores}
    (OUT / "shin-swept-coverage-audit.json").write_text(
        json.dumps(report, indent=2) + "\n")
    (OUT / SOURCE.name).write_bytes(SOURCE.read_bytes())
    print(json.dumps(scores, indent=2), flush=True)


if __name__ == "__main__":
    main()
