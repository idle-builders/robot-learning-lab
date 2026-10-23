"""Evaluator-only screen of range/footprint proxy against Cove-50 collision gaps.

Do not import this into the actor: true geometry is used only to assess a
candidate sensor-side proxy before a paired rollout.
"""

from __future__ import annotations

from collections import Counter
import json
import math

import mujoco
import numpy as np

from footprint_route import scan_points
from run import OUT, Room


CANDIDATES = {"old_rectangle": (.11, .21),
              "shell_rectangle": (.075, .17),
              "padded_shell": (.08, .18)}


def segmented_gap(points: np.ndarray, xy: np.ndarray, *, tight: bool = False) -> float:
    """Union of conservative torso, arm, and foot bands in the neutral pose."""
    if len(points) == 0:
        return 10.
    parts = (((0., 0., .028, .105),
              (0., .13, .022, .04), (0., -.13, .022, .04),
              (.01, .063, .065, .04), (.01, -.063, .065, .04))
             if tight else
             ((0., 0., .055, .105),
              (0., .125, .045, .055), (0., -.125, .045, .055),
              (.01, .063, .065, .045), (.01, -.063, .065, .045)))
    return min(proxy_gap(points, xy + np.array([cx, cy]), hx, hy)
               for cx, cy, hx, hy in parts)


def elevated_scan_points(room: Room) -> np.ndarray:
    """Virtual second 72-ray plane through the furnished tabletop height."""
    origin = np.array([room.actual[0], room.actual[1], .33])
    group = np.array([1, 0, 0, 0, 0, 0], dtype=np.uint8)
    ranges = []
    for angle in np.linspace(-math.pi, math.pi, 72, endpoint=False):
        direction = np.array([math.cos(angle), math.sin(angle), 0.])
        gid = np.array([-1], dtype=np.int32)
        distance = mujoco.mj_ray(room.model, room.data, origin, direction,
                                 group, True, room.model.body("torso").id, gid)
        ranges.append(float(distance) if distance >= 0 else None)
    return scan_points(ranges, room.actual)


def proxy_gap(points: np.ndarray, xy: np.ndarray, half_x: float,
              half_y: float) -> float:
    if len(points) == 0:
        return 10.
    delta = np.abs(points - xy)
    return float(np.min(np.hypot(np.maximum(delta[:, 0] - half_x, 0.),
                                 np.maximum(delta[:, 1] - half_y, 0.))))


def main() -> None:
    room = Room()
    rows = []
    candidates = (*CANDIDATES, "segmented_shell", "tight_segmented_shell",
                  "dual_height_tight_shell")
    failures = {name: Counter() for name in candidates}
    examples = {name: [] for name in candidates}
    try:
        for seed in (85, 88):
            for phase, y in (("blocking", .16), ("retreated", .40)):
                room.reset()
                room.layout(seed, absent=False, furnished=True)
                side = 1 if seed % 2 else -1
                room.obstacle_xy[3] = [.44, side * y]
                room.model.body_pos[room.model.body("loop_stool").id] = (
                    *room.obstacle_xy[3], .20)
                for x in np.arange(0., .721, .08):
                    for yy in np.arange(-.64, .641, .08):
                        room.actual[:] = (x, yy, 0.)
                        room._forward()
                        truth, body, obstacle = room._current_obstacle_clearance()
                        points = scan_points(room.range_scan(), room.actual)
                        elevated = elevated_scan_points(room)
                        row = {"seed": seed, "phase": phase,
                               "xy_m": [round(float(x), 3), round(float(yy), 3)],
                               "true_gap_m": round(truth, 4),
                               "closest_body": body,
                               "closest_obstacle": obstacle,
                               "scan_hits": len(points)}
                        for name in candidates:
                            predicted = (segmented_gap(
                                np.concatenate((points, elevated))
                                if name == "dual_height_tight_shell" else points,
                                room.actual[:2],
                                tight=name != "segmented_shell")
                                         if name in ("segmented_shell",
                                                     "tight_segmented_shell",
                                                     "dual_height_tight_shell") else
                                         proxy_gap(points, room.actual[:2],
                                                   *CANDIDATES[name]))
                            row[name + "_gap_m"] = round(predicted, 4)
                            if truth < .05 and predicted >= .055:
                                outcome = "false_safe"
                            elif truth >= .05 and predicted < .055:
                                outcome = "false_block"
                            else:
                                outcome = "correct"
                            failures[name][outcome] += 1
                            if outcome != "correct" and len(examples[name]) < 12:
                                examples[name].append({**row, "outcome": outcome})
                        rows.append(row)
    finally:
        room.close()
    output = {"scope": "Fixed-yaw neutral shell; 72 rays at 11 cm height; furnished seeds 85/88, two stool positions; actor proxy compared with evaluator-only full-body signed distance.",
              "grid_step_m": .08, "required_true_clearance_m": .05,
              "actor_threshold_m": .055,
              "candidates_half_extents_m": CANDIDATES,
              "counts": {name: dict(count) for name, count in failures.items()},
              "examples": examples, "rows": rows}
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "footprint-calibration-audit.json").write_text(
        json.dumps(output, indent=2) + "\n")
    print(json.dumps(output["counts"], indent=2))


if __name__ == "__main__":
    main()
