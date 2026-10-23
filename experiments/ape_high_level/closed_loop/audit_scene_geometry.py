"""Audit the v13 moving-obstacle layout before evaluating planner decisions."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import mujoco

from run import OUT, Room


def gap(scene: Room, first: str, second: str) -> float:
    return float(mujoco.mj_geomDistance(
        scene.model, scene.data, scene.model.geom(first).id,
        scene.model.geom(second).id, 10.0, None))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seeds", nargs="+", type=int, default=[35, 57, 58, 59, 60])
    args = parser.parse_args()
    room = Room()
    rows = []
    try:
        for seed in args.seeds:
            room.reset()
            room.layout(seed, absent=False)
            room.obstacle_xy[3] = [.44, .15 if seed % 2 else -.15]
            room.model.body_pos[room.model.body("loop_stool").id] = (
                *room.obstacle_xy[3], .20)
            room._forward()
            row = {"seed": seed,
                   "target_stool_gap_m": gap(room, "loop_TomatoSauce_geom",
                                               "loop_stool_geom"),
                   "crate_stool_gap_m": gap(room, "loop_crate_geom",
                                              "loop_stool_geom"),
                   "robot_furniture_gap_at_initial_pose_m": room.obstacle_clearance()[0]}
            row["valid_initial_geometry"] = min(row[key] for key in (
                "target_stool_gap_m", "crate_stool_gap_m",
                "robot_furniture_gap_at_initial_pose_m")) > .005
            rows.append(row)
    finally:
        room.close()
    output = OUT / "scene-geometry-audit.json"
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps({"rows": rows,
                                  "all_valid": all(row["valid_initial_geometry"]
                                                   for row in rows)}, indent=2) + "\n")
    for row in rows:
        print(row)
    if not all(row["valid_initial_geometry"] for row in rows):
        raise SystemExit("Invalid scene geometry")


if __name__ == "__main__":
    main()
