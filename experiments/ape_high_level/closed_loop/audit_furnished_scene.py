"""Audit every 20 ms of furnished moving-prop layouts before evaluation."""

from __future__ import annotations

import argparse
import itertools
import json

import mujoco

from run import OUT, Room


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seeds", nargs="+", type=int,
                        default=[35, 64, 73, 74, 75, 76])
    args = parser.parse_args()
    room = Room()
    model = room.model
    names = [model.geom(gid).name for gid in range(model.ngeom)]
    scene_geoms = [gid for gid, name in enumerate(names)
                   if name.startswith("loop_") and
                   not name.startswith("loop_rug")]
    wall_geoms = [gid for gid, name in enumerate(names)
                  if name.startswith("room_")]
    pairs = [(a, b) for a, b in itertools.combinations(
        scene_geoms + wall_geoms, 2)
        if model.geom_bodyid[a] != model.geom_bodyid[b] and
        not (a in wall_geoms and b in wall_geoms)]
    rows = []
    try:
        for seed in args.seeds:
            room.reset()
            room.layout(seed, absent=False, furnished=True)
            room.configure_continuous_obstacle(seed)
            min_gap = (10., "", "", 0.)
            min_robot = 10.
            for tick in range(501):
                room.time_s = tick * .02
                room._update_moving_obstacle()
                room._forward()
                min_robot = min(min_robot,
                                room._current_obstacle_clearance()[0])
                for a, b in pairs:
                    gap = float(mujoco.mj_geomDistance(
                        model, room.data, a, b, 10., None))
                    if gap < min_gap[0]:
                        min_gap = (gap, names[a], names[b], room.time_s)
            rows.append({"seed": seed,
                         "min_distinct_scene_gap_m": min_gap[0],
                         "closest_pair": list(min_gap[1:3]),
                         "closest_time_s": round(min_gap[3], 3),
                         "stationary_robot_min_clearance_m": min_robot,
                         "valid": min_gap[0] > .005 and min_robot > .05})
    finally:
        room.close()
    report = {"seeds": args.seeds, "sample_dt_s": .02,
              "duration_s": 10., "rows": rows,
              "all_valid": all(row["valid"] for row in rows)}
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "furnished-geometry-audit.json").write_text(
        json.dumps(report, indent=2) + "\n")
    for row in rows:
        print(row)
    if not report["all_valid"]:
        raise SystemExit("Invalid furnished geometry")


if __name__ == "__main__":
    main()
