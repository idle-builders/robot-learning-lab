"""Check every 20 ms of the v14 moving prop path for invalid geometry."""

from __future__ import annotations

import argparse
import json

import mujoco

from run import OUT, Room


def geom_gap(room: Room, first: str, second: str) -> float:
    return float(mujoco.mj_geomDistance(
        room.model, room.data, room.model.geom(first).id,
        room.model.geom(second).id, 10., None))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seeds", nargs="+", type=int, default=[35, 61, 62, 63, 64])
    args = parser.parse_args()
    room = Room()
    rows = []
    try:
        for seed in args.seeds:
            room.reset()
            room.layout(seed, absent=False)
            room.configure_continuous_obstacle(seed)
            closest = {"target_stool_gap_m": 10., "crate_stool_gap_m": 10.,
                       "robot_initial_path_gap_m": 10.}
            for tick in range(501):
                room.time_s = tick * .02
                room._update_moving_obstacle()
                room._forward()
                closest["target_stool_gap_m"] = min(
                    closest["target_stool_gap_m"],
                    geom_gap(room, "loop_TomatoSauce_geom", "loop_stool_geom"))
                closest["crate_stool_gap_m"] = min(
                    closest["crate_stool_gap_m"],
                    geom_gap(room, "loop_crate_geom", "loop_stool_geom"))
                closest["robot_initial_path_gap_m"] = min(
                    closest["robot_initial_path_gap_m"],
                    room._current_obstacle_clearance()[0])
            rows.append({"seed": seed, **closest,
                         "valid_geometry": min(closest.values()) > .005})
    finally:
        room.close()
    report = {"seeds": args.seeds, "sample_dt_s": .02, "duration_s": 10.,
              "rows": rows, "all_valid": all(row["valid_geometry"] for row in rows)}
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "scene-trajectory-audit.json").write_text(json.dumps(report, indent=2) + "\n")
    for row in rows:
        print(row)
    if not report["all_valid"]:
        raise SystemExit("Invalid continuous scene geometry")


if __name__ == "__main__":
    main()
