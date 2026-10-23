"""Retain the actor's terminal grid and exact fused range packets at V710's stop."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

import run_robocasa_layout2_live_station_v710 as live
import run_robocasa_layout2_multitrack_v485 as multi
import run_robocasa_layout2_route_v407 as route


SOURCE = Path(__file__)
OUT = live.side.ROOT / "artifacts/ape-robocasa-layout2-live-map-v711"


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    old_out = live.OUT
    try:
        live.OUT = OUT
        live.main()
    finally:
        live.OUT = old_out
    grid = multi.CAPTURED[-1]
    grid_path = OUT / "actor-terminal-grid.npz"
    np.savez_compressed(grid_path, xs_m=grid.xs, ys_m=grid.ys,
                        free=grid.free, occupied=grid.occupied,
                        visits=grid.visits)
    path = OUT / "report.json"
    report = json.loads(path.read_text())
    episode = report["episode"]
    result = {"schema": "robot-learning-lab.layout2-live-map-packet-probe.v1",
              "source_sha256": route.sha256(SOURCE),
              "trial_report_sha256": route.sha256(path),
              "terminal_grid_path": str(grid_path),
              "terminal_grid_sha256": route.sha256(grid_path),
              "route_range_packets": len(episode["route_episode"]["trace"]),
              "all_route_packets_have_72_beams": all(
                  len(row["range_72_fused_m"]) == 72
                  for row in episode["route_episode"]["trace"]),
              "all_capture_times_ordered": all(
                  a["time_s"] < b["time_s"] for a, b in zip(
                      episode["route_episode"]["trace"],
                      episode["route_episode"]["trace"][1:])),
              "route_state": episode["route_state"],
              "goal_xy_m": episode["pre_route_goal_xy_m"]}
    (OUT / "map-probe.json").write_text(json.dumps(result, indent=2) + "\n")
    print(result, flush=True)


if __name__ == "__main__":
    main()
