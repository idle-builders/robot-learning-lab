"""Test a mapped person-view station with a separate south/east refuge option."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

import run_robocasa_layout2_multitrack_v485 as multi
import run_robocasa_layout2_person_route_v697 as full
import run_robocasa_layout2_route_v407 as route
import run_robocasa_layout2_side_prop_v701 as side


SOURCE = Path(__file__)
OUT = side.ROOT / "artifacts/ape-robocasa-layout2-joint-station-v712"
STANDOFF_M = 1.04
LATERAL_M = .18
HORIZON_S = 23.


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    declaration = {"schema": "robot-learning-lab.layout2-joint-station-declaration.v1",
                   "scene": "same layout2/style1 owner and 17.8–21.8s side-prop path as V701",
                   "actor_station_rule": "first head RGB-D person surface +1.04m toward ape +0.18m lateral; surveyed grid cell approximately (1.1,-3.55)m",
                   "actor_route_evidence": "seed route grid from layout2's sensor-derived survey and update with fresh dual-height range; target cell free in V711's saved live grid; unknown cells are not free",
                   "development_view_preflight": "At approximately (1.14,-3.53)m the scripted person and shown item are in the aimed view and central depth validity is above the released 0.5 packet-health gate; this is preflight, not an actor decision.",
                   "candidate_not_qualified": "The preflight does not prove executed route, item find, moving held pose or selected-body clearance; score all through 23s",
                   "standoff_m": STANDOFF_M, "lateral_m": LATERAL_M,
                   "post_stop_until_s": HORIZON_S,
                   "source_sha256": route.sha256(SOURCE),
                   "prior_map_sha256": route.sha256(side.transfer.PRIOR)}
    (OUT / "predeclared.json").write_text(json.dumps(declaration, indent=2) + "\n")
    previous = (side.OUT, full.POST_STOP_UNTIL_S, full.STANDOFF_M,
                full.STATION_LATERAL_M, full.SEED_ROUTE_FROM_SURVEY)
    try:
        side.OUT = OUT
        full.POST_STOP_UNTIL_S = HORIZON_S
        full.STANDOFF_M = STANDOFF_M
        full.STATION_LATERAL_M = LATERAL_M
        full.SEED_ROUTE_FROM_SURVEY = True
        side.main()
    finally:
        (side.OUT, full.POST_STOP_UNTIL_S, full.STANDOFF_M,
         full.STATION_LATERAL_M, full.SEED_ROUTE_FROM_SURVEY) = previous
    grid = multi.CAPTURED[-1]
    grid_path = OUT / "actor-terminal-grid.npz"
    np.savez_compressed(grid_path, xs_m=grid.xs, ys_m=grid.ys,
                        free=grid.free, occupied=grid.occupied,
                        visits=grid.visits)
    path = OUT / "report.json"
    report = json.loads(path.read_text())
    report["schema"] = "robot-learning-lab.layout2-joint-station-result.v1"
    report["joint_station_source_sha256"] = route.sha256(SOURCE)
    report["predeclaration_sha256"] = route.sha256(OUT / "predeclared.json")
    report["terminal_grid_sha256"] = route.sha256(grid_path)
    path.write_text(json.dumps(report, indent=2) + "\n")
    ep = report["episode"]
    memory = ep["memory_return"] or {}
    print(ep["route_state"], ep["outcome"], memory.get("route_status"),
          memory.get("item_status"), memory.get("post_stop_minimum_proxy_gap_m_evaluator_only"),
          flush=True)


if __name__ == "__main__":
    main()
