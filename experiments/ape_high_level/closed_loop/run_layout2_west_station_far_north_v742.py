"""Development margin sweep: extend V740's north exit by 15 cm."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

import run_layout2_west_station_north_exit_v740 as north
import run_robocasa_layout2_route_v407 as route


SOURCE = Path(__file__)
OUT = north.approach.side.ROOT / "artifacts/ape-robocasa-layout2-west-station-far-north-v742"
GOAL = np.array([.15, -2.50])


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    declaration = {"schema": "robot-learning-lab.layout2-west-station-far-north.v1",
                   "development_goal_xy_m": GOAL.tolist(),
                   "goal_source": "Chosen after V741's 5.5cm minimum gap, to test whether a farther mapped cell buys useful margin.",
                   "changed_variable_vs_v740": "north goal y -2.50m rather than -2.65m",
                   "source_sha256": route.sha256(SOURCE)}
    (OUT / "far-north-declaration.json").write_text(json.dumps(declaration, indent=2) + "\n")
    previous = north.OUT, north.GOAL.copy()
    try:
        north.OUT, north.GOAL = OUT, GOAL.copy()
        north.main()
    finally:
        north.OUT, north.GOAL = previous
    report_path = OUT / "report.json"
    report = json.loads(report_path.read_text())
    report["far_north_declaration_sha256"] = route.sha256(OUT / "far-north-declaration.json")
    report_path.write_text(json.dumps(report, indent=2) + "\n")


if __name__ == "__main__":
    main()
