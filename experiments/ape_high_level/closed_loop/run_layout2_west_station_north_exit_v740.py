"""Development north exit after V738's fixed south exit failed west approach."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

import run_layout2_west_station_approach_v736 as approach
import run_robocasa_layout2_live_inference_escape_v721 as response
import run_robocasa_layout2_route_v407 as route
import run_robocasa_person_joined_v497 as joined


SOURCE = Path(__file__)
OUT = approach.side.ROOT / "artifacts/ape-robocasa-layout2-west-station-north-exit-v740"
GOAL = np.array([.15, -2.65])


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    declaration = {"schema": "robot-learning-lab.layout2-west-station-north-exit.v1",
                   "development_goal_xy_m": GOAL.tolist(),
                   "goal_source": "Chosen after V738 failure; saved actor survey marks this northern cell known free. Not a general online goal selector.",
                   "changed_variable_vs_v738": "north rather than south fixed response goal",
                   "prop_path": "same capture-triggered west approach as V736/V738",
                   "persistent_20ms_monitor": True,
                   "source_sha256": route.sha256(SOURCE)}
    (OUT / "north-exit-declaration.json").write_text(json.dumps(declaration, indent=2) + "\n")
    previous = approach.OUT, joined.CONTINUE_INTERRUPTED_MONITOR, response.GOAL.copy()
    try:
        approach.OUT = OUT
        joined.CONTINUE_INTERRUPTED_MONITOR = True
        response.GOAL = GOAL.copy()
        approach.main()
    finally:
        approach.OUT, joined.CONTINUE_INTERRUPTED_MONITOR, response.GOAL = previous
    report_path = OUT / "report.json"
    report = json.loads(report_path.read_text())
    report["north_exit_declaration_sha256"] = route.sha256(OUT / "north-exit-declaration.json")
    report["development_response_goal_xy_m"] = GOAL.tolist()
    report_path.write_text(json.dumps(report, indent=2) + "\n")


if __name__ == "__main__":
    main()
