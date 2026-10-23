"""Full owner-task control and straight prop at the V728 west-view station.

Keep V721's existing live range interrupt and exit as an unmodified baseline.
The purpose is to expose what changes when only owner-station placement moves.
"""

from __future__ import annotations

import json
from pathlib import Path

import run_robocasa_layout2_joint_station_v712 as station
import run_robocasa_layout2_live_inference_escape_v721 as baseline
import run_robocasa_layout2_route_v407 as route


SOURCE = Path(__file__)
OUT = station.side.ROOT / "artifacts/ape-robocasa-layout2-west-station-baseline-v729"
STANDOFF_M = .40
LATERAL_M = -.70


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    declaration = {"schema": "robot-learning-lab.layout2-west-station-baseline.v1",
                   "station_source": "V727 saved-survey screen and V728 fresh rendered RGB-D owner view",
                   "station_rule": {"standoff_m": STANDOFF_M, "lateral_m": LATERAL_M},
                   "changed_variable": "first-person-surface-derived owner station",
                   "control": "same V721 live range-track motion threshold and fixed development exit",
                   "profiles": ["stationary", "capture_triggered_straight_prop"],
                   "source_sha256": route.sha256(SOURCE),
                   "baseline_source_sha256": route.sha256(Path(baseline.__file__))}
    (OUT / "predeclared.json").write_text(json.dumps(declaration, indent=2) + "\n")
    previous = (station.STANDOFF_M, station.LATERAL_M, baseline.OUT)
    try:
        station.STANDOFF_M, station.LATERAL_M = STANDOFF_M, LATERAL_M
        baseline.OUT = OUT
        baseline.main()
    finally:
        station.STANDOFF_M, station.LATERAL_M, baseline.OUT = previous


if __name__ == "__main__":
    main()
