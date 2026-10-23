"""Keep the V736 range-guided interrupt alive after image inference ends."""

from __future__ import annotations

import json
from pathlib import Path

import run_layout2_west_station_approach_v736 as approach
import run_robocasa_layout2_route_v407 as route
import run_robocasa_person_joined_v497 as joined


SOURCE = Path(__file__)
OUT = approach.side.ROOT / "artifacts/ape-robocasa-layout2-west-station-persistent-v738"


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    declaration = {"schema": "robot-learning-lab.layout2-west-station-persistent.v1",
                   "changed_variable": "continue the identical actor inference monitor at 20ms after the image worker ends",
                   "control_report_sha256": route.sha256(approach.OUT / "report.json"),
                   "prop_and_station": "same actor station, threat rule, and fixture-clear west approach as V736",
                   "source_sha256": route.sha256(SOURCE),
                   "joined_source_sha256": route.sha256(Path(joined.__file__))}
    (OUT / "persistent-declaration.json").write_text(json.dumps(declaration, indent=2) + "\n")
    previous = approach.OUT, joined.CONTINUE_INTERRUPTED_MONITOR
    try:
        approach.OUT = OUT
        joined.CONTINUE_INTERRUPTED_MONITOR = True
        approach.main()
    finally:
        approach.OUT, joined.CONTINUE_INTERRUPTED_MONITOR = previous
    report_path = OUT / "report.json"
    report = json.loads(report_path.read_text())
    report["persistent_declaration_sha256"] = route.sha256(OUT / "persistent-declaration.json")
    report_path.write_text(json.dumps(report, indent=2) + "\n")


if __name__ == "__main__":
    main()
