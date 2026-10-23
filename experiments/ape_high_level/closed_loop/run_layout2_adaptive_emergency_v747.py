"""Actor-selected unqualified emergency move when every certified goal fails."""

from __future__ import annotations

import json
from pathlib import Path

import run_layout2_adaptive_refuge_v744 as adaptive
import run_layout2_west_station_approach_v736 as approach
import run_robocasa_layout2_route_v407 as route


SOURCE = Path(__file__)
OUT = adaptive.west.side.ROOT / "artifacts/ape-robocasa-layout2-adaptive-emergency-v747"


class EmergencyMonitor(adaptive.AdaptiveThreatMonitor):
    allow_best_effort = True


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    declaration = {"schema": "robot-learning-lab.layout2-adaptive-emergency.v1",
                   "control_report_sha256": route.sha256(
                       adaptive.west.side.ROOT /
                       "artifacts/ape-robocasa-layout2-adaptive-west-approach-v745/report.json"),
                   "changed_variable": "When no candidate meets every hypothesis, dispatch the actor-map-known route with the largest lower predicted gap only if it improves on current hold by >=15cm and clears other tracked objects. Preserve unqualified status.",
                   "same_prop": "V736 fixture-clear west approach",
                   "source_sha256": route.sha256(SOURCE),
                   "selector_source_sha256": route.sha256(Path(adaptive.selector.__file__))}
    (OUT / "emergency-declaration.json").write_text(json.dumps(declaration, indent=2) + "\n")
    adaptive.main(prop_factory=approach.WestApproachProp, out=OUT,
                  profile="capture-triggered direct west approach, unqualified actor-ranked emergency",
                  monitor_factory=EmergencyMonitor)
    path = OUT / "report.json"
    report = json.loads(path.read_text())
    report["emergency_declaration_sha256"] = route.sha256(OUT / "emergency-declaration.json")
    path.write_text(json.dumps(report, indent=2) + "\n")


if __name__ == "__main__":
    main()
