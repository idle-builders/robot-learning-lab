"""Try the mapped northwest view with stronger person detection at arrival."""

from __future__ import annotations

import json
from pathlib import Path

import robocasa_owner_reacquire_rgbd_v560 as detector
import run_layout2_adaptive_refuge_v744 as adaptive
import run_layout2_adaptive_renewed_hold_v757 as renewed
import run_layout2_adaptive_exit_turn_v749 as turn
import run_layout2_northwest_owner_station_v775 as previous
import run_layout2_west_station_baseline_v729 as station
import run_robocasa_layout2_route_v407 as route
import run_robocasa_person_joined_v497 as joined


SOURCE = Path(__file__)
OUT = previous.side.ROOT / "artifacts/ape-robocasa-layout2-northwest-owner-station-v776"
STANDOFF_M = .04
LATERAL_M = -1.04


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    declaration = {
        "schema": "robot-learning-lab.layout2-northwest-owner-station-v776.v1",
        "prior_failed_station_report_sha256": route.sha256(
            previous.OUT / "static/report.json"),
        "changed_station_rule": {"standoff_m": STANDOFF_M,
                                 "lateral_m": LATERAL_M},
        "source": "V772/V773 northwest static view had person score 0.36, valid person ROI 0.72, shown can 252 pixels; V775 far northwest arrival lost person detection",
        "same_actor_person_roi_gate_and_V757_replanning_monitor": True,
        "profiles": ["static", "capture_triggered_direct_then_northwest_turn"],
        "source_sha256": route.sha256(SOURCE)}
    (OUT / "predeclared.json").write_text(json.dumps(declaration, indent=2) + "\n")
    old = (station.STANDOFF_M, station.LATERAL_M,
           detector.observe, joined.CONTINUE_INTERRUPTED_MONITOR)
    try:
        station.STANDOFF_M, station.LATERAL_M = STANDOFF_M, LATERAL_M
        detector.observe = previous.roi.observe
        joined.CONTINUE_INTERRUPTED_MONITOR = True
        for label, factory in (("static", previous.StaticProp),
                               ("turned", turn.ExitTurnProp)):
            adaptive.west.main(prop_factory=factory, out=OUT / label,
                               profile=f"northwest station V776 {label}",
                               monitor_factory=renewed.ReplanningEmergencyMonitor)
            path = OUT / label / "report.json"
            report = json.loads(path.read_text())
            report["northwest_station_declaration_sha256"] = route.sha256(
                OUT / "predeclared.json")
            path.write_text(json.dumps(report, indent=2) + "\n")
    finally:
        (station.STANDOFF_M, station.LATERAL_M, detector.observe,
         joined.CONTINUE_INTERRUPTED_MONITOR) = old


if __name__ == "__main__":
    main()
