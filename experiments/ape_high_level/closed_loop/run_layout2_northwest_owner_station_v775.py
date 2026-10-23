"""Execute the northwest show view with person-local depth health."""

from __future__ import annotations

import json
from pathlib import Path

import person_roi_depth_health_v774 as roi
import robocasa_owner_reacquire_rgbd_v560 as detector
import run_layout2_adaptive_refuge_v744 as adaptive
import run_layout2_adaptive_renewed_hold_v757 as renewed
import run_layout2_adaptive_exit_turn_v749 as turn
import run_layout2_west_station_baseline_v729 as west
import run_robocasa_layout2_owner_inference_trigger_v716 as trigger
import run_robocasa_layout2_route_v407 as route
import run_robocasa_layout2_side_prop_v701 as side


SOURCE = Path(__file__)
OUT = side.ROOT / "artifacts/ape-robocasa-layout2-northwest-owner-station-v775"
STANDOFF_M = -.07
LATERAL_M = -1.12


class StaticProp(trigger.CaptureTriggeredProp):
    def __call__(self, t):
        return side.START.copy()


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    declaration = {
        "schema": "robot-learning-lab.layout2-northwest-owner-station.v1",
        "development_station_rule": {"standoff_m": STANDOFF_M,
                                     "lateral_m": LATERAL_M},
        "station_source": "V772/V773 stationary view/ROI depth preflight after V769 west-station stop-short failure",
        "person_local_depth_gate": "after broad depth-health failure, require DINO person score >=0.25 and >=60% valid 0.15-2.8m depth within its current box",
        "profiles": ["static", "capture_triggered_direct_then_northwest_turn"],
        "same_virtual_range_and_V757_replanning_monitor": True,
        "no_future_prop_path_in_actor_station_or_detection": True,
        "source_sha256": route.sha256(SOURCE),
        "roi_source_sha256": route.sha256(Path(roi.__file__)),
        "monitor_source_sha256": route.sha256(Path(renewed.__file__))}
    (OUT / "predeclared.json").write_text(json.dumps(declaration, indent=2) + "\n")
    previous = west.STANDOFF_M, west.LATERAL_M, detector.observe
    try:
        west.STANDOFF_M, west.LATERAL_M = STANDOFF_M, LATERAL_M
        detector.observe = roi.observe
        for label, factory in (("static", StaticProp),
                               ("turned", turn.ExitTurnProp)):
            adaptive.main(prop_factory=factory, out=OUT / label,
                          profile=f"northwest station, person ROI depth, {label} prop",
                          monitor_factory=renewed.ReplanningEmergencyMonitor)
            path = OUT / label / "report.json"
            report = json.loads(path.read_text())
            report["station_declaration_sha256"] = route.sha256(OUT / "predeclared.json")
            path.write_text(json.dumps(report, indent=2) + "\n")
    finally:
        west.STANDOFF_M, west.LATERAL_M, detector.observe = previous


if __name__ == "__main__":
    main()
