"""Ask for a view-facing owner show at a mapped east-side station."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

import person_roi_depth_health_v774 as roi
import probe_layout2_cooperative_east_show_v778 as probe
import robocasa_owner_reacquire_rgbd_v560 as detector
import robocasa_person_interaction as interaction
import run_layout2_adaptive_refuge_v744 as adaptive
import run_layout2_adaptive_renewed_hold_v757 as renewed
import run_layout2_adaptive_exit_turn_v749 as turn
import run_layout2_northwest_owner_station_v775 as static
import run_layout2_west_station_baseline_v729 as station
import run_robocasa_layout2_person_route_v697 as full
import run_robocasa_layout2_route_v407 as route
import run_robocasa_person_joined_v497 as joined


SOURCE = Path(__file__)
OUT = probe.side.ROOT / "artifacts/ape-robocasa-layout2-cooperative-east-show-v779"
STANDOFF_M = .70
LATERAL_M = .085
SHOW_HAND_WORLD_M = np.array([.7, -2.75, .55])
OWNER_SHIFT_X_M = -.3


def main(*, turned_factory=turn.ExitTurnProp):
    OUT.mkdir(parents=True, exist_ok=True)
    declaration = {
        "schema": "robot-learning-lab.layout2-cooperative-east-show.v1",
        "station": "initial RGB-D person surface + 0.70m start-side standoff + 0.085m lateral",
        "station_source": "V778 east-middle static view; selected after V776 northwest moving contact",
        "scripted_owner_show_hand_world_xyz_m": SHOW_HAND_WORLD_M.tolist(),
        "presentation_source": "V778 center-front rendered can pixels and metric person; explicit high-level request for owner to show toward east camera",
        "profiles": ["static", "capture_triggered_direct_then_northwest_turn"],
        "same_virtual_range_V757_replan_and_ROI_depth": True,
        "no_future_prop_path_in_actor_station_or_response": True,
        "source_sha256": route.sha256(SOURCE),
        "probe_report_sha256": route.sha256(probe.OUT / "report.json")}
    (OUT / "predeclared.json").write_text(json.dumps(declaration, indent=2) + "\n")
    old = (station.STANDOFF_M, station.LATERAL_M,
           interaction.SHOW_HAND.copy(), detector.observe,
           full.pre_owner_hook, joined.CONTINUE_INTERRUPTED_MONITOR)

    def request_view_facing_show(**kwargs):
        result = old[4](**kwargs)
        if result["proceed"]:
            result["trace"].append({
                "stage": "request_owner_show_toward_east_view",
                "time_s": round(result["end_time_s"], 4),
                "speech_intent": "Could you hold the can toward me so I can see it?",
                "actor_station_estimated_xy_m": kwargs["estimate"][:2].tolist()})
        return result

    try:
        station.STANDOFF_M, station.LATERAL_M = STANDOFF_M, LATERAL_M
        interaction.SHOW_HAND = (SHOW_HAND_WORLD_M -
                                 np.array([OWNER_SHIFT_X_M, 0., 0.]))
        detector.observe = roi.observe
        full.pre_owner_hook = request_view_facing_show
        joined.CONTINUE_INTERRUPTED_MONITOR = True
        for label, factory in (("static", static.StaticProp),
                               ("turned", turned_factory)):
            adaptive.west.main(prop_factory=factory, out=OUT / label,
                               profile=f"owner shows toward east station, {label}",
                               monitor_factory=renewed.ReplanningEmergencyMonitor)
            path = OUT / label / "report.json"
            report = json.loads(path.read_text())
            report["cooperative_show_declaration_sha256"] = route.sha256(
                OUT / "predeclared.json")
            path.write_text(json.dumps(report, indent=2) + "\n")
    finally:
        (station.STANDOFF_M, station.LATERAL_M, interaction.SHOW_HAND,
         detector.observe, full.pre_owner_hook,
         joined.CONTINUE_INTERRUPTED_MONITOR) = old


if __name__ == "__main__":
    main()
