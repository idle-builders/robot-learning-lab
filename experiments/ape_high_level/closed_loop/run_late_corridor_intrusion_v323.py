"""V323: challenge recent free rays with a bounded late low-prop intrusion.

The room matches V322 until t=15.6 s. An 8 cm prop then approaches the
forward corridor at 0.24 m/s, including during any forward recovery command.
This is a deliberately adverse actor-blind dynamics change; evaluator truth
scores clearance but does not enter the route gate.
"""

from __future__ import annotations

import json
from pathlib import Path

from depth_quality import reject_isolated_depth_spikes
from run import ROOT
from run_blocked_forward_viewpoint_v321 import main as run_live
import run_blocked_forward_viewpoint_v321 as live_module
from run_heldout_moving_low_v305 import HeldoutMovingRoom
from run_moving_low_clutter_v302 import STOOL_SIZE


SOURCE = Path(__file__)
OUT = ROOT / "artifacts/ape-closed-loop-cove50-v323"
INTRUSION_START_S = 15.6
INTRUSION_SPEED_M_S = .24
FINAL_Y_M = .258


class LateCorridorRoom(HeldoutMovingRoom):
    def _update_moving_obstacle(self) -> None:
        super()._update_moving_obstacle()
        if self.motion_side is None or self.time_s < INTRUSION_START_S:
            return
        y = max(FINAL_Y_M,
                .45 - INTRUSION_SPEED_M_S *
                (self.time_s - INTRUSION_START_S))
        self.obstacle_xy[3] = (0., y)
        self.model.body_pos[self.model.body("loop_stool").id] = (
            0., y, STOOL_SIZE[2])


def main() -> None:
    original = live_module.HeldoutMovingRoom
    try:
        live_module.HeldoutMovingRoom = LateCorridorRoom
        run_live(output_root=OUT, source=SOURCE,
                 map_depth_filter=reject_isolated_depth_spikes,
                 experiment_label="V323")
    finally:
        live_module.HeldoutMovingRoom = original
    path = OUT / "blocked-forward-viewpoint-audit.json"
    report = json.loads(path.read_text())
    report["late_intrusion_evaluator_setup"] = {
        "prop_height_m": 2 * STOOL_SIZE[2],
        "motion_start_s": INTRUSION_START_S,
        "initial_y_m": .45,
        "inward_speed_m_s": INTRUSION_SPEED_M_S,
        "final_y_m": FINAL_Y_M,
        "scope": "The original case preflight validates the shared initial room only; this late trajectory intentionally approaches the robot after the search and is scored by the evaluator throughout the live replay."}
    path.write_text(json.dumps(report, indent=2) + "\n")


if __name__ == "__main__":
    main()
