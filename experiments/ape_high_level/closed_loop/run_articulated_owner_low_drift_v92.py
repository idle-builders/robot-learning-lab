"""V92: a low prop keeps moving through the articulated owner's show-again.

The V343 furnished task, person, cameras and six-shin guard stay fixed. After
the guard's blocked-route response, the prop continues toward the held ape at
12 cm/s through the owner presentation and first fresh view, instead of
stopping at 25.8 cm.
This is a scripted development challenge, not a physical obstacle model.
"""

from __future__ import annotations

import json
from pathlib import Path

from articulated_person_scene import add_person
from direct_structural_rooms_v270 import selected_visual
from run import ROOT, sha
import run_blocked_owner_help_v317 as owner_module
import run_live_articulated_owner_v343 as v343
import run_live_moving_owner_offer_v338 as prior
from run_late_corridor_intrusion_v323 import LateCorridorRoom
from run_moving_low_clutter_v302 import STOOL_SIZE


SOURCE = Path(__file__)
OUT = ROOT / "artifacts/ape-closed-loop-cove50-v92"
OFFER_START_S = 16.2
START_Y_M = .306
END_Y_M = .210
OFFER_DURATION_S = .8
CASES = tuple(case for case in owner_module.CASES
              if case[0] in (170, 172) and case[1] == "present")


class OfferDriftRoom(LateCorridorRoom):
    def _update_moving_obstacle(self) -> None:
        super()._update_moving_obstacle()
        if self.motion_side is None or self.time_s < OFFER_START_S:
            return
        fraction = min(1., max(0., (self.time_s - OFFER_START_S) /
                               OFFER_DURATION_S))
        y = START_Y_M + fraction * (END_Y_M - START_Y_M)
        self.obstacle_xy[3] = (0., y)
        self.model.body_pos[self.model.body("loop_stool").id] = (
            0., y, STOOL_SIZE[2])


def main() -> None:
    original = (prior.OUT, prior.add_owner_hand,
                prior.move_owner_during_wait, prior.LateCorridorRoom,
                prior.CASES, owner_module.OWNER_SHOW_XYZ_M)
    try:
        prior.OUT = OUT
        prior.add_owner_hand = add_person
        prior.move_owner_during_wait = v343.move_person_during_wait
        prior.LateCorridorRoom = OfferDriftRoom
        prior.CASES = CASES
        owner_module.OWNER_SHOW_XYZ_M = v343.OWNER_SHOW_XYZ_M
        prior.main()
    finally:
        (prior.OUT, prior.add_owner_hand,
         prior.move_owner_during_wait, prior.LateCorridorRoom,
         prior.CASES, owner_module.OWNER_SHOW_XYZ_M) = original
    audit_path = OUT / "live-moving-owner-offer-audit.json"
    report = json.loads(audit_path.read_text())
    report.update({
        "schema": "robot-learning-lab.ape-articulated-owner-low-drift.v1",
        "source_sha256": sha(SOURCE),
        "v338_source_sha256": sha(Path(prior.__file__)),
        "v343_source_sha256": sha(Path(v343.__file__)),
        "selected_visual_sha256": sha(selected_visual()),
        "offer_drift": {
            "start_time_s": OFFER_START_S,
            "start_y_m": START_Y_M,
            "end_y_m": END_Y_M,
            "duration_s": OFFER_DURATION_S,
            "speed_m_s": (START_Y_M-END_Y_M)/OFFER_DURATION_S,
        },
        "scope": (
            "Furnished historical Cove-50 Direct Frame owner-help task with "
            "articulated person, mounted cameras/depth, six noisy shin views, "
            "joint readings and scripted low prop. The moving branch changes "
            "only the low prop's final 0.8s approach through owner presentation "
            "and the first fresh view. "
            "The parked branch is the unchanged V343 parked control. Two "
            "predeclared lighting seeds, both with the taught item present. "
            "Contact and 5cm-margin breaches are evaluator-only; no selected "
            "physical frame, dynamic stop or motor response is qualified."
        ),
    })
    audit_path.write_text(json.dumps(report, indent=2) + "\n")
    (OUT / SOURCE.name).write_bytes(SOURCE.read_bytes())
    print(audit_path)


if __name__ == "__main__":
    main()
