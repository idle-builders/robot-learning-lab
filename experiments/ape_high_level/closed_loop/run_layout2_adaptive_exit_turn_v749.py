"""Held-out turn toward V747's actor-selected northwest emergency goal."""

from __future__ import annotations

import json
from pathlib import Path

import mujoco
import numpy as np

from audit_robocasa_dynamic_suite_v440 import prop_fixture_gap
import robocasa_owner_person as person
import run_layout2_adaptive_emergency_v747 as emergency
import run_layout2_adaptive_refuge_v744 as adaptive
import run_layout2_west_station_approach_v736 as approach
import run_robocasa_layout2_route_v407 as route
import run_robocasa_layout2_side_prop_v701 as side


SOURCE = Path(__file__)
OUT = side.ROOT / "artifacts/ape-robocasa-layout2-adaptive-exit-turn-v749"
TURN_AFTER_CAPTURE_S = .8
FINAL_AFTER_CAPTURE_S = 2.65
END = np.array([-.3, -2.55])
MID = side.START + ((TURN_AFTER_CAPTURE_S - .15) / approach.DURATION_S) * (
    approach.END - side.START)


class ExitTurnProp(approach.WestApproachProp):
    def __call__(self, time_s):
        if self.capture_time_s is None:
            return side.START.copy()
        elapsed = time_s - self.capture_time_s
        if elapsed <= TURN_AFTER_CAPTURE_S:
            return super().__call__(time_s)
        fraction = np.clip((elapsed - TURN_AFTER_CAPTURE_S) /
                           (FINAL_AFTER_CAPTURE_S - TURN_AFTER_CAPTURE_S),
                           0., 1.)
        return MID + fraction * (END - MID)


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    model = person.scene_with_person(side.START, layout_id=2, style_id=1,
                                     clutter_mode=0)
    data = mujoco.MjData(model)
    body = model.body("route_obstacle").id
    geom = model.geom("route_obstacle_geom").id
    ape_geom = model.geom("audit_ape_geom").id
    gaps = []
    prop = ExitTurnProp()
    prop.arm(0.)
    for t in np.arange(0., FINAL_AFTER_CAPTURE_S + .01, .02):
        xy = prop(float(t))
        model.body_pos[body, :2] = xy
        mujoco.mj_forward(model, data)
        gaps.append(prop_fixture_gap(model, data, geom, ape_geom, xy))
    fixture = min(gaps, key=lambda row: row[0])
    assert fixture[0] >= .05
    declaration = {"schema": "robot-learning-lab.layout2-adaptive-exit-turn.v1",
                   "development_control_report_sha256": route.sha256(emergency.OUT / "report.json"),
                   "frozen_selector_sha256": route.sha256(Path(adaptive.selector.__file__)),
                   "same_prop_path_until_after_capture_s": TURN_AFTER_CAPTURE_S,
                   "new_endpoint_xy_m": END.tolist(),
                   "fixture_minimum_gap_m_capped_at_30cm": fixture[0],
                   "fixture_nearest_geom": fixture[1],
                   "chosen_after": "V747 selected the northwest goal; this is a held-out bearing challenge of that rule, not a calibration path.",
                   "source_sha256": route.sha256(SOURCE)}
    (OUT / "turn-declaration.json").write_text(json.dumps(declaration, indent=2) + "\n")
    adaptive.main(prop_factory=ExitTurnProp, out=OUT,
                  profile="west approach turns toward V747 northwest exit after 0.8s",
                  monitor_factory=emergency.EmergencyMonitor)
    path = OUT / "report.json"
    report = json.loads(path.read_text())
    report["turn_declaration_sha256"] = route.sha256(OUT / "turn-declaration.json")
    path.write_text(json.dumps(report, indent=2) + "\n")


if __name__ == "__main__":
    main()
