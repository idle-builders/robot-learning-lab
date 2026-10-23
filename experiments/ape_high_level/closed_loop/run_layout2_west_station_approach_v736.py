"""Held-out fixture-clear prop approach directly toward the west station."""

from __future__ import annotations

import json
from pathlib import Path

import mujoco
import numpy as np

from audit_robocasa_dynamic_suite_v440 import prop_fixture_gap
import robocasa_owner_person as person
import run_layout2_west_station_threat_gate_v732 as trial
import run_robocasa_layout2_owner_inference_trigger_v716 as trigger
import run_robocasa_layout2_route_v407 as route
import run_robocasa_layout2_side_prop_v701 as side


SOURCE = Path(__file__)
OUT = side.ROOT / "artifacts/ape-robocasa-layout2-west-station-approach-v736"
END = np.array([.2, -3.1])
DURATION_S = 2.5


class WestApproachProp(trigger.CaptureTriggeredProp):
    def __call__(self, time_s):
        if self.capture_time_s is None:
            return side.START.copy()
        fraction = np.clip((time_s - self.capture_time_s - .15) /
                           DURATION_S, 0., 1.)
        return side.START + fraction * (END - side.START)


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    model = person.scene_with_person(side.START, layout_id=2, style_id=1,
                                     clutter_mode=0)
    data = mujoco.MjData(model)
    body = model.body("route_obstacle").id
    geom = model.geom("route_obstacle_geom").id
    ape_geom = model.geom("audit_ape_geom").id
    gaps = []
    for fraction in np.linspace(0., 1., 126):
        xy = side.START + fraction * (END - side.START)
        model.body_pos[body, :2] = xy
        mujoco.mj_forward(model, data)
        gaps.append(prop_fixture_gap(model, data, geom, ape_geom, xy))
    nearest = min(gaps, key=lambda row: row[0])
    assert nearest[0] >= .05
    declaration = {"schema": "robot-learning-lab.layout2-west-station-approach.v1",
                   "frozen_station_and_threat_source_sha256": route.sha256(Path(trial.__file__)),
                   "prop_start_xy_m": side.START.tolist(),
                   "prop_end_xy_m": END.tolist(),
                   "motion_start_after_capture_s": .15,
                   "duration_s": DURATION_S,
                   "fixture_minimum_gap_m_capped_at_30cm": nearest[0],
                   "fixture_nearest_geom": nearest[1],
                   "source_sha256": route.sha256(SOURCE)}
    (OUT / "approach-declaration.json").write_text(json.dumps(declaration, indent=2) + "\n")
    trial.main(prop_factory=WestApproachProp, out=OUT,
               profile="fixture-clear prop approaches the west station after owner item capture")
    report_path = OUT / "report.json"
    report = json.loads(report_path.read_text())
    report["approach_declaration_sha256"] = route.sha256(OUT / "approach-declaration.json")
    report_path.write_text(json.dumps(report, indent=2) + "\n")


if __name__ == "__main__":
    main()
