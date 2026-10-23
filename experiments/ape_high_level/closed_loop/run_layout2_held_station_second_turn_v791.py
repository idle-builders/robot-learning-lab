"""Challenge a frozen held-view refind with a second fixture-clear prop turn."""

from __future__ import annotations

import json
from pathlib import Path

import mujoco
import numpy as np

from audit_robocasa_dynamic_suite_v440 import prop_fixture_gap
import robocasa_owner_person as person
import run_layout2_adaptive_exit_turn_v749 as first_turn
import run_layout2_held_station_view_probe_v781 as probe
import run_robocasa_layout2_route_v407 as route
import run_robocasa_layout2_side_prop_v701 as side


SOURCE = Path(__file__)
OUT = probe.east.probe.side.ROOT / "artifacts/ape-robocasa-layout2-held-station-second-turn-v791"
SECOND_START_AFTER_CAPTURE_S = 6.4
SECOND_END_AFTER_CAPTURE_S = 8.9
SECOND_END = np.array([.8, -3.5])


class SecondTurnProp(first_turn.ExitTurnProp):
    def __call__(self, time_s):
        if self.capture_time_s is None:
            return side.START.copy()
        elapsed = time_s - self.capture_time_s
        if elapsed <= SECOND_START_AFTER_CAPTURE_S:
            return super().__call__(time_s)
        fraction = np.clip((elapsed - SECOND_START_AFTER_CAPTURE_S) /
                           (SECOND_END_AFTER_CAPTURE_S -
                            SECOND_START_AFTER_CAPTURE_S), 0., 1.)
        return first_turn.END + fraction * (SECOND_END - first_turn.END)


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    model = person.scene_with_person(side.START, layout_id=2, style_id=1,
                                     clutter_mode=0)
    data = mujoco.MjData(model)
    body = model.body("route_obstacle").id
    geom = model.geom("route_obstacle_geom").id
    ape_geom = model.geom("audit_ape_geom").id
    prop = SecondTurnProp()
    prop.arm(0.)
    fixture = []
    for t in np.arange(0., SECOND_END_AFTER_CAPTURE_S + .01, .02):
        xy = prop(float(t))
        model.body_pos[body, :2] = xy
        mujoco.mj_forward(model, data)
        fixture.append(prop_fixture_gap(model, data, geom, ape_geom, xy))
    closest_fixture = min(fixture, key=lambda row: row[0])
    assert closest_fixture[0] >= .05
    declaration = {
        "schema": "robot-learning-lab.layout2-held-station-second-turn.v1",
        "source_sha256": route.sha256(SOURCE),
        "reference_audit_sha256": route.sha256(
            probe.east.probe.side.ROOT /
            "artifacts/ape-robocasa-layout2-held-station-recheck-v789/audit-v790.json"),
        "frozen_actor_rule": "V789 held-view recheck, 16s probe, 40cm raw-view gate",
        "same_prop_path_until_after_capture_s": SECOND_START_AFTER_CAPTURE_S,
        "second_turn_end_after_capture_s": SECOND_END_AFTER_CAPTURE_S,
        "second_endpoint_xy_m": SECOND_END.tolist(),
        "fixture_minimum_gap_m_capped_at_30cm": closest_fixture[0],
        "fixture_nearest_geom": closest_fixture[1],
        "evaluation": "complete static control and second-turn owner task, independent 20ms geometry/media replay; actor receives no future turn"}
    (OUT / "challenge-declaration.json").write_text(
        json.dumps(declaration, indent=2) + "\n")
    probe.main(out=OUT, probe_min_raw_range_m=.40, recheck=True,
               turned_factory=SecondTurnProp)
    for label in ("static", "turned"):
        path = OUT / label / "report.json"
        report = json.loads(path.read_text())
        report["second_turn_declaration_sha256"] = route.sha256(
            OUT / "challenge-declaration.json")
        path.write_text(json.dumps(report, indent=2) + "\n")


if __name__ == "__main__":
    main()
