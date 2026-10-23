"""Side-path late prop approaching the RGB-D owner station during return."""

from __future__ import annotations

import json
from pathlib import Path

import mujoco
import numpy as np

from audit_robocasa_dynamic_suite_v440 import prop_fixture_gap
import robocasa_owner_person as person
import run_robocasa_layout2_owner_transfer_v695 as transfer
import run_robocasa_layout2_person_route_v697 as full
import run_robocasa_layout2_route_v407 as route


ROOT = Path(__file__).resolve().parents[3]
SOURCE = Path(__file__)
OUT = ROOT / "artifacts/ape-robocasa-layout2-side-prop-v701"
START = np.array([0., -4.7])
END = np.array([.6, -3.4])
BEGIN_S = 17.8
END_S = 21.8


def prop_at(t):
    fraction = np.clip((t - BEGIN_S) / (END_S - BEGIN_S), 0., 1.)
    return START + fraction * (END - START)


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    model = person.scene_with_person(START, layout_id=2, style_id=1, clutter_mode=0)
    data = mujoco.MjData(model)
    prop_body = model.body("route_obstacle").id
    prop_geom = model.geom("route_obstacle_geom").id
    ape_geom = model.geom("audit_ape_geom").id
    fixture_gaps = []
    for t in np.arange(BEGIN_S, END_S + .0001, .02):
        xy = prop_at(t)
        model.body_pos[prop_body, :2] = xy
        mujoco.mj_forward(model, data)
        gap, geom = prop_fixture_gap(model, data, prop_geom, ape_geom, xy)
        fixture_gaps.append((gap, float(t), geom))
    nearest = min(fixture_gaps)
    declaration = {"schema": "robot-learning-lab.layout2-side-prop-declaration.v1",
                   "scene": "RoboCasa layout2/style1, shifted owner from V697",
                   "prop_start_xy_m": START.tolist(), "prop_end_xy_m": END.tolist(),
                   "motion_begin_s": BEGIN_S, "motion_end_s": END_S,
                   "speed_m_s": float(np.linalg.norm(END - START) / (END_S - BEGIN_S)),
                   "fixture_clearance_preflight_evaluator_only": {
                       "samples_20ms": len(fixture_gaps),
                       "minimum_signed_gap_m_capped_at_30cm": nearest[0],
                       "time_s": nearest[1], "nearest_geom": nearest[2]},
                   "comparison": "Same layout2 RGB-D owner station and memory loop as V697, low prop begins farther left and moves along a room-clear side path during the away/return leg. Score actor use, complete task and time-continuous body margin through slow inference.",
                   "source_sha256": route.sha256(SOURCE),
                   "full_runner_source_sha256": route.sha256(full.SOURCE)}
    assert nearest[0] >= .05
    (OUT / "predeclared.json").write_text(json.dumps(declaration, indent=2) + "\n")
    previous = (full.OUT, full.PROP_SCRIPT_FACTORY, full.RUN_SCOPE,
                transfer.PROP_START.copy())
    try:
        full.OUT = OUT
        full.PROP_SCRIPT_FACTORY = lambda _: prop_at
        full.RUN_SCOPE = "actual layout2/style1 owner RGB-D station and full memory loop with fixture-clear late side-path moving low prop; 50cm kinematic proxy; actor timed range and image observations only"
        transfer.PROP_START = START.copy()
        full.main()
    finally:
        full.OUT, full.PROP_SCRIPT_FACTORY, full.RUN_SCOPE, transfer.PROP_START = previous
    report_path = OUT / "report.json"
    report = json.loads(report_path.read_text())
    report["schema"] = "robot-learning-lab.layout2-side-prop-result.v1"
    report["side_prop_source_sha256"] = route.sha256(SOURCE)
    report["predeclaration_sha256"] = route.sha256(OUT / "predeclared.json")
    report["script"] = {"start_xy_m": START.tolist(), "end_xy_m": END.tolist(),
                        "begin_s": BEGIN_S, "end_s": END_S}
    report_path.write_text(json.dumps(report, indent=2) + "\n")


if __name__ == "__main__":
    main()
