"""Evaluator-only scene admission for a second furnished owner-task room.

The resulting placement may define a new simulation fixture. None of these
true room distances enter the actor's route or meeting-station decisions.
"""

from __future__ import annotations

import json
from pathlib import Path

import mujoco
import numpy as np

from audit_robocasa_dynamic_suite_v440 import prop_fixture_gap
from audit_robocasa_person_joined_v498 import nearest_can_room
import robocasa_owner_person as person
import robocasa_person_interaction as interaction
from run_robocasa_layout2_route_v407 import sha256
from screen_robocasa_owner_person_v489 import gap_to_room


ROOT = Path(__file__).resolve().parents[3]
SOURCE = Path(__file__)
OUT = ROOT / "artifacts/ape-robocasa-layout2-owner-scene-v694"
OWNER_SHIFTS_XY_M = ((-.3, 0.), (-.5, 0.), (-.6, 0.), (-.5, -.2))
PROP_POSITIONS_XY_M = ((3., -1.2), (2.5, -1.2), (3., -4.2), (2.3, -4.2))
APE_START_XY_M = np.array([1.05, -4.2])


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    declaration = OUT / "predeclared.json"
    declaration.write_text(json.dumps({
        "schema": "robot-learning-lab.layout2-owner-scene-admission.v1",
        "layout_id": 2,
        "style_id": 1,
        "clutter_mode": 0,
        "owner_shifts_xy_m": OWNER_SHIFTS_XY_M,
        "static_prop_candidates_xy_m": PROP_POSITIONS_XY_M,
        "criterion": "At least 5 cm signed person-to-room clearance in rest and full show, shown-can nonoverlap, static prop-to-fixture clearance and no ape-start overlap. This is evaluator scene setup, not actor planning evidence.",
        "source_sha256": sha256(SOURCE),
        "scene_source_sha256": sha256(Path(person.__file__)),
    }, indent=2) + "\n")
    model = person.scene_with_person(prop_xy=PROP_POSITIONS_XY_M[0], layout_id=2)
    data = mujoco.MjData(model)
    model.body_pos[model.body("audit_ape").id, :2] = APE_START_XY_M
    prop_body = model.body("route_obstacle").id
    prop_geom = model.geom("route_obstacle_geom").id
    ape_geom = model.geom("audit_ape_geom").id
    owner_rows = []
    for dx, dy in OWNER_SHIFTS_XY_M:
        shift = np.array([dx, dy])
        torso = interaction.TORSO_XY + shift
        phases = []
        for phase, hand in (("rest", interaction.REST_HAND.copy()),
                            ("show", interaction.SHOW_HAND.copy())):
            hand[:2] += shift
            pose = person.set_owner_pose(model, torso, hand, show_can=True)
            if phase == "show":
                direction = hand[:2] - torso
                direction /= np.linalg.norm(direction)
                model.body_pos[model.body("shown_can").id, :2] = (
                    hand[:2] + .14 * direction)
            mujoco.mj_forward(model, data)
            room, ape = gap_to_room(model, data)
            can = nearest_can_room(model, data)
            phases.append({"phase": phase, "pose": pose,
                           "person_room": room, "person_ape": ape,
                           "can_room": can})
        owner_rows.append({"shift_xy_m": shift.tolist(),
                           "phases": phases,
                           "placement_pass": all(
                               row["person_room"]["gap_m_capped_at_30cm"] >= .05 and
                               row["person_ape"]["gap_m_capped_at_50cm"] >= .05 and
                               row["can_room"]["gap_m_capped_at_10cm"] >= 0.
                               for row in phases)})
    person.set_owner_pose(model, [-10., -10.], [-10., -10., .43], show_can=False)
    prop_rows = []
    for xy in PROP_POSITIONS_XY_M:
        model.body_pos[prop_body, :2] = xy
        mujoco.mj_forward(model, data)
        gap, near = prop_fixture_gap(model, data, prop_geom, ape_geom,
                                     np.asarray(xy))
        prop_rows.append({"xy_m": xy, "fixture_gap_m_capped_at_30cm": gap,
                          "nearest_geom": near, "placement_pass": gap >= .05})
    report = {"schema": "robot-learning-lab.layout2-owner-scene-admission-result.v1",
              "declaration_sha256": sha256(declaration),
              "owner_candidates": owner_rows,
              "prop_candidates": prop_rows,
              "qualified_for_full_episode": False,
              "limits": "Scene geometry only; route, owner image visibility, sensor map, timing and all moving paths untested."}
    (OUT / "report.json").write_text(json.dumps(report, indent=2) + "\n")
    for row in owner_rows:
        print("owner", row["shift_xy_m"], row["placement_pass"],
              [(phase["phase"], phase["person_room"]["gap_m_capped_at_30cm"],
                phase["can_room"]["gap_m_capped_at_10cm"])
               for phase in row["phases"]], flush=True)
    for row in prop_rows:
        print("prop", row["xy_m"], row["placement_pass"],
              row["fixture_gap_m_capped_at_30cm"], flush=True)


if __name__ == "__main__":
    main()
