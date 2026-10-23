"""Joined route/owner-turn screen with a second articulated person present."""

from __future__ import annotations

import json
from pathlib import Path

import mujoco
import numpy as np
from PIL import Image

import robocasa_first_return_person_reacquire_turn_v576 as turn
import robocasa_owner_person as person
import robocasa_person_interaction as interaction
import run_robocasa_joined_owner_route_v410 as owner
import run_robocasa_layout2_route_v407 as route
import run_robocasa_person_joined_v497 as joined
from run_robocasa_dynamic_suite_v439 import motion
from run_robocasa_first_return_person_reacquire_v577 import owner_scan
import screen_robocasa_two_people_v582 as two


ROOT = Path(__file__).resolve().parents[3]
OUT = ROOT / "artifacts/ape-robocasa-two-people-joined-v584"
SCENE_XML = ROOT / "artifacts/ape-robocasa-two-people-v582/two-people.xml"


def scene_with_two_people(start_xy):
    model = mujoco.MjModel.from_xml_path(str(SCENE_XML))
    model.body_pos[model.body("route_obstacle").id, :2] = start_xy
    person.set_owner_pose(model, interaction.TORSO_XY,
                          interaction.REST_HAND, show_can=False)
    two.set_distractor(model, two.DISTRACTOR_TORSO_XY)
    return model


def main():
    policy = json.loads(joined.CONFIG.read_text())
    case = next(x for x in policy["cases"] if x["id"] == "late_crossing")
    prior = np.load(joined.PRIOR)
    OUT.mkdir(parents=True, exist_ok=True)
    original_scene = joined.scene_with_owner_and_prop
    original_turn = joined.owner_turn
    original_out = joined.OUT
    original_scan = turn.SCAN
    original_hand = interaction.SHOW_HAND.copy()
    original_margin = route.MARGIN_M
    vision = owner.Vision(teach_selector="generic-proposal")
    exemplar = vision.feature([Image.open(owner.TEACH_CROP).convert("RGB")])[0]
    try:
        joined.scene_with_owner_and_prop = scene_with_two_people
        joined.owner_turn = turn.owner_turn
        joined.OUT = OUT
        route.MARGIN_M = .15
        interaction.SHOW_HAND = np.asarray([1.30, -2.98, .55])
        turn.SCAN = owner_scan(True)
        turn.OWNER_APPROACH_M = 0.
        turn.FIRST_RETURN_OWNER_M = .4
        turn.FIRST_RETURN_OWNER_S = 1.
        turn.LATE_OWNER_APPROACH_M = 0.
        turn.SECOND_RETURN_OWNER_M = 0.
        turn.DEPTH_NOISE_SIGMA_M = 0.
        turn.DEPTH_DROPOUT_FRACTION = 0.
        turn.PROP_SCRIPT = motion(case)
        turn.PACKET_OUT = OUT / "two-person-depth-packets.npz"
        episode = joined.episode(case, "early_owner_mid_sector_blackout",
                                 True, prior, policy, vision, exemplar)
    finally:
        joined.scene_with_owner_and_prop = original_scene
        joined.owner_turn = original_turn
        joined.OUT = original_out
        turn.SCAN = original_scan
        interaction.SHOW_HAND = original_hand
        route.MARGIN_M = original_margin
    report = {"scope": "full furnished route and owner-turn with one extra stationary articulated person near the early-owner track; kinematic motion and scripted people; released visual models; no owner identity protection beyond the existing V576 rule",
              "source_sha256": route.sha256(Path(__file__)),
              "scene_xml_sha256": route.sha256(SCENE_XML),
              "turn_source_sha256": route.sha256(Path(turn.__file__)),
              "two_person_source_sha256": route.sha256(Path(two.__file__)),
              "prior_map_sha256": route.sha256(joined.PRIOR),
              "person_detector_weights_sha256": route.sha256(
                  turn.person_reacquire.DETECTOR / "model.safetensors"),
              "distractor_torso_xy_m": two.DISTRACTOR_TORSO_XY.tolist(),
              "episode": episode}
    (OUT / "report.json").write_text(json.dumps(report, indent=2) + "\n")
    print("outcome", episode["outcome"], "route_state", episode["route_state"],
          "video", episode["video"]["path"], flush=True)


if __name__ == "__main__":
    main()
