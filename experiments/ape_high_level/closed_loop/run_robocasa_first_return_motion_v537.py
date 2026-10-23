"""Join an approaching owner to the pre-show return and subsequent task."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
from PIL import Image

import robocasa_first_return_motion_turn_v536 as turn
import robocasa_owner_person as person
import robocasa_person_interaction as interaction
import run_robocasa_joined_owner_route_v410 as owner
import run_robocasa_layout2_route_v407 as route
import run_robocasa_person_joined_v497 as joined
from run_robocasa_dynamic_suite_v439 import motion


ROOT = Path(__file__).resolve().parents[3]
OUT = ROOT / "artifacts/ape-robocasa-first-return-motion-v537"
MATRIX = (("stationary_nearer_show", [1.30, -2.98, .55]),
          ("approach_30cm_before_show_nearer", [1.30, -2.98, .55]))


def main():
    policy = json.loads(joined.CONFIG.read_text())
    case = next(case for case in policy["cases"] if case["id"] == "late_crossing")
    prior = np.load(joined.PRIOR)
    OUT.mkdir(parents=True, exist_ok=True)
    joined.OUT = OUT
    joined.owner_turn = turn.owner_turn
    route.MARGIN_M = .15
    vision = owner.Vision(teach_selector="generic-proposal")
    exemplar = vision.feature([Image.open(owner.TEACH_CROP).convert("RGB")])[0]
    original_hand = interaction.SHOW_HAND.copy()
    episodes = []
    try:
        for response, hand in MATRIX:
            interaction.SHOW_HAND = np.asarray(hand)
            turn.OWNER_APPROACH_M = 0.
            turn.FIRST_RETURN_OWNER_M = .3
            turn.FIRST_RETURN_OWNER_S = 1.
            turn.SECOND_RETURN_OWNER_M = 0.
            turn.DEPTH_NOISE_SIGMA_M = 0.
            turn.DEPTH_DROPOUT_FRACTION = 0.
            turn.PROP_SCRIPT = motion(case)
            turn.PACKET_OUT = OUT / f"{response}-depth-packets.npz"
            result = joined.episode(case, response, True, prior, policy,
                                    vision, exemplar)
            episodes.append(result)
            chosen = next((row["chosen"] for row in result["trace"]
                           if row["stage"] == "choose_owner_standoff"), None)
            print(response, result["outcome"],
                  "standoff", chosen["distance_m"] if chosen else None,
                  "depth_error", result["depth_point_error_m"], flush=True)
    finally:
        interaction.SHOW_HAND = original_hand
        route.MARGIN_M = .10
        route.LIDAR_Z_M = .30
    report = {"source_sha256": route.sha256(Path(__file__)),
              "joined_route_source_sha256": route.sha256(Path(joined.__file__)),
              "depth_standoff_turn_source_sha256": route.sha256(Path(turn.__file__)),
              "depth_change_source_sha256": route.sha256(Path(turn.changed_points.__code__.co_filename)),
              "person_source_sha256": route.sha256(Path(person.__file__)),
              "config_sha256": route.sha256(joined.CONFIG),
              "prior_map_sha256": route.sha256(joined.PRIOR),
              "taught_crop_sha256": route.sha256(owner.TEACH_CROP),
              "matrix": MATRIX,
              "scope": "same joined late-crossing RoboCasa route with stationary-nearer control and an owner who approaches 30cm in 1s during the first 40cm pre-show return, then holds for a translated articulated near-hand show; actor retains the original range-track center and camera aim, so the experiment challenges stale owner belief and prompt timing; kinematic/scripted development case",
              "episodes": episodes}
    (OUT / "report.json").write_text(json.dumps(report, indent=2) + "\n")


if __name__ == "__main__":
    main()
