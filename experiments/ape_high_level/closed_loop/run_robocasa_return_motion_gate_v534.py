"""Evaluate online owner-range motion gating during the second return."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
from PIL import Image

import robocasa_return_motion_gate_turn_v533 as turn
import robocasa_owner_person as person
import robocasa_person_interaction as interaction
import run_robocasa_joined_owner_route_v410 as owner
import run_robocasa_layout2_route_v407 as route
import run_robocasa_person_joined_v497 as joined
from run_robocasa_dynamic_suite_v439 import motion


ROOT = Path(__file__).resolve().parents[3]
OUT = ROOT / "artifacts/ape-robocasa-return-motion-gate-v534"
MATRIX = (("stationary_nearer_show", [1.30, -2.98, .55]),
          ("nearer_show_fast_motion_gated", [1.30, -2.98, .55]),
          ("reference_show_fast_motion_gated", [1.30, -2.80, .55]))


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
            turn.SECOND_RETURN_OWNER_M = .6
            turn.SECOND_RETURN_OWNER_S = 1.
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
              "scope": "same joined late-crossing RoboCasa route with stationary-nearer and 60cm-in-1s nearer/reference articulated owner hand-show poses; 35cm pre-show range reserve; online 10Hz ego-compensated return-range track flags owner motion, invalidates pre-return pose and requests pause; owner script ignores request; return/recognition remain kinematic and bound-specific",
              "episodes": episodes}
    (OUT / "report.json").write_text(json.dumps(report, indent=2) + "\n")


if __name__ == "__main__":
    main()
