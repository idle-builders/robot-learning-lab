"""Compare owner-range scan heights in the complete early-motion episode."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
from PIL import Image

import robocasa_first_return_active_view_turn_v544 as turn
import robocasa_owner_person as person
import robocasa_person_interaction as interaction
import run_robocasa_joined_owner_route_v410 as owner
import run_robocasa_layout2_route_v407 as route
import run_robocasa_person_joined_v497 as joined
from run_robocasa_dynamic_suite_v439 import motion
from run_robocasa_scan_height_v433 import height_scan


ROOT = Path(__file__).resolve().parents[3]
OUT = ROOT / "artifacts/ape-robocasa-first-return-height-v548"
MATRIX = (("early_owner_low_5cm", (.05,)),
          ("early_owner_mid_30cm", (.30,)),
          ("early_owner_fused_5_30cm", (.05, .30)))


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
    original_scan = turn.SCAN
    episodes = []
    try:
        for response, heights in MATRIX:
            interaction.SHOW_HAND = np.asarray([1.30, -2.98, .55])
            turn.SCAN = height_scan(heights)
            turn.OWNER_APPROACH_M = 0.
            turn.FIRST_RETURN_OWNER_M = .4
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
        turn.SCAN = original_scan
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
              "scope": "same joined late-crossing RoboCasa route, same 40cm/1s first-return owner approach and near-hand show; owner turn alone alternates 5cm, 30cm and fused 5/30cm ideal 72-beam scan streams while route/map and RGB-D stay fixed; actor can re-stage and select healthy pre-show camera pitch only from its available packets; kinematic/scripted development case",
              "episodes": episodes}
    (OUT / "report.json").write_text(json.dumps(report, indent=2) + "\n")


if __name__ == "__main__":
    main()
