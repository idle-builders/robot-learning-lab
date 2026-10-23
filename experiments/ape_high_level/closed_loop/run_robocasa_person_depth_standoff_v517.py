"""Same-scene robust presentation depth, standoff and owner watch/find."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
from PIL import Image

import robocasa_depth_standoff_turn_v517 as turn
import robocasa_owner_person as person
import run_robocasa_joined_owner_route_v410 as owner
import run_robocasa_layout2_route_v407 as route
import run_robocasa_person_joined_v497 as joined
from run_robocasa_dynamic_suite_v439 import motion


ROOT = Path(__file__).resolve().parents[3]
OUT = ROOT / "artifacts/ape-robocasa-person-depth-standoff-v517"
MATRIX = (("stationary_ideal", True, 0., 0., 0.),
          ("approach_60cm_ideal", True, .6, 0., 0.),
          ("approach_60cm_noise_1cm", True, .6, .01, 0.),
          ("approach_60cm_dropout_30pct", True, .6, 0., .3),
          ("approach_70cm_ideal", True, .7, 0., 0.),
          ("approach_60cm_rgb", False, .6, 0., 0.))


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
    episodes = []
    for response, use_depth, distance, noise, dropout in MATRIX:
        turn.OWNER_APPROACH_M = distance
        turn.DEPTH_NOISE_SIGMA_M = noise
        turn.DEPTH_DROPOUT_FRACTION = dropout
        turn.PROP_SCRIPT = motion(case)
        turn.PACKET_OUT = OUT / f"{response}-depth-packets.npz" if use_depth else None
        result = joined.episode(case, response, use_depth,
                                prior, policy, vision, exemplar)
        episodes.append(result)
        chosen = next((row["chosen"] for row in result["trace"]
                       if row["stage"] == "choose_owner_standoff"), None)
        actions = [(row["time_s"], row["actor_action"]) for row in result["trace"]
                   if row.get("actor_action")]
        print(response, result["outcome"], "standoff",
              chosen["distance_m"] if chosen else None,
              "depth_error", result["depth_point_error_m"],
              "actions", actions, flush=True)
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
              "owner_approach_duration_s": turn.OWNER_APPROACH_S,
              "scope": "same seeded articulated-owner RoboCasa route; actor uses health-gated connected before/after depth and 1pct lower-tail surface at range stop, plans known-free farther pose against a 60cm assumed surface-motion bound, with 1cm Gaussian noise and 30pct independent after-depth dropout cases; kinematic unqualified return, ideal 10Hz complete range watch during slow RGB recognition, fresh RGB-D confirmation; RGB-only ablation abstains; no native gait or human-motion guarantee",
              "episodes": episodes}
    (OUT / "report.json").write_text(json.dumps(report, indent=2) + "\n")


if __name__ == "__main__":
    main()
