"""Reacquire a lost owner range track from fresh person RGB-D."""

from __future__ import annotations

import json
import math
from pathlib import Path

import numpy as np
from PIL import Image

import robocasa_first_return_person_reacquire_turn_v565 as turn
import robocasa_owner_person as person
import robocasa_person_interaction as interaction
import run_robocasa_joined_owner_route_v410 as owner
import run_robocasa_layout2_route_v407 as route
import run_robocasa_person_joined_v497 as joined
import run_robocasa_layout2_multitrack_v485 as multi
from run_robocasa_dynamic_suite_v439 import motion
from run_robocasa_scan_height_v433 import height_scan


ROOT = Path(__file__).resolve().parents[3]
OUT = ROOT / "artifacts/ape-robocasa-first-return-person-reacquire-v566"
MATRIX = (("early_owner_mid_ideal", False),
          ("early_owner_mid_sector_blackout", True),
          ("stationary_owner_mid_sector_blackout", True))


def owner_scan(blackout):
    raw = height_scan((.30,))
    calls = 0
    center = None

    def scan(model, data, xy, max_range_m, rng):
        nonlocal calls, center
        calls += 1
        packet = raw(model, data, xy, max_range_m, rng)
        if not blackout or calls > 5:
            return packet
        if center is None:
            tracks = multi.CAPTURED[-1].multi_history[-1]["tracks"]
            center = np.asarray(max(tracks, key=lambda row: row["hits"])["center_xy_m"])
        direction = center - np.asarray(xy)
        bearing = math.atan2(direction[1], direction[0])
        angles = np.linspace(-math.pi, math.pi, len(packet), endpoint=False)
        delta = (angles - bearing + math.pi) % (2 * math.pi) - math.pi
        return [None if abs(delta[index]) <= math.radians(25.) else value
                for index, value in enumerate(packet)]

    return scan


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
        for response, blackout in MATRIX:
            interaction.SHOW_HAND = np.asarray([1.30, -2.98, .55])
            turn.SCAN = owner_scan(blackout)
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
              "person_reacquire_source_sha256": route.sha256(
                  Path(turn.person_reacquire.__file__)),
              "person_detector_weights_sha256": route.sha256(
                  turn.person_reacquire.DETECTOR / "model.safetensors"),
              "depth_change_source_sha256": route.sha256(Path(turn.changed_points.__code__.co_filename)),
              "person_source_sha256": route.sha256(Path(person.__file__)),
              "config_sha256": route.sha256(joined.CONFIG),
              "prior_map_sha256": route.sha256(joined.PRIOR),
              "taught_crop_sha256": route.sha256(owner.TEACH_CROP),
              "matrix": MATRIX,
              "scope": "same joined late-crossing RoboCasa route and ideal/moving/stationary first-five-owner-sector blackout matrix as V551; the two person RGB-D captures use 20 and 15 degree camera pitches, require 3D depth-point agreement, and need score .35 when relocated or .25 only within prior owner-surface uncertainty; kinematic/scripted development case",
              "episodes": episodes}
    (OUT / "report.json").write_text(json.dumps(report, indent=2) + "\n")


if __name__ == "__main__":
    main()
