"""Challenge the fixed planned standoff with longer scripted owner approaches."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
from PIL import Image

import robocasa_owner_person as person
import robocasa_standoff_turn_v507 as turn
import run_robocasa_joined_owner_route_v410 as owner
import run_robocasa_layout2_route_v407 as route
import run_robocasa_person_joined_v497 as joined
from run_robocasa_dynamic_suite_v439 import motion


ROOT = Path(__file__).resolve().parents[3]
OUT = ROOT / "artifacts/ape-robocasa-person-standoff-bound-v509"
MATRIX = ((.25, .6), (.25, .7), (.35, .6), (.35, .7))


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
    for reserve, distance in MATRIX:
        turn.MIN_RESERVE_M = reserve
        turn.OWNER_APPROACH_M = distance
        turn.PROP_SCRIPT = motion(case)
        response = (f"approach_{round(distance * 100):.0f}cm_"
                    f"reserve_{round(reserve * 100):.0f}cm")
        result = joined.episode(case, response, True, prior, policy, vision, exemplar)
        episodes.append(result)
        actions = [(row["time_s"], row["actor_action"]) for row in result["trace"]
                   if row.get("actor_action")]
        print(response, result["outcome"], "actions", actions,
              "depth_error", result["depth_point_error_m"], flush=True)
    route.MARGIN_M = .10
    route.LIDAR_Z_M = .30
    report = {"source_sha256": route.sha256(Path(__file__)),
              "joined_route_source_sha256": route.sha256(Path(joined.__file__)),
              "standoff_turn_source_sha256": route.sha256(Path(turn.__file__)),
              "person_source_sha256": route.sha256(Path(person.__file__)),
              "config_sha256": route.sha256(joined.CONFIG),
              "prior_map_sha256": route.sha256(joined.PRIOR),
              "taught_crop_sha256": route.sha256(owner.TEACH_CROP),
              "owner_approach_duration_s": turn.OWNER_APPROACH_S,
              "matrix_reserve_approach_m": MATRIX,
              "scope": "four revealed development challenge combinations, one scripted southward articulated owner, 30/40cm planned standoff, ideal 10Hz complete range and kinematic unqualified return; 60cm equals policy assumed max displacement, 70cm exceeds it; same seeded late crossing, not a held-out human-motion distribution",
              "episodes": episodes}
    (OUT / "report.json").write_text(json.dumps(report, indent=2) + "\n")


if __name__ == "__main__":
    main()
