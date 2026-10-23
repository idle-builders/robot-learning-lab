"""Live same-scene route, planned owner standoff, motion watch, and fresh find."""

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
OUT = ROOT / "artifacts/ape-robocasa-person-standoff-v507"
MATRIX = (("late_crossing", "stationary_standoff"),
          ("late_crossing", "approach_after_standoff"))


def main():
    policy = json.loads(joined.CONFIG.read_text())
    cases = {case["id"]: case for case in policy["cases"]}
    prior = np.load(joined.PRIOR)
    OUT.mkdir(parents=True, exist_ok=True)
    joined.OUT = OUT
    joined.owner_turn = turn.owner_turn
    route.MARGIN_M = .15
    vision = owner.Vision(teach_selector="generic-proposal")
    exemplar = vision.feature([Image.open(owner.TEACH_CROP).convert("RGB")])[0]
    episodes = []
    for case_id, response in MATRIX:
        turn.PROP_SCRIPT = motion(cases[case_id])
        result = joined.episode(cases[case_id], response, True,
                                prior, policy, vision, exemplar)
        episodes.append(result)
        actions = [row["actor_action"] for row in result["trace"]
                   if row.get("actor_action")]
        print(case_id, response, result["route_state"], result["outcome"],
              "actions", actions, "depth_error", result["depth_point_error_m"],
              flush=True)
    route.MARGIN_M = .10
    route.LIDAR_Z_M = .30
    report = {"source_sha256": route.sha256(Path(__file__)),
              "joined_route_source_sha256": route.sha256(Path(joined.__file__)),
              "standoff_turn_source_sha256": route.sha256(Path(turn.__file__)),
              "person_source_sha256": route.sha256(Path(person.__file__)),
              "config_sha256": route.sha256(joined.CONFIG),
              "prior_map_sha256": route.sha256(joined.PRIOR),
              "taught_crop_sha256": route.sha256(owner.TEACH_CROP),
              "scope": "one seeded owner-present MuJoCo scene, localized range route, actor prior/map-backed 30cm standoff choice, kinematic unqualified return, 10Hz ideal range motion watch during slow visual recognition, settle/re-capture/fresh RGB-D find; scripted owner and prop, no native gait or threaded timing guarantee",
              "matrix": MATRIX, "episodes": episodes}
    (OUT / "report.json").write_text(json.dumps(report, indent=2) + "\n")


if __name__ == "__main__":
    main()
