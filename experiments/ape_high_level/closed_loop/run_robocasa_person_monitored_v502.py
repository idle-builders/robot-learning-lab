"""Same-scene owner task with range interrupts during slow recognition.

Reuse the V497 seeded route and its owner-present scene. Only the owner turn
changes: range packets can interrupt recognition, request a pause, and if
movement persists request a backtrack skill. A kinematic retreat surrogate
is executed for geometry screening; native skill dynamics remain unproven.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
from PIL import Image

import robocasa_owner_person as person
import robocasa_person_monitored_turn_v502 as turn
import run_robocasa_joined_owner_route_v410 as owner
import run_robocasa_layout2_route_v407 as route
import run_robocasa_person_joined_v497 as joined
from run_robocasa_dynamic_suite_v439 import motion


ROOT = Path(__file__).resolve().parents[3]
OUT = ROOT / "artifacts/ape-robocasa-person-monitored-v502"
MATRIX = (
    ("stationary", "stationary_lowmid"),
    ("late_crossing", "approach_off"),
    ("late_crossing", "approach_pause_lowmid"),
    ("late_crossing", "approach_ignore_lowmid"),
    ("late_crossing", "approach_pause_upper"),
    ("late_crossing", "approach_ignore_upper"),
)


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
        episode = joined.episode(cases[case_id], response, True,
                                 prior, policy, vision, exemplar)
        episodes.append(episode)
        actions = [row["actor_action"] for row in episode["trace"]
                   if row.get("actor_action")]
        print(case_id, response, episode["outcome"],
              [round(x, 3) for x in episode["final_actual_xy_m"]],
              actions, flush=True)
    route.MARGIN_M = .10
    route.LIDAR_Z_M = .30
    report = {"source_sha256": route.sha256(Path(__file__)),
              "joined_route_source_sha256": route.sha256(Path(joined.__file__)),
              "monitored_turn_source_sha256": route.sha256(Path(turn.__file__)),
              "person_source_sha256": route.sha256(Path(person.__file__)),
              "config_sha256": route.sha256(joined.CONFIG),
              "prior_map_sha256": route.sha256(joined.PRIOR),
              "taught_crop_sha256": route.sha256(owner.TEACH_CROP),
              "scope": "same-seeded furnished route and owner task with discrete-event range monitoring during slow recognition; scripted owner pause/ignored motion, hypothetical 55cm range ablation and kinematic 25cm/s backtrack surrogate; no native gait or real-time threaded guarantee",
              "matrix": MATRIX, "episodes": episodes}
    (OUT / "report.json").write_text(json.dumps(report, indent=2) + "\n")


if __name__ == "__main__":
    main()
