"""V375: live blocked-route help while the owner's hand moves on a still torso.

The V373 decision loop is reused unchanged. This scoped adapter changes only
the evaluator's person motion, output directory, and final report metadata.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

import run_live_moving_distant_help_v373 as base
from articulated_person_scene import START_HAND_XYZ_M, set_person_pose
from challenge_independent_hand_show_v374 import hand_offset
from run import ROOT, sha


SOURCE = Path(__file__)
OUT = ROOT / "artifacts/ape-closed-loop-cove50-v375"


def independent_hand_scene(room, start_s: float, elapsed_s: float,
                           carried_item: str | None) -> None:
    room.time_s = start_s + elapsed_s
    room._update_moving_obstacle()
    room.head = min(.8, 2. * max(0., elapsed_s))
    room.model.body_pos[room.model.body("loop_person").id] = (0., 0., 0.)
    set_person_pose(room.model, START_HAND_XYZ_M + hand_offset(
        "hand_faster", elapsed_s), carried_item=carried_item)
    room._forward()


def main() -> None:
    old_scene, old_out = base.set_scene, base.OUT
    try:
        base.set_scene = independent_hand_scene
        base.OUT = OUT
        base.main()
    finally:
        base.set_scene, base.OUT = old_scene, old_out
    path = OUT / "blocked-owner-help-audit.json"
    report = json.loads(path.read_text())
    report["scope"] = (
        "V375 selected Cove-50 live guarded blocked-route no-contact owner "
        "help: the owner torso stays fixed and the IK hand independently "
        "moves a taught can, identical lookalike or no item on a 0.4s lateral "
        "cycle. The reused V373 loop has 10Hz 20ms-exposed lower RGB, noisy "
        "long depth, released taught-object vision, robust finite-depth 3D "
        "location, <=12cm adjacent association and scripted owner yes/no. "
        "The 20ms person/scene gap is evaluator-only. Hand dynamics, real "
        "sensor optics, audio and native head/stop remain absent; this does "
        "not authorize any close reach, base move or physical handoff. "
        "V373 remains the unchanged decision/render harness and its panel "
        "text is inherited.")
    report["v375_source_sha256"] = sha(SOURCE)
    report["v374_motion_source_sha256"] = sha(Path(
        hand_offset.__code__.co_filename))
    path.write_text(json.dumps(report, indent=2) + "\n")
    (OUT / SOURCE.name).write_bytes(SOURCE.read_bytes())


if __name__ == "__main__":
    main()
