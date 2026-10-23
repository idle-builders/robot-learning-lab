"""V379: an identical floor can beside a hand-held taught can.

This is a first multi-candidate stress screen for the no-contact owner show.
The owner torso stays fixed and moves one hand independently; a physically
supported replica remains on the floor nearby. The unchanged detector may
select either. Identity scoring is evaluator-only.
"""

from __future__ import annotations

import json
from pathlib import Path

import challenge_independent_hand_show_v374 as base
from run import ROOT, sha


SOURCE = Path(__file__)
OUT = ROOT / "artifacts/ape-closed-loop-cove50-v379"
FLOOR_REPLICA_XYZ_M = (.55, .47, .052)


def dual_can_scene(room, pattern: str, elapsed_s: float) -> None:
    base_scene = _original_hand_scene
    base_scene(room, pattern, elapsed_s)
    room.model.body_pos[room.model.body("loop_Imposter").id] = (
        FLOOR_REPLICA_XYZ_M)
    room._forward()


_original_hand_scene = base.hand_scene


def main() -> None:
    old_scene, old_out, old_patterns = (
        base.hand_scene, base.OUT, base.PATTERNS)
    try:
        base.hand_scene = dual_can_scene
        base.OUT = OUT
        base.PATTERNS = ("hand_faster",)
        base.main()
    finally:
        base.hand_scene, base.OUT, base.PATTERNS = (
            old_scene, old_out, old_patterns)
    path = OUT / "independent-hand-show.json"
    report = json.loads(path.read_text())
    report["scope"] = (
        "V379 selected Cove-50 no-contact owner show with V374 faster "
        "independent hand-held taught can and a visually identical replica "
        "supported on the floor at (0.55,0.47,0.052)m. Both may enter the "
        "same moving lower-head RGB-D view. The released appearance gate "
        "chooses one box, then robust depth and 3D continuity are scored. "
        "Evaluator truth identifies the actual held can only after the "
        "actor choice; no owner response or multi-candidate track is added. "
        "The person, hand, optics and head remain procedural. The inherited "
        "V370 video panel names the old screen.")
    report["v379_source_sha256"] = sha(SOURCE)
    report["floor_replica_xyz_m"] = FLOOR_REPLICA_XYZ_M
    for row in report["rows"]:
        row["evaluator_wrong_accepted_frames"] = sum(
            sample["actor_choice"]["accepted_box"] is not None and not
            sample["evaluator_accepted_presented_can"]
            for sample in row["samples"])
        print(row["case"], "held_correct", row[
            "evaluator_correct_frames"], "wrong", row[
                "evaluator_wrong_accepted_frames"], flush=True)
    path.write_text(json.dumps(report, indent=2) + "\n")
    (OUT / SOURCE.name).write_bytes(SOURCE.read_bytes())


if __name__ == "__main__":
    main()
