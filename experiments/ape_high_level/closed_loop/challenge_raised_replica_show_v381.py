"""V381: a raised, supported identical can beside the owner's moving can.

This tests whether the V380 'held above the floor' cue confuses an object
on a short podium with one in the owner's hand. The podium is a relocated
existing box and the replica rests at its top; all choices remain actor RGB-D.
"""

from __future__ import annotations

import json
from pathlib import Path

import challenge_independent_hand_show_v374 as base
from run import ROOT, sha


SOURCE = Path(__file__)
OUT = ROOT / "artifacts/ape-closed-loop-cove50-v381"
BOX_XYZ_M = (.65, .30, .14)
REPLICA_XYZ_M = (.65, .30, .332)
_original_hand_scene = base.hand_scene


def raised_replica_scene(room, pattern: str, elapsed_s: float) -> None:
    _original_hand_scene(room, pattern, elapsed_s)
    room.model.body_pos[room.model.body("loop_box").id] = BOX_XYZ_M
    room.model.body_pos[room.model.body("loop_Imposter").id] = REPLICA_XYZ_M
    room._forward()


def main() -> None:
    old_scene, old_out, old_patterns = (
        base.hand_scene, base.OUT, base.PATTERNS)
    try:
        base.hand_scene = raised_replica_scene
        base.OUT = OUT
        base.PATTERNS = ("hand_faster",)
        base.main()
    finally:
        base.hand_scene, base.OUT, base.PATTERNS = (
            old_scene, old_out, old_patterns)
    path = OUT / "independent-hand-show.json"
    report = json.loads(path.read_text())
    report["scope"] = (
        "V381 selected Cove-50 no-contact owner show with V374 faster "
        "independent hand-held taught can and a visually identical replica "
        "resting on the relocated room box. Box center (0.65,0.30,0.14)m "
        "with top z0.28m; replica center z0.332m. Same exposed RGB, noisy "
        "long depth, released appearance choice and robust point trace. "
        "Evaluator distinguishes held target from supported replica after "
        "actor choice. This procedural two-seed screen challenges the "
        "revealed V380 floor/held z threshold, not a physical hand skill.")
    report["v381_source_sha256"] = sha(SOURCE)
    report["support_box_xyz_m"] = BOX_XYZ_M
    report["supported_replica_xyz_m"] = REPLICA_XYZ_M
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
