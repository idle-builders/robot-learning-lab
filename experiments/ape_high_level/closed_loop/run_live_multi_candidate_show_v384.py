"""V384: live owner-help with simultaneous moving and raised identical cans.

The V373 guarded task still owns the turn. This scoped wrapper replaces its
single appearance winner with all-candidate 3D motion tracking, and stages a
raised replica at the owner show. The hand starts moving after 0.3 s, but
there is no explicit contingent 'wiggle' request or verified response yet.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

import run_live_moving_distant_help_v373 as base
from articulated_person_scene import START_HAND_XYZ_M, set_person_pose
from challenge_independent_hand_show_v374 import hand_offset
from challenge_raised_replica_show_v381 import BOX_XYZ_M, REPLICA_XYZ_M
from moving_candidate_tracks import (MovingCandidateTracks,
                                     appearance_candidates)
from run import ROOT, sha


SOURCE = Path(__file__)
OUT = ROOT / "artifacts/ape-closed-loop-cove50-v384"
_tracker: MovingCandidateTracks | None = None
_sample_index = -1


def staged_scene(room, start_s: float, elapsed_s: float,
                 carried_item: str | None) -> None:
    room.time_s = start_s + elapsed_s
    room._update_moving_obstacle()
    room.head = min(.8, 2. * max(0., elapsed_s))
    room.model.body_pos[room.model.body("loop_person").id] = (0., 0., 0.)
    hand = START_HAND_XYZ_M.copy()
    if elapsed_s >= .3:
        hand += hand_offset("hand_faster", elapsed_s - .3)
    set_person_pose(room.model, hand, carried_item=carried_item)
    room.model.body_pos[room.model.body("loop_box").id] = BOX_XYZ_M
    replica = ("loop_TomatoSauce" if carried_item == "loop_Imposter"
               else "loop_Imposter")
    room.model.body_pos[room.model.body(replica).id] = REPLICA_XYZ_M
    room._forward()


def frame_available(seed: int, mode: str, index: int) -> bool:
    global _tracker, _sample_index
    if index == 0:
        _tracker = MovingCandidateTracks()
    _sample_index = index
    return True


def multi_candidate_infer(vision, exemplar: np.ndarray,
                          owner_hist: np.ndarray, rgb, depth,
                          camera_pose: tuple, truth, true_target) -> dict:
    if _tracker is None:
        raise RuntimeError("Missing show tracker reset")
    candidates = appearance_candidates(vision, exemplar, owner_hist,
                                       rgb, depth, camera_pose)
    decision = _tracker.update(_sample_index, candidates)
    selected = decision["selected_candidate"]
    return {
        "proposals": len(candidates),
        "accepted_box": None if selected is None else selected["box"],
        "accepted_score": (None if selected is None else
                           selected["appearance_score"]),
        "accepted_hue_intersection": (
            None if selected is None else selected["hue_overlap"]),
        "accepted_world_xyz_m": (
            None if selected is None else selected["world_xyz_m"]),
        "accepted_correct_evaluator_only": False,
        "accepted_location_error_evaluator_only_m": None,
        "actor_all_candidates": candidates,
        "actor_candidate_count": decision["candidate_count"],
        "actor_moving_track_count": decision["moving_track_count"],
        "actor_selected_track_id": decision["selected_track_id"],
        "actor_selected_motion_span_m": decision[
            "selected_motion_span_m"],
    }


def main() -> None:
    old_scene, old_out, old_infer, old_available = (
        base.set_scene, base.OUT, base.infer, base.frame_available)
    try:
        base.set_scene = staged_scene
        base.OUT = OUT
        base.infer = multi_candidate_infer
        base.frame_available = frame_available
        base.main()
    finally:
        base.set_scene, base.OUT, base.infer, base.frame_available = (
            old_scene, old_out, old_infer, old_available)
    path = OUT / "blocked-owner-help-audit.json"
    report = json.loads(path.read_text())
    report["scope"] = (
        "V384 selected Cove-50 guarded blocked-route no-contact owner help "
        "with independently moving hand, simultaneously visible identical "
        "raised replica and all-appearance-candidate 3D motion tracks. "
        "The hand holds still for0.3s then follows a procedural0.4s wiggle; "
        "the owner was asked only to hold the item, so this is not a "
        "contingent response test. The actor selects only a unique track "
        "with >=8cm span over its last3 observations, then the existing "
        "V373 consecutive point/120ms freshness and scripted owner yes/no "
        "gate apply. The replica/box enter at owner-help onset; this is a "
        "task fallback perception challenge, not a validated room during "
        "navigation. Camera/head/hand dynamics, audio and exact identity "
        "remain procedural. The inherited video panel says V373.")
    report["v384_source_sha256"] = sha(SOURCE)
    report["multi_candidate_tracker_source_sha256"] = sha(Path(
        appearance_candidates.__code__.co_filename))
    path.write_text(json.dumps(report, indent=2) + "\n")
    (OUT / SOURCE.name).write_bytes(SOURCE.read_bytes())


if __name__ == "__main__":
    main()
