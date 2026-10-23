"""V390: gap-tolerant owner-item belief while the owner pauses a fast show.

Right/bright shifted-person scenes from V389 are rerun with a scripted hand
hold after 0.8 s. A two-frame-gap motion track latches item identity once
motion was seen, then still demands current observation at the answer. The
owner pause is procedural, not yet a response to an actual second request.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from articulated_person_scene import (ITEM_SIDE_OFFSET_M,
                                      START_HAND_XYZ_M, add_person,
                                      set_person_pose)
from challenge_independent_hand_show_v374 import hand_offset
from challenge_person_belief_v346 import POSITIVE_VARIANTS
from challenge_raised_replica_show_v381 import REPLICA_XYZ_M
from gap_latched_candidate_tracks import GapLatchedCandidateTracks
import run_blocked_owner_help_v317 as owner_module
import run_live_moving_distant_help_v373 as base
import run_live_multi_candidate_show_v384 as live
from run import ROOT, sha


SOURCE = Path(__file__)
OUT = ROOT / "artifacts/ape-closed-loop-cove50-v390"
VARIANT = next(row for row in POSITIVE_VARIANTS
               if row["name"] == "right_bright")
FREEZE_AT_S = .8
FROZEN_HAND = START_HAND_XYZ_M + hand_offset(
    "hand_faster", FREEZE_AT_S - .3)


def expected_item(elapsed_s: float) -> np.ndarray:
    if elapsed_s < .3:
        hand = START_HAND_XYZ_M
    elif elapsed_s <= FREEZE_AT_S:
        hand = START_HAND_XYZ_M + hand_offset(
            "hand_faster", elapsed_s - .3)
    else:
        hand = FROZEN_HAND
    return hand + ITEM_SIDE_OFFSET_M + np.asarray(
        VARIANT["offset_xyz_m"])


def score(report: dict) -> list[dict]:
    rows = []
    for entry in report["owner_show_rows"]:
        response = entry["response"]
        samples = response["actor_samples"]
        start = float(samples[0]["capture_time_s"])
        classes = []
        for sample in samples:
            point = sample["actor_choice"]["accepted_world_xyz_m"]
            if point is None:
                classes.append("none")
                continue
            position = np.asarray(point)
            held = expected_item(float(sample["capture_time_s"]) - start)
            classes.append("moving_then_held_item" if np.linalg.norm(
                position - held) < np.linalg.norm(
                    position - REPLICA_XYZ_M)
                else "static_raised_replica")
        rows.append({
            "seed": entry["seed"], "status": response["status"],
            "actor_track_confirmed": response["actor_track_confirmed"],
            "actor_track_fresh_at_answer": response[
                "actor_track_fresh_at_answer"],
            "evaluator_selected_owner_item_frames": classes.count(
                "moving_then_held_item"),
            "evaluator_selected_replica_frames": classes.count(
                "static_raised_replica"),
            "evaluator_final_selected_class": classes[-1],
            "evaluator_min_person_gap_m": response[
                "evaluator_min_person_gap_m"],
            "evaluator_min_scene_gap_m": response[
                "evaluator_min_scene_gap_m"],
        })
    return rows


def main() -> None:
    offset = np.asarray(VARIANT["offset_xyz_m"])
    original_scene = live.staged_scene
    original_out = live.OUT
    original_tracker = live.MovingCandidateTracks
    original_add_person = base.add_person
    original_cases = owner_module.CASES

    def paused_scene(room, start_s: float, elapsed_s: float,
                     carried_item: str | None) -> None:
        original_scene(room, start_s, elapsed_s, carried_item)
        if elapsed_s > FREEZE_AT_S:
            set_person_pose(room.model, FROZEN_HAND,
                            carried_item=carried_item)
        room.model.body_pos[room.model.body("loop_person").id] = offset
        if carried_item is not None:
            room.model.body_pos[room.model.body(carried_item).id] += offset
        if not hasattr(room, "_v390_base_light"):
            room._v390_base_light = room.model.light_diffuse.copy()
        room.model.light_diffuse[:] = (room._v390_base_light *
                                       VARIANT["light_scale"])
        room._forward()

    try:
        live.staged_scene = paused_scene
        live.OUT = OUT
        live.MovingCandidateTracks = GapLatchedCandidateTracks
        base.add_person = lambda model: add_person(
            model, shirt_rgba=VARIANT["shirt_rgba"])
        owner_module.CASES = tuple(case for case in original_cases
                                  if case[1] == "present")
        live.main()
    finally:
        live.staged_scene = original_scene
        live.OUT = original_out
        live.MovingCandidateTracks = original_tracker
        base.add_person = original_add_person
        owner_module.CASES = original_cases
    path = OUT / "blocked-owner-help-audit.json"
    report = json.loads(path.read_text())
    report["scope"] = (
        "V390 selected Cove-50 guarded blocked-route no-contact show in "
        "V346 right/bright shifted-owner scenes, seeds170/172. The owner "
        "hand wiggles procedurally from0.3–0.8s then holds at the final "
        "visible pose. Frozen V384 all-candidate DINO/hue/noisy-long-depth "
        "observations feed a gap<=2-frame, <=12cm-per-frame 3D tracker "
        "that latches prior motion evidence but selects only a currently "
        "seen unique candidate. V373 still requires two consecutive final "
        "points, <=120ms age and scripted owner yes. The owner hold is not "
        "triggered by an explicit second request, and pose/head health, "
        "audio, optics and native motion remain procedural.")
    report["v390_source_sha256"] = sha(SOURCE)
    report["gap_tracker_source_sha256"] = sha(Path(
        GapLatchedCandidateTracks.__init__.__code__.co_filename))
    report["freeze_at_elapsed_s"] = FREEZE_AT_S
    report["evaluator_identity_rows"] = score(report)
    path.write_text(json.dumps(report, indent=2) + "\n")
    (OUT / SOURCE.name).write_bytes(SOURCE.read_bytes())
    for row in report["evaluator_identity_rows"]:
        print(row["seed"], row["status"], "held", row[
            "evaluator_selected_owner_item_frames"], "replica", row[
                "evaluator_selected_replica_frames"], "final", row[
                    "evaluator_final_selected_class"], flush=True)


if __name__ == "__main__":
    main()
