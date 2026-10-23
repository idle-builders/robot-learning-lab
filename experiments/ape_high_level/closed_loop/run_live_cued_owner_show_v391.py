"""V391: actor issues wiggle/hold requests and verifies a scripted response.

The guarded Cove-50 owner-help loop is retained. The owner can comply,
delay, or ignore the *second* hold request; only visual stillness of the
previously moving candidate, then two fresh accepted frames, may promote a
present-can answer. Spoken audio and physical human compliance are absent.
"""

from __future__ import annotations

import json
import math
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
OUT = ROOT / "artifacts/ape-closed-loop-cove50-v391"
VARIANT = next(row for row in POSITIVE_VARIANTS
               if row["name"] == "right_bright")
RESPONSE_MODES = ("compliant", "delayed", "ignored")
HOLD_DELAY_S = .3
STILL_STEP_M = .025


def natural_hand(elapsed_s: float) -> np.ndarray:
    if elapsed_s < .3:
        return START_HAND_XYZ_M.copy()
    return START_HAND_XYZ_M + hand_offset(
        "hand_faster", elapsed_s - .3)


def run_mode(response_mode: str) -> tuple[dict, list[dict]]:
    offset = np.asarray(VARIANT["offset_xyz_m"])
    state: dict = {}
    original_scene = live.staged_scene
    original_frame_available = live.frame_available
    original_infer = live.multi_candidate_infer
    original_out = live.OUT
    original_tracker = live.MovingCandidateTracks
    original_add_person = base.add_person
    original_cases = owner_module.CASES

    def reset() -> None:
        state.clear()
        state.update({"last_elapsed": -1., "wiggle_at": None,
                      "hold_at": None, "hold_index": None,
                      "frozen_hand": None, "last_selected": None,
                      "last_selected_index": None, "still_run": 0,
                      "hold_observed": False})

    reset()

    def cued_scene(room, start_s: float, elapsed_s: float,
                   carried_item: str | None) -> None:
        if elapsed_s <= 1e-9 and state["last_elapsed"] > .5:
            reset()
        original_scene(room, start_s, elapsed_s, carried_item)
        hand = START_HAND_XYZ_M.copy()
        if state["wiggle_at"] is not None and elapsed_s >= .3:
            hand = natural_hand(elapsed_s)
        hold_at = state["hold_at"]
        if hold_at is not None and response_mode != "ignored":
            response_at = (hold_at if response_mode == "compliant" else
                           hold_at + HOLD_DELAY_S)
            if elapsed_s > response_at:
                if state["frozen_hand"] is None:
                    state["frozen_hand"] = natural_hand(response_at)
                hand = state["frozen_hand"]
        set_person_pose(room.model, hand, carried_item=carried_item)
        room.model.body_pos[room.model.body("loop_person").id] = offset
        if carried_item is not None:
            room.model.body_pos[room.model.body(carried_item).id] += offset
        if not hasattr(room, "_v391_base_light"):
            room._v391_base_light = room.model.light_diffuse.copy()
        room.model.light_diffuse[:] = (room._v391_base_light *
                                       VARIANT["light_scale"])
        room._forward()
        state["last_elapsed"] = elapsed_s

    def frame_available(seed: int, mode: str, index: int) -> bool:
        return original_frame_available(seed, mode, index)

    def social_infer(vision, exemplar, owner_hist, rgb, depth,
                     camera_pose, truth, true_target) -> dict:
        choice = original_infer(vision, exemplar, owner_hist, rgb, depth,
                                camera_pose, truth, true_target)
        index = live._sample_index
        events = []
        if index == 0 and state["wiggle_at"] is None:
            state["wiggle_at"] = 0.
            events.append({"action": "please_wiggle_this_one",
                           "elapsed_s": 0.})
        selected = choice["accepted_world_xyz_m"]
        if (index >= 8 and state["hold_at"] is None and
                selected is not None):
            state["hold_at"] = index * .1
            state["hold_index"] = index
            state["frozen_hand"] = None
            state["last_selected"] = selected
            state["last_selected_index"] = index
            events.append({"action": "please_hold_that_one_still",
                           "elapsed_s": round(index * .1, 3),
                           "actor_track_id": choice[
                               "actor_selected_track_id"]})
        elif state["hold_at"] is not None and index > state[
                "hold_index"]:
            if selected is None:
                state["still_run"] = 0
                state["last_selected"] = None
                state["last_selected_index"] = None
            else:
                linked_still = (
                    state["last_selected"] is not None and
                    state["last_selected_index"] == index - 1 and
                    math.dist(selected, state["last_selected"]) <=
                    STILL_STEP_M)
                state["still_run"] = (state["still_run"] + 1
                                      if linked_still else 0)
                state["last_selected"] = selected
                state["last_selected_index"] = index
                if state["still_run"] >= 2:
                    state["hold_observed"] = True
        choice["actor_social_events"] = events
        choice["actor_hold_observed"] = state["hold_observed"]
        choice["actor_still_link_run"] = state["still_run"]
        if not state["hold_observed"]:
            choice["accepted_box"] = None
            choice["accepted_world_xyz_m"] = None
            choice["accepted_score"] = None
            choice["accepted_hue_intersection"] = None
        return choice

    try:
        live.staged_scene = cued_scene
        live.frame_available = frame_available
        live.multi_candidate_infer = social_infer
        live.OUT = OUT / response_mode
        live.MovingCandidateTracks = GapLatchedCandidateTracks
        base.add_person = lambda model: add_person(
            model, shirt_rgba=VARIANT["shirt_rgba"])
        owner_module.CASES = tuple(case for case in original_cases
                                  if case[1] == "present")
        live.main()
    finally:
        live.staged_scene = original_scene
        live.frame_available = original_frame_available
        live.multi_candidate_infer = original_infer
        live.OUT = original_out
        live.MovingCandidateTracks = original_tracker
        base.add_person = original_add_person
        owner_module.CASES = original_cases
    path = OUT / response_mode / "blocked-owner-help-audit.json"
    report = json.loads(path.read_text())
    rows = []
    for entry in report["owner_show_rows"]:
        response = entry["response"]
        samples = response["actor_samples"]
        events = [event for sample in samples for event in
                  sample["actor_choice"].get("actor_social_events", [])]
        rows.append({
            "response_mode_evaluator_only": response_mode,
            "seed": entry["seed"], "status": response["status"],
            "actor_social_events": events,
            "actor_hold_observed_at_answer": samples[-1][
                "actor_choice"]["actor_hold_observed"],
            "actor_track_fresh_at_answer": response[
                "actor_track_fresh_at_answer"],
            "evaluator_min_person_gap_m": response[
                "evaluator_min_person_gap_m"],
            "evaluator_min_scene_gap_m": response[
                "evaluator_min_scene_gap_m"],
        })
    report["scope"] = (
        "V391 selected Cove-50 guarded no-contact owner show, right/bright "
        "shifted person and raised identical replica. Actor emits a wiggle "
        "event after its first image, then a hold-still event only after "
        "unique moving-item evidence from a gap-tolerant 3D track. The "
        "scripted owner response mode is " + response_mode + ". The actor "
        "does not see that mode; it requires two adjacent <=2.5cm item "
        "steps after the hold request before passing a candidate to V373's "
        "two-frame/120ms fresh owner-answer gate. This tests high-level "
        "request/observation order, not actual audio or human compliance; "
        "head, depth and hand dynamics remain procedural.")
    report["v391_source_sha256"] = sha(SOURCE)
    report["gap_tracker_source_sha256"] = sha(Path(
        GapLatchedCandidateTracks.__init__.__code__.co_filename))
    path.write_text(json.dumps(report, indent=2) + "\n")
    return report, rows


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    rows = []
    reports = []
    for mode in RESPONSE_MODES:
        report, scored = run_mode(mode)
        rows.extend(scored)
        path = OUT / mode / "blocked-owner-help-audit.json"
        reports.append({"mode": mode,
                        "report_path": str(path.relative_to(ROOT)),
                        "report_sha256": sha(path)})
        for row in scored:
            print(mode, row["seed"], row["status"],
                  "hold", row["actor_hold_observed_at_answer"],
                  "fresh", row["actor_track_fresh_at_answer"],
                  flush=True)
    aggregate = {
        "scope": "Three scripted owner reactions (compliant, delayed0.3s, ignored) to actual actor-emitted wiggle/hold-still events, crossed with Cove-50 right/bright seeds170/172. The actor sees only RGB/depth/pose/track evidence, not reaction labels. A claim needs observed stillness and V373 fresh track plus scripted owner yes. Responses are procedural; no speech audio, real person/servo timing or hardware sensor selection.",
        "source_sha256": sha(SOURCE),
        "v384_source_sha256": sha(Path(live.__file__)),
        "gap_tracker_source_sha256": sha(Path(
            GapLatchedCandidateTracks.__init__.__code__.co_filename)),
        "reports": reports,
        "rows": rows,
    }
    (OUT / "cued-owner-show-audit.json").write_text(
        json.dumps(aggregate, indent=2) + "\n")
    (OUT / SOURCE.name).write_bytes(SOURCE.read_bytes())


if __name__ == "__main__":
    main()
