"""V393: retry a cued owner show after a delayed or ignored hold.

The actor requests another show if its first hold leaves fewer than two
fresh accepted samples. A delayed scripted owner moves the item into view
and holds it; an ignored owner continues moving. Owner behavior is a
procedural test fixture.
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
OUT = ROOT / "artifacts/ape-closed-loop-cove50-v393"
VARIANT = next(row for row in POSITIVE_VARIANTS
               if row["name"] == "right_bright")
RESPONSE_MODES = ("compliant", "delayed", "ignored")
HOLD_DELAY_S = .3
STILL_STEP_M = .025
RETRY_INDEX = 13
RETRY_RESPONSE_DELAY_S = .1
RETRY_MOVE_S = .2


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
    original_show_steps = base.SHOW_EVALUATOR_STEPS

    def reset() -> None:
        state.clear()
        state.update({"last_elapsed": -1., "wiggle_at": None,
                      "hold_at": None, "hold_index": None,
                      "frozen_hand": None, "last_selected": None,
                      "last_selected_index": None, "still_run": 0,
                      "hold_observed": False, "accepted_run": 0,
                      "retry_at": None})

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
        retry_at = state["retry_at"]
        if retry_at is not None and response_mode != "ignored":
            retry_start = retry_at + RETRY_RESPONSE_DELAY_S
            if elapsed_s >= retry_start - 1e-9:
                fraction = min(1., max(0., (elapsed_s - retry_start) /
                                       RETRY_MOVE_S))
                hand = ((1. - fraction) * natural_hand(response_at) +
                        fraction * natural_hand(.8))
        set_person_pose(room.model, hand, carried_item=carried_item)
        room.model.body_pos[room.model.body("loop_person").id] = offset
        if carried_item is not None:
            room.model.body_pos[room.model.body(carried_item).id] += offset
        if not hasattr(room, "_v393_base_light"):
            room._v393_base_light = room.model.light_diffuse.copy()
        room.model.light_diffuse[:] = (room._v393_base_light *
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
        if not state["hold_observed"]:
            choice["accepted_box"] = None
            choice["accepted_world_xyz_m"] = None
            choice["accepted_score"] = None
            choice["accepted_hue_intersection"] = None
        state["accepted_run"] = (state["accepted_run"] + 1
                                 if choice["accepted_world_xyz_m"] is not None
                                 else 0)
        if index == RETRY_INDEX and state["accepted_run"] < 2:
            state["retry_at"] = index * .1
            state["hold_observed"] = False
            state["still_run"] = 0
            state["last_selected"] = None
            state["last_selected_index"] = None
            state["accepted_run"] = 0
            choice["accepted_box"] = None
            choice["accepted_world_xyz_m"] = None
            choice["accepted_score"] = None
            choice["accepted_hue_intersection"] = None
            events.append({"action": "please_raise_and_hold_it_in_view_again",
                           "elapsed_s": round(index * .1, 3)})
        choice["actor_social_events"] = events
        choice["actor_hold_observed"] = state["hold_observed"]
        choice["actor_still_link_run"] = state["still_run"]
        choice["actor_accepted_run"] = state["accepted_run"]
        choice["actor_retry_requested"] = state["retry_at"] is not None
        return choice

    try:
        live.staged_scene = cued_scene
        live.frame_available = frame_available
        live.multi_candidate_infer = social_infer
        live.OUT = OUT / response_mode
        live.MovingCandidateTracks = GapLatchedCandidateTracks
        base.add_person = lambda model: add_person(
            model, shirt_rgba=VARIANT["shirt_rgba"])
        base.SHOW_EVALUATOR_STEPS = 96
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
        base.SHOW_EVALUATOR_STEPS = original_show_steps
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
            "actor_show_request_count": len(events),
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
        "V393 selected Cove-50 guarded no-contact owner show, right/bright "
        "shifted person and raised identical replica. Actor requests wiggle "
        "then hold after observed motion. At index13 it requests raise/show "
        "again only if fewer than two accepted current samples exist. The "
        "scripted owner response mode is " + response_mode + "; the actor "
        "does not see that mode. A delayed owner moves over0.2s from its "
        "poorly visible held pose to a visible held pose; ignored continues "
        "wiggling. Retry resets stillness; two adjacent <=2.5cm steps and "
        "V373's two-frame/120ms gate are required. The show lasts1.9s at "
        "10Hz, with a20ms evaluator. The inherited owner_questions counter "
        "excludes extra show events; consult actor_show_request_count. "
        "Speech and dynamics remain procedural.")
    report["v393_source_sha256"] = sha(SOURCE)
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
        "scope": "V393 extends V391 to1.9s, adding an actor retry at1.3s only when fewer than two current accepted samples exist. Scripted delayed owner moves into visible hold after that request; ignored owner does not comply. Actor sees RGB/depth/pose/track evidence, not response labels. Fresh claim needs observed stillness, V373 two-frame/120ms track and scripted owner yes. Response dynamics and speech remain procedural; this cannot select hardware sensors.",
        "source_sha256": sha(SOURCE),
        "v384_source_sha256": sha(Path(live.__file__)),
        "gap_tracker_source_sha256": sha(Path(
            GapLatchedCandidateTracks.__init__.__code__.co_filename)),
        "reports": reports,
        "rows": rows,
    }
    (OUT / "cued-retry-show-audit.json").write_text(
        json.dumps(aggregate, indent=2) + "\n")
    (OUT / SOURCE.name).write_bytes(SOURCE.read_bytes())


if __name__ == "__main__":
    main()
