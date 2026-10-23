"""One owner response after an actor-certified no-supported-view state.

The owner script is external scene control. It is consulted only after the
ape has used its current RGB-D/range/shin evidence to refuse every nearby
8cm route and has requested help. Exact item identity comes from the owner
answer; the robot must still see and recheck a fresh candidate.
"""

from __future__ import annotations

from pathlib import Path

from run import iou
import run_blocked_forward_viewpoint_v321 as live
from run_no_supported_view_owner_help_v256p import OWNER_SHOW_XYZ_M
from run_slip_aware_third_view_v252p import two_extra_views
from screen_stopped_alternate_routes_v255p import replay_stage


RESPONSES: dict[int, str] = {}
VALID_RESPONSES = {"show_target", "show_lookalike", "cannot_show", "ignored"}


def ask_and_resolve(room, observe, video, robot, stage: dict,
                    seed: int) -> dict:
    alternatives = replay_stage(stage, room.model, robot)
    if any(option["status"] == "free" for option in
           alternatives["options"].values()):
        raise AssertionError("Help requested with an actor-supported route")
    response = RESPONSES[seed]
    if response not in VALID_RESPONSES:
        raise ValueError(f"Unsupported owner script {response}")
    audit = {"request_time_s": round(room.time_s, 4),
             "actor_statement": (
                 "I cannot get a clear view safely. Could you show me "
                 "the item, or tell me if it is gone?"),
             "claim": "item_not_found_no_supported_view",
             "alternate_routes_actor_only": alternatives,
             "owner_response_evaluator_script": response}
    rgb, _ = room.capture("lower_rgb", depth=False)
    video.append_data(live.panel(
        room, rgb, f"no supported view | seed {seed} | ask owner"))
    before_moves = len(room.applied_slips)
    room.wait()
    if room.motion_interrupted:
        status, questions = "contact_during_owner_help_stop", 1
    elif response == "ignored":
        status, questions = "owner_help_ignored_unresolved", 1
    elif response == "cannot_show":
        audit["owner_answer"] = "cannot_show"
        status, questions = "owner_cannot_show_unresolved", 1
    else:
        mode = room.case[1]
        if response == "show_target" and mode != "present":
            raise AssertionError("Owner cannot show a target absent from scene")
        if response == "show_lookalike" and mode != "identical_lookalike":
            raise AssertionError("Lookalike response needs a lookalike scene")
        body_name = ("loop_TomatoSauce" if response == "show_target" else
                     "loop_Imposter")
        body = room.model.body(body_name).id
        room.model.body_pos[body] = OWNER_SHOW_XYZ_M
        room._forward()
        gap = float(room._current_obstacle_clearance()[0])
        audit["evaluator_presented_body"] = body_name
        audit["evaluator_presentation_model_gap_m"] = round(gap, 5)
        if gap < .05:
            raise AssertionError("Owner presentation violates 5cm scene gap")
        room.look(0., "lower_rgb")
        first = observe("lower_rgb", "owner-presented fresh candidate")
        audit["first_actor_candidate_box"] = first["accepted_box"]
        if first["accepted_box"] is None:
            status, questions = "owner_help_candidate_not_seen", 1
        else:
            owner_yes = bool(response == "show_target" and
                             first["evaluator_correct_item"])
            audit["owner_identity_answer"] = "yes" if owner_yes else "no"
            if not owner_yes:
                status, questions = "owner_rejected_assisted_lookalike", 2
            else:
                room.wait()
                if room.motion_interrupted:
                    status, questions = "contact_after_owner_answer_stop", 2
                else:
                    again = observe("lower_rgb", "post-owner fresh recheck")
                    stable = (again["accepted_box"] is not None and
                              iou(first["accepted_box"], again["accepted_box"]) >= .6)
                    audit["fresh_same_region"] = stable
                    status = ("owner_assisted_visual_reacquisition" if stable
                              else "owner_yes_recheck_failed")
                    questions = 2
    audit["base_moves_during_help"] = len(room.applied_slips) - before_moves
    assert audit["base_moves_during_help"] == 0
    audit["status"] = status
    return {"audit": audit, "status": status,
            "owner_questions": questions}


def after_first_view(room, observe, branch: Path, video, robot,
                     filtered_depth, seed: int) -> dict:
    result = two_extra_views(
        room, observe, branch, video, robot, filtered_depth, seed)
    if result["found_event"] is not None:
        return result
    stages = result["audit"]["stages"]
    if not stages or stages[-1]["actor_gate"]["status"] != "blocked":
        return result
    help_result = ask_and_resolve(
        room, observe, video, robot, stages[-1], seed)
    result["audit"]["owner_help"] = help_result["audit"]
    result["status"] = help_result["status"]
    result["owner_questions"] = help_result["owner_questions"]
    return result


def on_initial_veto(room, observe, branch: Path, video, robot,
                    recovery: dict, seed: int) -> dict:
    del branch
    return ask_and_resolve(room, observe, video, robot, recovery, seed)
