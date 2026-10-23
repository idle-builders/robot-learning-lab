"""Use actor-only no-route evidence to ask for help at the first veto.

This is the paired V263p recovery-depth-loss fault with one task-level repair:
the initial blocked information-view gate now leads to an owner request after
all four nearby 8cm options are screened, without any extra base motion.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from run import ROOT, iou, sha
import run_blocked_forward_viewpoint_v321 as live
import run_no_supported_view_owner_help_v256p as help_policy
import run_varied_owner_route_v258p as challenge
from screen_stopped_alternate_routes_v255p import replay_stage


SOURCE = Path(__file__)
OUT = ROOT / "artifacts/ape-articulated-depth-route-v265p"
BASE = ROOT / "artifacts/ape-articulated-depth-route-v263p"
VARIATION_SOURCE = Path(challenge.__file__)
HELP_SOURCE = Path(help_policy.__file__)


def on_initial_veto(room, observe, branch: Path, video, robot,
                    recovery: dict, seed: int) -> dict:
    del branch
    alternatives = replay_stage(recovery, room.model, robot)
    if any(route["status"] == "free" for route in
           alternatives["options"].values()):
        raise AssertionError("Initial-veto help despite supported route")
    event = {"request_time_s": round(room.time_s, 4),
             "actor_statement": (
                 "I cannot see a safe way closer. Could you show me "
                 "the item from here?"),
             "claim": "item_not_found_no_supported_view",
             "alternate_routes_actor_only": alternatives,
             "response_mode_evaluator_script": (
                 "compliant" if seed in challenge.COMPLIANT_SEEDS else "ignored")}
    rgb, _ = room.capture("lower_rgb", depth=False)
    video.append_data(live.panel(
        room, rgb, f"depth unavailable | seed {seed} | ask owner to show"))
    before_moves = len(room.applied_slips)
    room.wait()
    if room.motion_interrupted:
        status, questions = "contact_during_owner_help_stop", 1
    elif seed not in challenge.COMPLIANT_SEEDS:
        status, questions = "owner_help_ignored_unresolved", 1
    else:
        # This scene intervention is the scripted owner's response. The actor
        # sees only its next image; object truth is used for the owner's answer.
        body = room.model.body("loop_TomatoSauce").id
        room.model.body_pos[body] = help_policy.OWNER_SHOW_XYZ_M
        room._forward()
        gap = float(room._current_obstacle_clearance()[0])
        event["evaluator_presentation_model_gap_m"] = round(gap, 5)
        if gap < .05:
            raise AssertionError("Owner presentation violates 5cm scene gap")
        room.look(0., "lower_rgb")
        first = observe("lower_rgb", "owner-presented fresh candidate")
        event["first_actor_candidate_box"] = first["accepted_box"]
        if first["accepted_box"] is None:
            status, questions = "owner_help_candidate_not_seen", 1
        elif not first["evaluator_correct_item"]:
            event["owner_identity_answer"] = "no"
            status, questions = "owner_rejected_assisted_lookalike", 2
        else:
            event["owner_identity_answer"] = "yes"
            room.wait()
            if room.motion_interrupted:
                status, questions = "contact_after_owner_answer_stop", 2
            else:
                again = observe("lower_rgb", "post-owner fresh recheck")
                stable = (again["accepted_box"] is not None and
                          iou(first["accepted_box"], again["accepted_box"]) >= .6)
                event["fresh_same_region"] = stable
                status = ("owner_assisted_visual_reacquisition" if stable
                          else "owner_yes_recheck_failed")
                questions = 2
    event["base_moves_during_help"] = len(room.applied_slips) - before_moves
    assert event["base_moves_during_help"] == 0
    event["status"] = status
    return {"audit": event, "status": status, "owner_questions": questions}


def main() -> None:
    old_scene = {seed: spec.copy() for seed, spec in challenge.SCENE.items()}
    old_out, old_source = challenge.OUT, challenge.SOURCE
    old_profile, old_description = challenge.profile_depth, challenge.SENSOR_PROFILE
    old_hook, old_actor = challenge.INITIAL_VETO_HOOK, challenge.ACTOR_POLICY

    def recovery_loss(profile: str, original):
        healthy = old_profile(profile, original)

        def measured(ideal: np.ndarray, mode: str, seed: int,
                     serial: int) -> np.ndarray:
            if mode == "short" and serial >= 400:
                return np.full_like(ideal, np.nan)
            return healthy(ideal, mode, seed, serial)

        return measured

    try:
        challenge.SCENE[180]["prop_x_m"] = -.20
        challenge.SCENE[184]["prop_x_m"] = -.20
        challenge.OUT, challenge.SOURCE = OUT, SOURCE
        challenge.profile_depth = recovery_loss
        challenge.SENSOR_PROFILE = (
            "virtual head depth 30cm-1m through earlier search; all "
            "serial>=400 recovery/map head-depth pixels missing; RGB, "
            "near-range and shin preserved")
        challenge.INITIAL_VETO_HOOK = on_initial_veto
        challenge.ACTOR_POLICY = (
            "V256p actor plus initial-veto actor-only alternate screen and "
            "owner-help transition; no scene label enters route decisions")
        challenge.main()
    finally:
        challenge.SCENE.clear()
        challenge.SCENE.update(old_scene)
        challenge.OUT, challenge.SOURCE = old_out, old_source
        challenge.profile_depth = old_profile
        challenge.SENSOR_PROFILE = old_description
        challenge.INITIAL_VETO_HOOK = old_hook
        challenge.ACTOR_POLICY = old_actor
    for source in (VARIATION_SOURCE, HELP_SOURCE):
        (OUT / source.name).write_bytes(source.read_bytes())
    path = OUT / "report.json"
    report = json.loads(path.read_text())
    report.update({
        "variation_source_sha256": sha(VARIATION_SOURCE),
        "help_policy_source_sha256": sha(HELP_SOURCE),
        "paired_v263_report_sha256": sha(BASE / "report.json"),
        "repair": "On initial recovery-depth veto, replay current actor "
                  "head/range/shin packets for four 8cm alternatives; when "
                  "none is supported, remain still and request owner help.",
    })
    path.write_text(json.dumps(report, indent=2) + "\n")


if __name__ == "__main__":
    main()
