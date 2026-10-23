"""Turn a sensor-certified no-route state into an honest owner-help branch.

Reuses V252p's live first/second/third viewpoint loop and V317's bounded
scripted owner presentation. Owner behavior is scenario control, never a
source of route permission or an autonomous find.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from depth_quality import reject_isolated_depth_spikes
from run import ROOT, iou, sha
import run_blocked_forward_viewpoint_v321 as live
from run_medium_depth_recovery_v246p import CASES, MediumWindowMap, profile_depth
from run_slip_aware_third_view_v252p import two_extra_views
from screen_stopped_alternate_routes_v255p import replay_stage


SOURCE = Path(__file__)
OUT = ROOT / "artifacts/ape-articulated-depth-route-v256p"
OWNER_SHOW_XYZ_M = (.35, .08, .20)
# Predeclared response modes, so an ignored request remains an honest task
# failure in the same four owner/prop timing scenes.
COMPLIANT_SEEDS = (170, 174)


def after_first_view(room, observe, branch: Path, video, robot,
                     filtered_depth, seed: int) -> dict:
    result = two_extra_views(
        room, observe, branch, video, robot, filtered_depth, seed)
    if result["found_event"] is not None:
        return result
    stages = result["audit"]["stages"]
    if not stages or stages[-1]["actor_gate"]["status"] != "blocked":
        return result
    alternatives = replay_stage(stages[-1], room.model, robot)
    if any(route["status"] == "free" for route in
           alternatives["options"].values()):
        raise AssertionError("Owner help requested while an observed-free route exists")
    help_audit = {"request_time_s": round(room.time_s, 4),
                  "actor_statement": (
                      "I can see only part of it, and I cannot move safely "
                      "from here. Could you show it to me?"),
                  "claim": "item_not_found_no_supported_view",
                  "alternate_routes_actor_only": alternatives,
                  "response_mode_evaluator_script": (
                      "compliant" if seed in COMPLIANT_SEEDS else "ignored"),
                  "base_moves_during_help": 0}
    rgb, _ = room.capture("lower_rgb", depth=False)
    video.append_data(live.panel(
        room, rgb, f"no supported view | seed {seed} | ask owner to show item"))
    before_moves = len(room.applied_slips)
    room.wait()
    if room.motion_interrupted:
        status, questions = "contact_during_owner_help_stop", 1
    elif seed not in COMPLIANT_SEEDS:
        status, questions = "owner_help_ignored_unresolved", 1
    else:
        # The scripted owner moves the true item; this is external scenario
        # control copied from V317, not a robot action or actor identity input.
        body = room.model.body("loop_TomatoSauce").id
        room.model.body_pos[body] = OWNER_SHOW_XYZ_M
        room._forward()
        gap = float(room._current_obstacle_clearance()[0])
        help_audit["evaluator_presentation_model_gap_m"] = round(gap, 5)
        if gap < .05:
            raise AssertionError("Scripted owner presentation violates 5cm gap")
        room.look(0., "lower_rgb")
        first = observe("lower_rgb", "owner-presented fresh candidate")
        help_audit["first_actor_candidate_box"] = first["accepted_box"]
        if first["accepted_box"] is None:
            status, questions = "owner_help_candidate_not_seen", 1
        elif not first["evaluator_correct_item"]:
            # Exact-instance yes/no comes from the scripted owner, as in V317.
            help_audit["owner_identity_answer"] = "no"
            status, questions = "owner_rejected_assisted_lookalike", 2
        else:
            help_audit["owner_identity_answer"] = "yes"
            room.wait()
            if room.motion_interrupted:
                status, questions = "contact_after_owner_answer_stop", 2
            else:
                again = observe("lower_rgb", "post-owner fresh recheck")
                stable = (again["accepted_box"] is not None and
                          iou(first["accepted_box"], again["accepted_box"]) >= .6)
                help_audit["fresh_same_region"] = stable
                status = ("owner_assisted_visual_reacquisition" if stable
                          else "owner_yes_recheck_failed")
                questions = 2
    help_audit["base_moves_during_help"] = len(room.applied_slips) - before_moves
    assert help_audit["base_moves_during_help"] == 0
    help_audit["status"] = status
    result["audit"]["owner_help"] = help_audit
    result["status"] = status
    result["owner_questions"] = questions
    return result


def main() -> None:
    original_cases = live.CASES
    original_depth = live.noisy_depth
    original_map = live.DepthVoxelMap
    try:
        live.CASES = CASES
        live.noisy_depth = profile_depth("medium_30cm_1m", original_depth)
        live.DepthVoxelMap = MediumWindowMap
        live.main(output_root=OUT, source=SOURCE,
                  map_depth_filter=reject_isolated_depth_spikes,
                  experiment_label="V256p no-supported-view owner help",
                  after_first_view=after_first_view,
                  max_recovery_steps=3)
    finally:
        live.CASES = original_cases
        live.noisy_depth = original_depth
        live.DepthVoxelMap = original_map
    path = OUT / "blocked-forward-viewpoint-audit.json"
    report = json.loads(path.read_text())
    report["scope_override"] = (
        "Same historical kinematic medium-depth V252p owner episodes. After "
        "a blocked third forward gate, replay current actor head/range/shin "
        "packets for forward/back/left/right 8cm alternatives. If none is "
        "supported, ask owner for a show; even seeds 170/174 comply with "
        "V317's instantaneous scripted presentation, 172/176 ignore. "
        "Recognition and fresh owner confirmation remain unchanged. No "
        "autonomous find, physical stop, spoken audio or real human motion "
        "is claimed.")
    path.write_text(json.dumps(report, indent=2) + "\n")
    (OUT / "report.json").write_text(json.dumps({
        "schema": "robot-learning-lab.no-supported-view-owner-help.v1",
        "source_sha256": sha(SOURCE),
        "runner_report_path": str(path.relative_to(ROOT)),
        "runner_report_sha256": sha(path),
        "cases": CASES,
        "compliant_seeds": COMPLIANT_SEEDS,
        "owner_show_xyz_m_evaluator_script": OWNER_SHOW_XYZ_M,
        "rows": [{"case": row["case"], "status": row["status"],
                  "owner_questions": row["owner_questions"],
                  "base_moves": row["base_moves"],
                  "minimum_clearance_m": row[
                      "evaluator_path_min_clearance_m"]}
                 for row in report["rows"]]}, indent=2) + "\n")


if __name__ == "__main__":
    main()
