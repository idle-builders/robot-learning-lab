"""V368: live teach/find owner help by looking at a distant held item.

The owner remains at the visible starting pose instead of reaching into the
ape's protected arm volume. The existing guarded blocked-route controller
asks for a show, takes a fresh RGB-D view, and checks owner identity/review.
Speech/owner response and robot motion remain scripted/kinematic.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

import direct_structural_rooms_v270 as structural
from articulated_person_scene import (START_HAND_XYZ_M, add_person,
                                      set_person_pose)
from audit_articulated_owner_v342 import gap_to_person
from direct_structural_rooms_v270 import selected_visual
from robot_geometry import RobotGeometry
from run import ROOT, sha
import run_blocked_owner_help_v317 as owner_module
from run_late_corridor_intrusion_v323 import LateCorridorRoom
from run_live_hazard_guard_v332 import LiveGuard


SOURCE = Path(__file__)
OUT = ROOT / "artifacts/ape-closed-loop-cove50-v368"
SEEDS = (170, 172)


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    original_model = structural.structural_model
    original_room = owner_module.HeldoutMovingRoom
    original_search = owner_module.search.run_case
    original_out = owner_module.OUT
    original_cases = owner_module.CASES
    structural.structural_model = (
        lambda kind, side: add_person(original_model(kind, side)))
    show_rows = []
    try:
        def guarded_run_case(*args, **kwargs):
            seed, mode = args[:2]

            def owner_help(room, observe, branch: Path, video):
                guard = LiveGuard(room, branch, video, seed,
                                  f"v368-seed{seed}-{mode}")
                guard.run()
                if room.motion_interrupted:
                    response = {"status": "guard_interrupted_before_show",
                                "owner_questions": 0,
                                "autonomous_find": False}
                    response["hazard_guard"] = guard.save(response["status"])
                    return response
                room.head = .8  # Hypothetical head-look skill.
                carried = ("loop_TomatoSauce" if mode == "present" else
                           "loop_Imposter" if mode == "identical_lookalike"
                           else None)
                set_person_pose(room.model, START_HAND_XYZ_M,
                                carried_item=carried)
                room._forward()
                lower, _ = room.capture("lower_rgb", depth=False)
                upper, _ = room.capture("upper_rgb", depth=False)
                packet_path = branch / "distant-show-invitation.npz"
                np.savez_compressed(
                    packet_path, lower_rgb=np.asarray(lower),
                    upper_rgb=np.asarray(upper),
                    actor_qpos=room.data.qpos.copy(),
                    estimated_pose_xyyaw=room.estimated.copy(),
                    capture_time_s=room.time_s)
                invitation = {
                    "actor_time_s": round(room.time_s, 3),
                    "social_action": "please_hold_the_item_where_you_are",
                    "actor_packet_path": str(packet_path.relative_to(ROOT)),
                    "actor_packet_sha256": sha(packet_path),
                    "physical_contact_authorized": False}
                video.append_data(owner_module.panel(
                    room, lower, f"V368 seed{seed} {mode}: show from there"))
                first = observe("lower_rgb", "owner distant first look")
                base = {"actor_invitation": invitation,
                        "owner_questions": 1,
                        "autonomous_find": False,
                        "evaluator_person_gap_m": round(gap_to_person(
                            room, RobotGeometry(room.model).geom_ids)[0], 5)}
                if mode == "absent":
                    response = {"status": "route_help_no_item_shown",
                                "owner_answer": "cannot_show",
                                "actor_first_candidate_seen":
                                    first["accepted_box"] is not None,
                                **base}
                elif first["accepted_box"] is None:
                    response = {"status": "route_help_candidate_not_seen",
                                "owner_answer": "unasked", **base}
                elif mode == "identical_lookalike":
                    response = {"status": "owner_rejected_assisted_lookalike",
                                "owner_answer": "no", **base,
                                "owner_questions": 2}
                else:
                    room.wait()
                    again = observe("lower_rgb", "owner distant fresh recheck")
                    stable = (again["accepted_box"] is not None and
                              owner_module.iou(first["accepted_box"],
                                               again["accepted_box"]) >= .6)
                    response = {
                        "status": ("owner_assisted_visual_reacquisition"
                                   if stable else
                                   "owner_yes_recheck_failed"),
                        "owner_answer": "yes", **base,
                        "owner_questions": 2,
                        "fresh_same_region": stable,
                    }
                response["hazard_guard"] = guard.save(response["status"])
                show_rows.append({"seed": seed, "mode": mode,
                                  "response": response})
                return response

            kwargs["on_route_blocked"] = owner_help
            return original_search(*args, **kwargs)

        owner_module.HeldoutMovingRoom = LateCorridorRoom
        owner_module.search.run_case = guarded_run_case
        owner_module.OUT = OUT
        owner_module.CASES = tuple(
            case for case in original_cases if case[0] in SEEDS)
        owner_module.main()
    finally:
        structural.structural_model = original_model
        owner_module.HeldoutMovingRoom = original_room
        owner_module.search.run_case = original_search
        owner_module.OUT = original_out
        owner_module.CASES = original_cases
    report_path = OUT / "blocked-owner-help-audit.json"
    report = json.loads(report_path.read_text())
    report["scope"] = (
        "V368 live selected Cove-50 guarded blocked-route owner help: "
        "articulated owner stays at the original visible hand pose while "
        "holding the taught can, an identical lookalike, or no item. Ape "
        "requests a no-contact show, sets head yaw0.8rad, uses the existing "
        "fresh RGB-D taught-object and owner-answer/recheck loop, and "
        "retains the moving-low-prop hazard guard. No owner reach or base "
        "reposition is invited. Scripted owner answers, direct-qpos head, "
        "visual-only object acquisition; no sound, physical grasp, native "
        "stopping, or general human behavior.")
    report["v368_source_sha256"] = sha(SOURCE)
    report["selected_visual_sha256"] = sha(selected_visual())
    report["owner_show_rows"] = show_rows
    report_path.write_text(json.dumps(report, indent=2) + "\n")
    (OUT / SOURCE.name).write_bytes(SOURCE.read_bytes())
    for row in report["rows"]:
        print(row["case"][0], row["case"][1], row["status"], flush=True)


if __name__ == "__main__":
    main()
