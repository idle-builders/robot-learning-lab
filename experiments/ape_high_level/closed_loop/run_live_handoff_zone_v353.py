"""V353: actor-selected far invitation in the live blocked-owner-help loop.

The room/search/low-prop guard and object vision are the V317/V332 workflow.
An actor-zone request precedes a scripted compliant or ignored owner offer.
10Hz actor RGB-D monitors the arm, but owner and robot do not physically stop.
"""

from __future__ import annotations

import json
from pathlib import Path

import imageio.v2 as imageio
import numpy as np
import torch
from transformers import AutoModelForZeroShotObjectDetection, AutoProcessor

import direct_structural_rooms_v270 as structural
from active_scan import DETECTOR
from articulated_person_scene import (FAR_HAND_XYZ_M, ITEM_SIDE_OFFSET_M,
                                      NEAR_HAND_XYZ_M, START_HAND_XYZ_M,
                                      add_person, set_person_pose)
from audit_articulated_owner_v342 import gap_to_person
from audit_direct_localization_v297 import depth_variant
from audit_person_grounding_v344 import detect
from bearing_person_belief import BearingPersonBelief
from direct_structural_rooms_v270 import selected_visual
from handoff_zone import choose_invitation_zone
from replay_endpoint_depth_gap_v350 import box_depth_points, score_points
from robot_geometry import RobotGeometry
from run import ROOT, sha
import run_blocked_owner_help_v317 as owner_module
from run_late_corridor_intrusion_v323 import LateCorridorRoom
from run_live_hazard_guard_v332 import LiveGuard
from self_depth_mask import SelfDepthMask


SOURCE = Path(__file__)
OUT = ROOT / "artifacts/ape-closed-loop-cove50-v353"
SEEDS = (170, 172)
CONDITIONS = ("compliant_far", "ignored_near")
ACTOR_BODY_WARNING_M = .10
ACTOR_OFFER_ZONE_TOLERANCE_M = .10


def move_offer(room, branch: Path, video, seed: int, mode: str,
               condition: str, processor, model, device: str) -> dict:
    if mode == "absent":
        room.wait()
        return {"person_motion_started": False, "actor_samples": [],
                "actor_first_warning_s": None,
                "actor_final_body_gap_m": None,
                "evaluator_minimum_human_gap_m": None}
    start_s = float(room.time_s)
    hand_target = (FAR_HAND_XYZ_M if condition == "compliant_far"
                   else NEAR_HAND_XYZ_M)
    item = "loop_TomatoSauce" if mode == "present" else "loop_Imposter"
    robot = RobotGeometry(room.model)
    mask = SelfDepthMask(room.model, camera="lower_rgb",
                         width=640, height=480, agreement_m=.04)
    belief = BearingPersonBelief()
    fovy_lower = float(room.model.cam_fovy[
        room.model.camera("lower_rgb").id])
    fovy_upper = float(room.model.cam_fovy[
        room.model.camera("upper_rgb").id])
    minimum_human = float("inf")
    minimum_scene = float("inf")
    evaluator_human_path = []
    samples = []
    try:
        for step in range(21):
            fraction = step / 20
            room.time_s = start_s + .4 * fraction
            room._update_moving_obstacle()
            hand = START_HAND_XYZ_M + fraction * (
                hand_target - START_HAND_XYZ_M)
            set_person_pose(room.model, hand, carried_item=item)
            room._forward()
            human_gap = gap_to_person(room, robot.geom_ids)[0]
            scene_gap = room._current_obstacle_clearance()[0]
            minimum_human = min(minimum_human, human_gap)
            minimum_scene = min(minimum_scene, scene_gap)
            evaluator_human_path.append({
                "time_s": round(room.time_s, 3),
                "person_gap_m": round(human_gap, 5)})
            room.path_min_clearance = min(
                room.path_min_clearance, room._current_obstacle_clearance())
            if scene_gap < 0:
                room.motion_interrupted = True
                break
            if step % 5:
                continue
            index = step // 5
            lower, ideal = room.capture("lower_rgb", depth=True)
            upper, _ = room.capture("upper_rgb", depth=False)
            lower_arm = detect(processor, model, device, lower,
                               "hand . arm .")
            upper_person = detect(processor, model, device, upper,
                                  "person .")
            short = depth_variant(ideal, "short", seed, 1600 + index)
            long = depth_variant(ideal, "noisy_long", seed, 1600 + index)
            lower_pose = room.camera_pose_estimate("lower_rgb")
            upper_pose = room.camera_pose_estimate("upper_rgb")
            actor_qpos = room.pose_data.qpos.copy()
            state = belief.ingest(
                room.time_s, upper_person, lower_arm,
                upper_camera_rotation=upper_pose[1],
                upper_camera_fovy_deg=fovy_upper)
            robot.set_state(actor_qpos)
            box = (max(lower_arm, key=lambda d: d["score"])["box_xyxy"]
                   if lower_arm else None)
            gap_by_sensor = {}
            for name, measured in (("short", short), ("noisy_long", long)):
                filtered, info = mask.apply(
                    measured, actor_qpos, max_depth_m=2.5)
                points = box_depth_points(filtered, box, lower_pose,
                                          fovy_lower)
                gap_by_sensor[name] = {**score_points(points, robot),
                                       "self_mask": info}
            warning = bool(gap_by_sensor["noisy_long"]["min_body_gap_m"]
                           is not None and gap_by_sensor["noisy_long"][
                               "min_body_gap_m"] < ACTOR_BODY_WARNING_M)
            packet = branch / f"owner-offer-{condition}-{index:02d}.npz"
            np.savez_compressed(
                packet, lower_rgb=np.asarray(lower), upper_rgb=np.asarray(upper),
                lower_short_depth_m=short,
                lower_noisy_long_depth_m=long,
                camera_origin_estimated_m=lower_pose[0],
                camera_rotation_estimated=lower_pose[1],
                actor_qpos=actor_qpos,
                estimated_pose_xyyaw=room.estimated.copy(),
                capture_time_s=room.time_s)
            samples.append({
                "time_s": round(room.time_s, 3),
                "actor_packet_path": str(packet.relative_to(ROOT)),
                "actor_packet_sha256": sha(packet),
                "actor_person_belief": state,
                "actor_arm_detections": lower_arm,
                "actor_depth_body_gap": gap_by_sensor,
                "actor_warning": warning,
                "evaluator_human_gap_m": round(human_gap, 5)})
            video.append_data(owner_module.panel(
                room, lower, f"V353 {condition} seed{seed} {mode} "
                f"t{room.time_s:.1f} | {'hold' if warning else 'observe'}"))
    finally:
        mask.close()
    first_warning = next((s["time_s"] for s in samples
                          if s["actor_warning"]), None)
    final_gap = (samples[-1]["actor_depth_body_gap"]["noisy_long"][
        "min_body_gap_m"] if samples else None)
    return {"person_motion_started": True,
            "actor_samples": samples,
            "actor_first_warning_s": first_warning,
            "actor_final_body_gap_m": final_gap,
            "evaluator_20ms_person_gap": evaluator_human_path,
            "evaluator_first_below_5cm_s": next(
                (p["time_s"] for p in evaluator_human_path
                 if p["person_gap_m"] < .05), None),
            "evaluator_minimum_human_gap_m": round(minimum_human, 5),
            "evaluator_minimum_scene_gap_m": round(minimum_scene, 5),
            "evaluator_owner_followed_request": condition == "compliant_far"}


def main(*, out: Path = OUT, conditions: tuple[str, ...] = CONDITIONS,
         prepare=None, variant: str = "V353",
         allow_research_reposition: bool = False,
         enforce_invitation_zone: bool = False) -> None:
    out.mkdir(parents=True, exist_ok=True)
    device = "mps" if torch.backends.mps.is_available() else "cpu"
    processor = AutoProcessor.from_pretrained(DETECTOR)
    detector = AutoModelForZeroShotObjectDetection.from_pretrained(
        DETECTOR).to(device).eval()
    original_model = structural.structural_model
    original_room = owner_module.HeldoutMovingRoom
    original_search = owner_module.search.run_case
    original_out = owner_module.OUT
    original_cases = owner_module.CASES
    structural.structural_model = (
        lambda kind, side: add_person(original_model(kind, side)))
    rows = []
    try:
        for condition in conditions:
            offered = []

            def guarded_run_case(*args, **kwargs):
                seed, mode = args[:2]

                def owner_help(room, observe, branch: Path, video):
                    guard = LiveGuard(room, branch, video, seed,
                                      f"v353-{condition}-seed{seed}-{mode}")
                    guard.run()
                    if room.motion_interrupted:
                        response = {"status": "guard_interrupted_before_invite",
                                    "owner_questions": 0,
                                    "autonomous_find": False}
                        response["hazard_guard"] = guard.save(response["status"])
                        return response
                    room.head = .8  # Hypothetical pre-invitation head skill.
                    room._forward()
                    preparation = (prepare(room, branch, video, seed, mode,
                                           condition) if prepare else None)
                    if room.motion_interrupted:
                        response = {"status": "preparation_scene_contact_stop",
                                    "owner_questions": 0,
                                    "autonomous_find": False,
                                    "actor_preparation": preparation}
                        response["hazard_guard"] = guard.save(response["status"])
                        return response
                    rgb, _ = room.capture("lower_rgb", depth=False)
                    upper, _ = room.capture("upper_rgb", depth=False)
                    invite_packet = branch / "handoff-invitation.npz"
                    np.savez_compressed(
                        invite_packet, lower_rgb=np.asarray(rgb),
                        upper_rgb=np.asarray(upper),
                        actor_qpos=room.data.qpos.copy(),
                        estimated_pose_xyyaw=room.estimated.copy(),
                        capture_time_s=room.time_s)
                    zone = choose_invitation_zone(
                        RobotGeometry(room.model), room.data.qpos.copy(),
                        room.estimated.copy())
                    invitation = {
                        "actor_time_s": round(room.time_s, 3),
                        "actor_packet_path": str(invite_packet.relative_to(ROOT)),
                        "actor_packet_sha256": sha(invite_packet),
                        "preparation": preparation,
                        **zone}
                    video.append_data(owner_module.panel(
                        room, rgb, f"{variant} seed{seed} {mode}: please offer at "
                        f"{zone['chosen_zone'] or 'no safe zone'}"))
                    if zone["chosen_zone"] != "far":
                        response = {"status": "no_invitation_zone_hold",
                                    "owner_questions": 0,
                                    "autonomous_find": False,
                                    "actor_invitation": invitation}
                    else:
                        motion = move_offer(room, branch, video, seed, mode,
                                            condition, processor, detector,
                                            device)
                        offered.append({"seed": seed, "mode": mode,
                                        "condition": condition,
                                        "actor_invitation": invitation,
                                        "offer_motion": motion})
                        base = {"owner_questions": 1,
                                "autonomous_find": False,
                                "actor_invitation": invitation,
                                "offer_motion": motion}
                        if room.motion_interrupted:
                            response = {"status": "scene_contact_during_offer_stop",
                                        **base}
                        elif mode == "absent":
                            room.look(0., "lower_rgb")
                            observe("lower_rgb", "owner could not show item")
                            response = {"status": "route_help_no_item_shown",
                                        "owner_answer": "cannot_show", **base}
                        elif (motion["actor_final_body_gap_m"] is None or
                              motion["actor_final_body_gap_m"] <
                              ACTOR_BODY_WARNING_M):
                            response = {"status": "offer_risk_hold_after_late_warning",
                                        "owner_answer": "unasked", **base}
                        else:
                            room.look(0., "lower_rgb")
                            first = observe("lower_rgb",
                                            "owner-presented fresh candidate")
                            if first["accepted_box"] is None:
                                response = {"status": "route_help_candidate_not_seen",
                                            "owner_answer": "unasked", **base}
                            elif (enforce_invitation_zone and
                                  first["accepted_world_xyz_m"] is None):
                                response = {
                                    "status": "owner_offer_zone_unknown_reask",
                                    "owner_answer": "unasked", **base}
                            elif (enforce_invitation_zone and
                                  float(np.linalg.norm(
                                      np.asarray(first[
                                          "accepted_world_xyz_m"][:2]) -
                                      (np.asarray(zone[
                                          "actor_zone_point_m"][:2]) +
                                       ITEM_SIDE_OFFSET_M[:2]))) >
                                  ACTOR_OFFER_ZONE_TOLERANCE_M):
                                observed = first["accepted_world_xyz_m"]
                                requested = (np.asarray(zone[
                                    "actor_zone_point_m"]) +
                                    ITEM_SIDE_OFFSET_M).tolist()
                                response = {
                                    "status": "owner_offer_outside_requested_zone_reask",
                                    "owner_answer": "unasked", **base,
                                    "actor_observed_item_xyz_m": observed,
                                    "actor_requested_item_xyz_m": requested,
                                    "actor_offer_zone_tolerance_m":
                                        ACTOR_OFFER_ZONE_TOLERANCE_M,
                                }
                            elif mode == "identical_lookalike":
                                response = {"status": "owner_rejected_assisted_lookalike",
                                            **base, "owner_questions": 2,
                                            "owner_answer": "no"}
                            else:
                                room.wait()
                                again = observe(
                                    "lower_rgb", "post-owner fresh recheck")
                                stable = (again["accepted_box"] is not None and
                                          owner_module.iou(first["accepted_box"],
                                                           again["accepted_box"]) >= .6)
                                response = {
                                    "status": ("owner_assisted_visual_reacquisition"
                                               if stable else
                                               "owner_yes_recheck_failed"),
                                    **base,
                                    "owner_questions": 2,
                                    "owner_answer": "yes",
                                    "fresh_same_region": stable,
                                }
                    response["hazard_guard"] = guard.save(response["status"])
                    if (allow_research_reposition and preparation is not None):
                        response["research_kinematic_reposition"] = True
                        response["evaluator_preparation_min_scene_gap_m"] = (
                            preparation["evaluator_min_scene_gap_m"])
                    return response

                kwargs["on_route_blocked"] = owner_help
                kwargs["allow_research_reposition"] = allow_research_reposition
                return original_search(*args, **kwargs)

            owner_module.HeldoutMovingRoom = LateCorridorRoom
            owner_module.search.run_case = guarded_run_case
            owner_module.OUT = out / condition
            owner_module.CASES = tuple(
                case for case in original_cases
                if case[0] in SEEDS and
                (condition == "compliant_far" or case[1] == "present"))
            owner_module.main()
            audit = json.loads((out / condition /
                                "blocked-owner-help-audit.json").read_text())
            for row in audit["rows"]:
                seed, mode = row["case"][:2]
                match = next((r for r in offered
                              if r["seed"] == seed and r["mode"] == mode), None)
                rows.append({"condition": condition,
                             **{**row, "owner_help_requested": True},
                             "actor_handoff": match})
    finally:
        structural.structural_model = original_model
        owner_module.HeldoutMovingRoom = original_room
        owner_module.search.run_case = original_search
        owner_module.OUT = original_out
        owner_module.CASES = original_cases
    report = {
        "scope": "Live selected Cove-50 blocked-route owner-help continuation with V332 low-prop guard, actor own-body far-zone selection before invitation, articulated person/can reaching at20ms, 10Hz actor RGB-D/DINO/bearing belief and existing fresh candidate/owner answer check. Compliant owner follows requested far point; noncompliant owner follows near path. Optional pre-invitation preparation is recorded per case. The owner does not stop on actor warnings, person/robot are kinematic, near-path risk is intentionally scored rather than called safe; audio speech, human dynamics, native stop and physical pickup are absent.",
        "source_sha256": sha(SOURCE),
        "preparation_source_sha256": (sha(Path(prepare.__code__.co_filename))
                                      if prepare else None),
        "zone_source_sha256": sha(SOURCE.with_name("handoff_zone.py")),
        "bearing_source_sha256": sha(SOURCE.with_name(
            "bearing_person_belief.py")),
        "v317_source_sha256": sha(Path(owner_module.__file__)),
        "search_source_sha256": sha(Path(
            owner_module.search.run_case.__code__.co_filename)),
        "research_kinematic_reposition_allowed": allow_research_reposition,
        "actor_invitation_zone_gate_enabled": enforce_invitation_zone,
        "actor_offer_zone_tolerance_m": (
            ACTOR_OFFER_ZONE_TOLERANCE_M if enforce_invitation_zone else None),
        "selected_visual_sha256": sha(selected_visual()),
        "detector_weights_sha256": sha(DETECTOR / "model.safetensors"),
        "rows": rows,
    }
    (out / "live-handoff-zone-audit.json").write_text(
        json.dumps(report, indent=2) + "\n")
    (out / SOURCE.name).write_bytes(SOURCE.read_bytes())
    for row in rows:
        print(row["condition"], row["case"][0], row["case"][1],
              row["status"], flush=True)


if __name__ == "__main__":
    main()
