"""V373: live no-contact owner help with a moving held item and 3D belief.

After the V332 moving-prop guard, the owner remains at social distance but
fidgets laterally. The ape slews lower gaze, receives exposed RGB and noisy
long depth at10Hz, uses the same taught-object image gate, and associates
robust 3D positions over time before a scripted owner answer/recheck. It
never requests a close reach or a base move. Audio and native skills absent.
"""

from __future__ import annotations

import json
import math
from pathlib import Path

import numpy as np
from PIL import Image

import direct_structural_rooms_v270 as structural
from articulated_person_scene import (START_HAND_XYZ_M, add_person,
                                      set_person_pose)
from audit_articulated_owner_v342 import gap_to_person
from audit_direct_localization_v297 import depth_variant
from direct_structural_rooms_v270 import selected_visual
from robot_geometry import RobotGeometry
from robust_item_depth import locate_box_center
from run import ROOT, sha
import run_blocked_owner_help_v317 as owner_module
from run_color_grounded_entry_v151 import infer
from run_late_corridor_intrusion_v323 import LateCorridorRoom
from run_live_hazard_guard_v332 import LiveGuard


SOURCE = Path(__file__)
OUT = ROOT / "artifacts/ape-closed-loop-cove50-v373"
SEEDS = (170, 172)
OWNER_PATTERN = "faster"
CAPTURE_STEP_S = .1
EVALUATOR_STEP_S = .02
SHOW_EVALUATOR_STEPS = 66
EXPOSURE_S = .02
MAX_TRACK_STEP_M = .12
MAX_ANSWER_AGE_S = .12


def frame_available(seed: int, mode: str, index: int) -> bool:
    """Camera callback availability; scenario wrappers may inject dropouts."""
    return True


def owner_offset(elapsed_s: float) -> np.ndarray:
    return np.asarray((.03 * math.sin(2 * math.pi * elapsed_s / .4),
                       .08 * math.sin(2 * math.pi * elapsed_s / .4), 0.))


def set_scene(room, start_s: float, elapsed_s: float,
              carried_item: str | None) -> None:
    room.time_s = start_s + elapsed_s
    room._update_moving_obstacle()
    room.head = min(.8, 2. * max(0., elapsed_s))
    offset = owner_offset(elapsed_s)
    room.model.body_pos[room.model.body("loop_person").id] = offset
    set_person_pose(room.model, START_HAND_XYZ_M,
                    carried_item=carried_item)
    if carried_item is not None:
        room.model.body_pos[room.model.body(carried_item).id] += offset
    room._forward()


def exposed_frame(room, start_s: float, elapsed_s: float,
                  carried_item: str | None):
    frames = []
    center_depth = None
    for shift in (-EXPOSURE_S / 2, 0., EXPOSURE_S / 2):
        set_scene(room, start_s, max(0., elapsed_s + shift), carried_item)
        rgb, depth = room.capture("lower_rgb", depth=shift == 0.)
        frames.append(np.asarray(rgb, dtype=np.float32))
        if shift == 0.:
            center_depth = depth
    set_scene(room, start_s, elapsed_s, carried_item)
    return Image.fromarray(np.mean(frames, axis=0).astype(np.uint8)), center_depth


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
            seed, mode, vision, exemplar, owner_hist = args[:5]

            def owner_help(room, observe, branch: Path, video):
                guard = LiveGuard(room, branch, video, seed,
                                  f"v373-seed{seed}-{mode}")
                guard.run()
                if room.motion_interrupted:
                    response = {"status": "guard_interrupted_before_show",
                                "owner_questions": 0,
                                "autonomous_find": False}
                    response["hazard_guard"] = guard.save(response["status"])
                    return response
                start_s = float(room.time_s)
                carried = ("loop_TomatoSauce" if mode == "present" else
                           "loop_Imposter" if mode == "identical_lookalike"
                           else None)
                robot_geoms = RobotGeometry(room.model).geom_ids
                minimum_person = float("inf")
                minimum_scene = float("inf")
                samples = []
                track_xyz = None
                track_time = None
                track_run = 0
                ever_confirmed = False
                missing_capture_indices = []
                for substep in range(SHOW_EVALUATOR_STEPS):
                    elapsed_s = substep * EVALUATOR_STEP_S
                    set_scene(room, start_s, elapsed_s, carried)
                    person_gap = gap_to_person(room, robot_geoms)[0]
                    scene_gap = room._current_obstacle_clearance()
                    minimum_person = min(minimum_person, person_gap)
                    minimum_scene = min(minimum_scene, scene_gap[0])
                    room.path_min_clearance = min(room.path_min_clearance,
                                                  scene_gap)
                    if scene_gap[0] < 0:
                        room.motion_interrupted = True
                        break
                    if substep % 5:
                        continue
                    index = substep // 5
                    if not frame_available(seed, mode, index):
                        missing_capture_indices.append(index)
                        continue
                    exposed, ideal = exposed_frame(
                        room, start_s, elapsed_s, carried)
                    measured = depth_variant(ideal, "noisy_long", seed,
                                             2300 + index)
                    measured[(measured < .08) | (measured > 1.5)] = np.nan
                    camera_pose = room.camera_pose_estimate("lower_rgb")
                    choice = infer(vision, exemplar, owner_hist, exposed,
                                   measured, camera_pose, None, None)
                    robust = locate_box_center(
                        choice["accepted_box"], camera_pose, measured,
                        fovy_deg=float(room.model.cam_fovy[
                            room.model.camera("lower_rgb").id]))
                    xyz = robust["world_xyz_m"]
                    linked = bool(
                        xyz is not None and track_xyz is not None and
                        track_time is not None and
                        room.time_s - track_time <= .12 and
                        math.dist(xyz, track_xyz) <= MAX_TRACK_STEP_M)
                    if xyz is None:
                        track_run = 0
                        track_xyz = None
                    else:
                        track_run = track_run + 1 if linked else 1
                        track_xyz = xyz
                        track_time = float(room.time_s)
                    ever_confirmed = ever_confirmed or track_run >= 2
                    packet_path = branch / f"moving-show-{index:02d}.npz"
                    np.savez_compressed(
                        packet_path, exposed_rgb=np.asarray(exposed),
                        noisy_long_depth_m=measured,
                        camera_origin_estimated_m=camera_pose[0],
                        camera_rotation_estimated=camera_pose[1],
                        actor_qpos=room.pose_data.qpos.copy(),
                        estimated_pose_xyyaw=room.estimated.copy(),
                        capture_time_s=room.time_s,
                        exposure_s=EXPOSURE_S)
                    samples.append({
                        "capture_time_s": round(room.time_s, 3),
                        "actor_packet_path": str(packet_path.relative_to(ROOT)),
                        "actor_packet_sha256": sha(packet_path),
                        "actor_choice": choice,
                        "actor_robust_depth": robust,
                        "actor_linked_previous": linked,
                        "actor_track_run": track_run,
                        "actor_track_confirmed": ever_confirmed,
                        "evaluator_person_gap_m": round(person_gap, 5),
                        "evaluator_scene_gap_m": round(scene_gap[0], 5)})
                    video.append_data(owner_module.panel(
                        room, exposed, f"V373 seed{seed} {mode} moving show "
                        f"t{room.time_s:.1f} "
                        f"{'track' if ever_confirmed else 'looking'}"))
                track_age_s = (None if track_time is None else
                               float(room.time_s - track_time))
                track_fresh = bool(
                    track_run >= 2 and track_age_s is not None and
                    track_age_s <= MAX_ANSWER_AGE_S and samples and
                    samples[-1]["actor_choice"]["accepted_box"] is not None)
                base = {"owner_questions": 1, "autonomous_find": False,
                        "actor_social_action":
                            "please_hold_the_item_where_you_are",
                        "actor_track_confirmed": ever_confirmed,
                        "actor_track_final_run": track_run,
                        "actor_track_final_age_s": (
                            None if track_age_s is None else round(
                                track_age_s, 3)),
                        "actor_track_fresh_at_answer": track_fresh,
                        "actor_missing_capture_indices":
                            missing_capture_indices,
                        "actor_samples": samples,
                        "evaluator_min_person_gap_m": round(minimum_person, 5),
                        "evaluator_min_scene_gap_m": round(minimum_scene, 5)}
                if room.motion_interrupted:
                    response = {"status": "person_or_scene_contact_stop",
                                "owner_answer": "unasked", **base}
                elif mode == "absent":
                    response = {"status": "route_help_no_item_shown",
                                "owner_answer": "cannot_show", **base}
                elif not ever_confirmed:
                    response = {"status": "route_help_candidate_not_seen",
                                "owner_answer": "unasked", **base}
                elif mode == "identical_lookalike":
                    response = {"status": "owner_rejected_assisted_lookalike",
                                "owner_answer": "no", **base,
                                "owner_questions": 2}
                elif track_fresh:
                    response = {"status": "owner_assisted_visual_reacquisition",
                                "owner_answer": "yes", **base,
                                "owner_questions": 2,
                                "fresh_moving_track": True}
                else:
                    response = {"status": "owner_yes_recheck_failed",
                                "owner_answer": "yes", **base,
                                "owner_questions": 2,
                                "fresh_moving_track": False}
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
        "V373 selected Cove-50 live guarded blocked-route no-contact owner "
        "help with a fidgeting side-held can/imposter/no item. Owner body "
        "root moves procedurally at a faster 0.4s period while staying at "
        "social distance; ape head yaw0→0.8rad in0.4s, 10Hz 20ms-exposed "
        "RGB plus noisy-long depth, released taught-object vision, robust "
        "5x5 finite-depth location and <=12cm/100ms consecutive 3D point "
        "association. A scripted owner yes/no/cannot-show answer follows "
        "the track, and a yes requires its last accepted point to be <=120ms "
        "old. Independent 20ms person/scene gaps never enter actor "
        "decision. No actual speech, identity proof, native head/stop, real "
        "depth optics or physical grasp.")
    report["v373_source_sha256"] = sha(SOURCE)
    report["robust_depth_source_sha256"] = sha(Path(
        locate_box_center.__code__.co_filename))
    report["selected_visual_sha256"] = sha(selected_visual())
    report["owner_show_rows"] = show_rows
    report_path.write_text(json.dumps(report, indent=2) + "\n")
    (OUT / SOURCE.name).write_bytes(SOURCE.read_bytes())
    for row in report["rows"]:
        print(row["case"][0], row["case"][1], row["status"], flush=True)


if __name__ == "__main__":
    main()
