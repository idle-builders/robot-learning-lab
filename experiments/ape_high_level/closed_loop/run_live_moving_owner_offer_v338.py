"""V338: carry an item through the live owner-help wait with a visible hand.

This tests whether the existing task vision still works when a scripted owner
actually moves the can into the agreed farther presentation volume. It does
not claim the actor can control or predict the person's motion.
"""

from __future__ import annotations

import json
from pathlib import Path

import mujoco
import numpy as np

import direct_structural_rooms_v270 as structural
from audit_direct_localization_v297 import depth_variant
from direct_structural_rooms_v270 import selected_visual
from owner_hand_scene import (END_BODY_XYZ_M, HAND_LOCAL_XYZ_M, OWNER_BODY,
                              START_BODY_XYZ_M, add_owner_hand)
from robot_geometry import RobotGeometry
from run import ROOT, sha
import run_blocked_owner_help_v317 as owner_module
from run_late_corridor_intrusion_v323 import LateCorridorRoom
from run_live_hazard_guard_v332 import LiveGuard, ParkedLateRoom
from run_owner_reach_pause_v337 import actor_warning


SOURCE = Path(__file__)
OUT = ROOT / "artifacts/ape-closed-loop-cove50-v338"
SEEDS = (170, 172)
CASES = tuple(case for case in owner_module.CASES if case[0] in SEEDS)


def move_owner_during_wait(room, branch: Path, video, seed: int,
                           mode: str, condition: str) -> dict:
    """Replace just the first owner-help wait with 20ms scripted hand motion."""
    start_time = room.time_s
    original_wait = room.wait
    item = "loop_TomatoSauce" if mode == "present" else "loop_Imposter"
    minimum_human = (float("inf"), "", "")
    minimum_scene = (float("inf"), "", "")
    packets = []
    observations = []
    robot_geoms = RobotGeometry(room.model).geom_ids
    shoulder = int(room.model.joint("left_shoulder_pitch").qposadr[0])
    started = False

    def moving_wait():
        nonlocal started, minimum_human, minimum_scene
        if started or mode == "absent":
            return original_wait()
        started = True
        start = np.asarray(START_BODY_XYZ_M)
        end = np.asarray(END_BODY_XYZ_M)
        for index, fraction in enumerate(np.linspace(0., 1., 21)):
            room.time_s = start_time + .4 * fraction
            room._update_moving_obstacle()
            body_pos = start + fraction * (end - start)
            room.model.body_pos[room.model.body(OWNER_BODY).id] = body_pos
            room.model.body_pos[room.model.body(item).id] = (
                body_pos + HAND_LOCAL_XYZ_M)
            room._forward()
            human = min((float(mujoco.mj_geomDistance(
                room.model, room.data, robot_geom,
                room.model.geom(human_geom).id, 10., None)),
                room.model.geom(robot_geom).name, human_geom)
                for robot_geom in robot_geoms
                for human_geom in ("loop_owner_forearm_geom",
                                   "loop_owner_hand_geom"))
            minimum_human = min(minimum_human, human)
            scene = room._current_obstacle_clearance()
            minimum_scene = min(minimum_scene, scene)
            room.path_min_clearance = min(room.path_min_clearance, scene)
            if scene[0] < 0:
                room.motion_interrupted = True
                break
            if index % 5 == 0:
                rgb, ideal = room.capture("lower_rgb", depth=True)
                measured = depth_variant(ideal, "noisy_long", seed, 800 + index)
                packet = branch / f"owner-hand-{index:02d}.npz"
                np.savez_compressed(
                    packet, lower_rgb=np.asarray(rgb),
                    lower_noisy_long_depth_m=measured,
                    lower_short_depth_m=depth_variant(
                        ideal, "short", seed, 800 + index),
                    actor_qpos=room.data.qpos.copy(),
                    estimated_pose_xyyaw=room.estimated.copy(),
                    capture_time_s=room.time_s)
                packets.append(packet)
                if len(packets) > 1:
                    observations.append({"time_s": round(room.time_s, 3),
                                         **actor_warning(packets[-2], packet,
                                                         "noisy_long_depth",
                                                         shoulder)})
                video.append_data(owner_module.panel(
                    room, rgb, f"V338 {condition} seed {seed} {mode} | "
                    f"owner offer t{room.time_s:.1f}"))

    room.wait = moving_wait
    return {"restore_wait": original_wait, "results": lambda: {
        "hand_motion_started": started,
        "evaluator_minimum_human_clearance_m": round(minimum_human[0], 5)
        if started else None,
        "evaluator_minimum_scene_clearance_m": round(minimum_scene[0], 5)
        if started else None,
        "owner_hand_actor_packets": [
            {"path": str(p.relative_to(ROOT)), "sha256": sha(p)} for p in packets],
        "owner_hand_actor_motion_observations": observations}}


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    rows = []
    original_model = structural.structural_model
    original_room = owner_module.HeldoutMovingRoom
    original_run_case = owner_module.search.run_case
    original_out = owner_module.OUT
    original_cases = owner_module.CASES
    structural.structural_model = (lambda kind, side:
                                   add_owner_hand(original_model(kind, side)))
    try:
        for condition, room_type in (("moving", LateCorridorRoom),
                                     ("parked", ParkedLateRoom)):
            actor_motions = []

            def guarded_run_case(*args, **kwargs):
                owner_callback = kwargs["on_route_blocked"]
                seed, mode = args[:2]

                def guarded_callback(room, observe, branch, video):
                    guard = LiveGuard(room, branch, video, seed,
                                      f"v338-seed{seed}-{mode}-{condition}")
                    guard.run()
                    wrapped = move_owner_during_wait(
                        room, branch, video, seed, mode, condition)
                    try:
                        response = owner_callback(room, observe, branch, video)
                    finally:
                        room.wait = wrapped["restore_wait"]
                    motion = wrapped["results"]()
                    actor_motions.append({"seed": seed, "mode": mode, **motion})
                    response["hazard_guard"] = guard.save(response["status"])
                    response["owner_motion"] = motion
                    return response

                kwargs["on_route_blocked"] = guarded_callback
                return original_run_case(*args, **kwargs)

            owner_module.HeldoutMovingRoom = room_type
            owner_module.search.run_case = guarded_run_case
            owner_module.OUT = OUT / condition
            owner_module.CASES = CASES
            owner_module.main()
            audit = json.loads((OUT / condition /
                                "blocked-owner-help-audit.json").read_text())
            for row in audit["rows"]:
                seed, mode = row["case"][:2]
                motion = next(x for x in actor_motions
                              if x["seed"] == seed and x["mode"] == mode)
                rows.append({"condition": condition, **row,
                             "owner_motion": motion})
    finally:
        structural.structural_model = original_model
        owner_module.HeldoutMovingRoom = original_room
        owner_module.search.run_case = original_run_case
        owner_module.OUT = original_out
        owner_module.CASES = original_cases
    report = {
        "scope": "V338 owner-help integration using selected Direct Frame, V332 six-shin hazard guard, existing V317 appearance/confirmation task, and a visible 20ms scripted human forearm/hand carrying the item through the first 0.4s owner wait into the farther offered volume. The actor captures five lower RGB/noisy-long-depth packets during that wait; the image-motion proposal is diagnostic and does not control the scripted person. Human compliance, arm geometry, base/look and object pickup remain kinematic; the generic appearance gate decides subsequent candidate acceptance from fresh images without simulator IDs.",
        "source_sha256": sha(SOURCE),
        "v317_source_sha256": sha(Path(owner_module.__file__)),
        "selected_visual_sha256": sha(selected_visual()),
        "owner_scene_source_sha256": sha(Path(add_owner_hand.__code__.co_filename)),
        "rows": rows,
    }
    (OUT / "live-moving-owner-offer-audit.json").write_text(
        json.dumps(report, indent=2) + "\n")
    (OUT / SOURCE.name).write_bytes(SOURCE.read_bytes())


if __name__ == "__main__":
    main()
