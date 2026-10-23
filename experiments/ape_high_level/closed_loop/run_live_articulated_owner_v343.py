"""V343: live owner-help loop with a crouched articulated person.

The selected ape search, six-shin low-hazard guard, image gate and owner answer
are unchanged from V338–V339. Only the scripted owner geometry and carried
item trajectory replace the simple floating forearm.
"""

from __future__ import annotations

import json
from pathlib import Path

import mujoco
import numpy as np

from articulated_person_scene import (FAR_HAND_XYZ_M, PERSON_GEOMS,
                                      START_HAND_XYZ_M, add_person,
                                      set_person_pose)
from audit_articulated_owner_v342 import gap_to_person
from audit_direct_localization_v297 import depth_variant
from direct_structural_rooms_v270 import selected_visual
from robot_geometry import RobotGeometry
from run import ROOT, sha
import run_blocked_owner_help_v317 as owner_module
import run_live_moving_owner_offer_v338 as prior
from run_owner_reach_pause_v337 import actor_warning


SOURCE = Path(__file__)
OUT = ROOT / "artifacts/ape-closed-loop-cove50-v343"
OWNER_SHOW_XYZ_M = tuple(FAR_HAND_XYZ_M + np.asarray((0., -.08, 0.)))


def move_person_during_wait(room, branch: Path, video, seed: int,
                            mode: str, condition: str) -> dict:
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
        for step, fraction in enumerate(np.linspace(0., 1., 21)):
            room.time_s = start_time + .4 * fraction
            room._update_moving_obstacle()
            hand = (START_HAND_XYZ_M + fraction *
                    (FAR_HAND_XYZ_M - START_HAND_XYZ_M))
            set_person_pose(room.model, hand, carried_item=item)
            room._forward()
            human = gap_to_person(room, robot_geoms)
            scene = room._current_obstacle_clearance()
            minimum_human = min(minimum_human, human)
            minimum_scene = min(minimum_scene, scene)
            room.path_min_clearance = min(room.path_min_clearance, scene)
            if scene[0] < 0:
                room.motion_interrupted = True
                break
            if step % 5:
                continue
            rgb, ideal = room.capture("lower_rgb", depth=True)
            index = step // 5
            packet = branch / f"articulated-owner-{index:02d}.npz"
            np.savez_compressed(
                packet, lower_rgb=np.asarray(rgb),
                lower_noisy_long_depth_m=depth_variant(
                    ideal, "noisy_long", seed, 800 + index),
                lower_short_depth_m=depth_variant(
                    ideal, "short", seed, 800 + index),
                actor_qpos=room.data.qpos.copy(),
                estimated_pose_xyyaw=room.estimated.copy(),
                capture_time_s=room.time_s)
            packets.append(packet)
            if len(packets) > 1:
                observations.append({
                    "time_s": round(room.time_s, 3),
                    **actor_warning(packets[-2], packet,
                                    "noisy_long_depth", shoulder)})
            video.append_data(owner_module.panel(
                room, rgb, f"V343 {condition} seed {seed} {mode} | "
                f"articulated owner offer t{room.time_s:.1f}"))

    room.wait = moving_wait
    return {"restore_wait": original_wait, "results": lambda: {
        "person_motion_started": started,
        "evaluator_minimum_human_clearance_m": round(minimum_human[0], 5)
        if started else None,
        "evaluator_minimum_scene_clearance_m": round(minimum_scene[0], 5)
        if started else None,
        "owner_actor_packets": [
            {"path": str(path.relative_to(ROOT)), "sha256": sha(path)}
            for path in packets],
        "owner_actor_motion_observations": observations}}


def main() -> None:
    original = (prior.OUT, prior.add_owner_hand,
                prior.move_owner_during_wait,
                owner_module.OWNER_SHOW_XYZ_M)
    try:
        prior.OUT = OUT
        prior.add_owner_hand = add_person
        prior.move_owner_during_wait = move_person_during_wait
        owner_module.OWNER_SHOW_XYZ_M = OWNER_SHOW_XYZ_M
        prior.main()
    finally:
        (prior.OUT, prior.add_owner_hand,
         prior.move_owner_during_wait,
         owner_module.OWNER_SHOW_XYZ_M) = original
    audit_path = OUT / "live-moving-owner-offer-audit.json"
    audit = json.loads(audit_path.read_text())
    audit["scope"] = ("V343 live selected Cove-50 owner-help search with the "
                      "same V317 appearance/owner-answer task and V332 "
                      "six-shin low-prop protection as V339. A scripted "
                      "crouched person's visible torso, head, upper arm, "
                      "forearm and hand carry the item sideways over0.4s "
                      "through the first owner-help wait. Camera/depth "
                      "packets and person/scene evaluator gaps are retained. "
                      "The person does not respond to actor warnings; "
                      "body/arm contact dynamics and native robot actions "
                      "remain unmodeled.")
    audit["source_sha256"] = sha(SOURCE)
    audit["v338_source_sha256"] = sha(Path(prior.__file__))
    audit["person_scene_source_sha256"] = sha(Path(add_person.__code__.co_filename))
    audit["owner_show_xyz_m"] = OWNER_SHOW_XYZ_M
    audit_path.write_text(json.dumps(audit, indent=2) + "\n")
    (OUT / SOURCE.name).write_bytes(SOURCE.read_bytes())


if __name__ == "__main__":
    main()
