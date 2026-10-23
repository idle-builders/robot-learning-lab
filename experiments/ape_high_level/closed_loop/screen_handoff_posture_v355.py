"""V355: can pre-invitation arm posture reduce person risk without prop risk?

Replay the V353 guarded invitation joint/pose. The owner waits during a 0.4s
shoulder transition, then reaches near/far over 0.4s. Every 20ms evaluator
scores person, moving low prop and all-scene gaps. This is geometry only;
direct qpos motion is not a qualified joint skill or human-safety controller.
"""

from __future__ import annotations

import json
from pathlib import Path

import imageio.v2 as imageio
import mujoco
import numpy as np

import direct_structural_rooms_v270 as structural
from articulated_person_scene import (FAR_HAND_XYZ_M, NEAR_HAND_XYZ_M,
                                      START_HAND_XYZ_M, add_person,
                                      set_person_pose)
from audit_articulated_owner_v342 import gap_to_person
from direct_structural_rooms_v270 import selected_visual
from robot_geometry import RobotGeometry
from run import ROOT, sha
from run_late_corridor_intrusion_v323 import LateCorridorRoom
from run_shin_intrusion_posture_v324 import actual_pose, estimated_pose
from run_blocked_owner_help_v317 import panel


SOURCE = Path(__file__)
OUT = ROOT / "artifacts/ape-closed-loop-cove50-v355"
V353 = ROOT / "artifacts/ape-closed-loop-cove50-v353"
SEEDS = (170, 172)
POSTURES = {"guarded": -.8, "minus0p7": -.7, "minus0p6": -.6,
            "minus0p5": -.5, "minus0p4": -.4, "minus0p2": -.2,
            "neutral": 0.}
REACHES = {"near": NEAR_HAND_XYZ_M, "far": FAR_HAND_XYZ_M}
START_S = 16.2
PREP_S = .4
OFFER_S = .4


def prop_gap(room, robot_geoms: tuple[int, ...]) -> tuple:
    prop = room.model.geom("loop_stool_geom").id
    return min((float(mujoco.mj_geomDistance(
        room.model, room.data, robot_geom, prop, 10., None)),
        room.model.geom(robot_geom).name)
        for robot_geom in robot_geoms)


def run_case(seed: int, posture: str, reach: str) -> dict:
    label = f"seed{seed}-{posture}-{reach}"
    branch = OUT / label
    branch.mkdir(parents=True, exist_ok=True)
    prior = (V353 / "compliant_far" / f"seed{seed}-present" /
             f"seed{seed}-present")
    trace = json.loads((prior / "episode-trace.json").read_text())
    with np.load(prior / "handoff-invitation.npz") as packet:
        invitation_qpos = packet["actor_qpos"].copy()
        invitation_estimate = packet["estimated_pose_xyyaw"].copy()
    original_model = structural.structural_model
    structural.structural_model = (
        lambda kind, side: add_person(original_model(kind, side)))
    try:
        room = LateCorridorRoom(case=tuple(trace["case"]),
                                pose_source="command", branch=branch)
    finally:
        structural.structural_model = original_model
    try:
        room.layout(seed, absent=False, furnished=True)
        room.actual[:] = actual_pose(trace)
        room.estimated[:] = invitation_estimate
        room.data.qpos[:] = invitation_qpos
        room.head = .8
        shoulder = room.model.joint("left_shoulder_pitch").qposadr[0]
        first_joint = float(invitation_qpos[shoulder])
        target_joint = POSTURES[posture]
        robot_geoms = RobotGeometry(room.model).geom_ids
        samples = []
        video = branch / "posture-offer.mp4"
        with imageio.get_writer(video, fps=5, codec="libx264", quality=7) as writer:
            for step in range(41):
                if step <= 20:
                    time_s = START_S + PREP_S * step / 20
                    fraction = 0.
                    room.data.qpos[shoulder] = (
                        first_joint + step / 20 * (target_joint - first_joint))
                    phase = "prepare"
                else:
                    time_s = START_S + PREP_S + OFFER_S * (step - 20) / 20
                    fraction = (step - 20) / 20
                    room.data.qpos[shoulder] = target_joint
                    phase = "offer"
                room.time_s = time_s
                room._update_moving_obstacle()
                hand = START_HAND_XYZ_M + fraction * (
                    REACHES[reach] - START_HAND_XYZ_M)
                set_person_pose(room.model, hand)
                room._forward()
                human = gap_to_person(room, robot_geoms)
                prop = prop_gap(room, robot_geoms)
                scene = room._current_obstacle_clearance()
                samples.append({"time_s": round(time_s, 3), "phase": phase,
                                "shoulder_pitch_rad": round(float(
                                    room.data.qpos[shoulder]), 4),
                                "evaluator_person_gap_m": round(human[0], 5),
                                "evaluator_prop_gap_m": round(prop[0], 5),
                                "evaluator_scene_gap_m": round(scene[0], 5),
                                "evaluator_closest_person_pair":
                                    [human[1], human[2]],
                                "evaluator_closest_prop_robot": prop[1]})
                if step % 5 == 0:
                    rgb, _ = room.capture("lower_rgb", depth=False)
                    writer.append_data(panel(
                        room, rgb, f"V355 {label} {phase} "
                        f"t{time_s:.2f} person {human[0]:.3f} "
                        f"prop {prop[0]:.3f}"))
        return {"case": label, "seed": seed,
                "evaluator_posture": posture,
                "evaluator_reach": reach,
                "actor_invitation_qpos_source": str((prior /
                    "handoff-invitation.npz").relative_to(ROOT)),
                "actor_invitation_qpos_sha256": sha(prior /
                    "handoff-invitation.npz"),
                "evaluator_min_person_gap_m": round(min(
                    s["evaluator_person_gap_m"] for s in samples), 5),
                "evaluator_min_prop_gap_m": round(min(
                    s["evaluator_prop_gap_m"] for s in samples), 5),
                "evaluator_min_scene_gap_m": round(min(
                    s["evaluator_scene_gap_m"] for s in samples), 5),
                "samples_20ms": samples,
                "video_path": str(video.relative_to(ROOT)),
                "video_sha256": sha(video)}
    finally:
        room.close()


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    rows = [run_case(seed, posture, reach)
            for seed in SEEDS for posture in POSTURES for reach in REACHES]
    report = {"scope": "Evaluator geometry screen from the actor's V353 guarded invitation state on selected Cove-50 Direct Frame. Owner waits0.4s while left shoulder stays guarded or moves through one of six partial/neutral endpoint angles, then executes a near/far articulated0.4s offer. Person, late moving stool and scene gaps are checked every20ms. Human and robot movement are direct kinematic poses, with no force/contact response or native joint skill. These gaps are not actor-perceived or safety-certified.",
              "source_sha256": sha(SOURCE),
              "selected_visual_sha256": sha(selected_visual()),
              "person_scene_source_sha256": sha(Path(add_person.__code__.co_filename)),
              "rows": rows}
    (OUT / "posture-offer-screen.json").write_text(
        json.dumps(report, indent=2) + "\n")
    (OUT / SOURCE.name).write_bytes(SOURCE.read_bytes())
    for row in rows:
        print(row["case"], "person", row["evaluator_min_person_gap_m"],
              "prop", row["evaluator_min_prop_gap_m"],
              "scene", row["evaluator_min_scene_gap_m"], flush=True)


if __name__ == "__main__":
    main()
