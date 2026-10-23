"""V356: execute the development-selected intermediate arm posture in the live loop.

The shoulder command is a fixed policy choice from V355, not an inferred
human-arm trajectory or a qualified actuator skill. Its 20 ms replay and
10 Hz joint/image packets test whether the full task still works when the
person reaches after preparation. Evaluator gaps never enter actor decisions.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from audit_articulated_owner_v342 import gap_to_person
from robot_geometry import RobotGeometry
from run import ROOT, sha
import run_blocked_owner_help_v317 as owner_module
import run_live_handoff_zone_v353 as live
from screen_handoff_posture_v355 import prop_gap


SOURCE = Path(__file__)
OUT = ROOT / "artifacts/ape-closed-loop-cove50-v356"
TARGET_SHOULDER_RAD = -.5
PREPARE_S = .4


def prepare(room, branch: Path, video, seed: int, mode: str,
            condition: str) -> dict:
    start_s = float(room.time_s)
    shoulder = room.model.joint("left_shoulder_pitch").qposadr[0]
    initial_shoulder = float(room.data.qpos[shoulder])
    robot_geoms = RobotGeometry(room.model).geom_ids
    samples = []
    for step in range(21):
        fraction = step / 20
        room.time_s = start_s + PREPARE_S * fraction
        room._update_moving_obstacle()
        room.data.qpos[shoulder] = (
            initial_shoulder + fraction *
            (TARGET_SHOULDER_RAD - initial_shoulder))
        room._forward()
        person = gap_to_person(room, robot_geoms)
        prop = prop_gap(room, robot_geoms)
        scene = room._current_obstacle_clearance()
        room.path_min_clearance = min(room.path_min_clearance, scene)
        if scene[0] < 0:
            room.motion_interrupted = True
        sample = {"time_s": round(room.time_s, 3),
                  "actor_command_shoulder_pitch_rad": round(float(
                      room.data.qpos[shoulder]), 5),
                  "evaluator_person_gap_m": round(person[0], 5),
                  "evaluator_prop_gap_m": round(prop[0], 5),
                  "evaluator_scene_gap_m": round(scene[0], 5)}
        if step % 5 == 0:
            lower, _ = room.capture("lower_rgb", depth=False)
            packet = branch / f"shoulder-prep-{step // 5:02d}.npz"
            np.savez_compressed(
                packet, lower_rgb=np.asarray(lower),
                actor_qpos=room.data.qpos.copy(),
                estimated_pose_xyyaw=room.estimated.copy(),
                capture_time_s=room.time_s,
                commanded_shoulder_pitch_rad=room.data.qpos[shoulder])
            sample.update({"actor_packet_path": str(packet.relative_to(ROOT)),
                           "actor_packet_sha256": sha(packet)})
            video.append_data(owner_module.panel(
                room, lower, f"V356 seed{seed} {mode} {condition} "
                f"arm {room.data.qpos[shoulder]:.2f} rad | prepare"))
        samples.append(sample)
        if room.motion_interrupted:
            break
    return {
        "policy": "fixed_development_posture_then_request_far_zone",
        "actor_target_shoulder_pitch_rad": TARGET_SHOULDER_RAD,
        "actor_initial_shoulder_pitch_rad": round(initial_shoulder, 5),
        "actor_final_shoulder_pitch_rad": round(float(
            room.data.qpos[shoulder]), 5),
        "preparation_duration_s": round(room.time_s - start_s, 3),
        "evaluator_min_person_gap_m": min(s["evaluator_person_gap_m"]
                                          for s in samples),
        "evaluator_min_prop_gap_m": min(s["evaluator_prop_gap_m"]
                                        for s in samples),
        "evaluator_min_scene_gap_m": min(s["evaluator_scene_gap_m"]
                                         for s in samples),
        "samples_20ms": samples,
        "native_joint_skill_verified": False,
        "physical_contact_authorized": False,
    }


def main() -> None:
    live.main(out=OUT, prepare=prepare, variant="V356")
    path = OUT / "live-handoff-zone-audit.json"
    report = json.loads(path.read_text())
    report["scope"] = (
        "V356 live owner-help iteration: after the V332 low-prop guard, "
        "kinematically preposition the selected Cove-50's left shoulder "
        "from guard to -0.5rad over0.4s while the owner waits, then choose "
        "the actor's invitation zone and replay compliant far and ignored "
        "near offers. Keep the V353 RGB-D person belief, depth warning, "
        "fresh object/owner-answer gate and 20ms independent evaluator. "
        "The posture is selected on development geometry; real joint "
        "tracking, physical stop, human compliance and grasp are absent.")
    report["v356_source_sha256"] = sha(SOURCE)
    report["posture_target_shoulder_rad"] = TARGET_SHOULDER_RAD
    path.write_text(json.dumps(report, indent=2) + "\n")
    (OUT / SOURCE.name).write_bytes(SOURCE.read_bytes())
    for row in report["rows"]:
        handoff = row["actor_handoff"]
        if handoff is None:
            continue
        prep = handoff["actor_invitation"]["preparation"]
        motion = handoff["offer_motion"]
        print(row["condition"], row["case"][0], row["case"][1],
              row["status"], "prep person", prep["evaluator_min_person_gap_m"],
              "prop", prep["evaluator_min_prop_gap_m"],
              "offer person", motion["evaluator_minimum_human_gap_m"],
              flush=True)


if __name__ == "__main__":
    main()
