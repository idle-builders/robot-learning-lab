"""V360: test a geometry-selected retreat in the live owner-help loop.

The ape commands a 10cm back, 5cm side step and partial arm relaxation before
its far offer request. Both base and arm motion are direct kinematics with
command odometry; native gait/stop and human intent are still unqualified.
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
OUT = ROOT / "artifacts/ape-closed-loop-cove50-v360"
RETREAT_XY_M = np.asarray((-.10, -.05))
TARGET_SHOULDER_RAD = -.4
PREPARE_S = .4


def prepare(room, branch: Path, video, seed: int, mode: str,
            condition: str) -> dict:
    start_s = float(room.time_s)
    shoulder = room.model.joint("left_shoulder_pitch").qposadr[0]
    initial_shoulder = float(room.data.qpos[shoulder])
    start_actual = room.actual[:2].copy()
    start_estimated = room.estimated[:2].copy()
    robot_geoms = RobotGeometry(room.model).geom_ids
    samples = []
    for step in range(21):
        fraction = step / 20
        room.time_s = start_s + PREPARE_S * fraction
        room._update_moving_obstacle()
        room.actual[:2] = start_actual + fraction * RETREAT_XY_M
        room.estimated[:2] = start_estimated + fraction * RETREAT_XY_M
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
                  "actor_command_offset_xy_m": (fraction *
                      RETREAT_XY_M).round(5).tolist(),
                  "actor_command_shoulder_pitch_rad": round(float(
                      room.data.qpos[shoulder]), 5),
                  "evaluator_person_gap_m": round(person[0], 5),
                  "evaluator_prop_gap_m": round(prop[0], 5),
                  "evaluator_scene_gap_m": round(scene[0], 5)}
        if step % 5 == 0:
            lower, _ = room.capture("lower_rgb", depth=False)
            packet = branch / f"reposition-{step // 5:02d}.npz"
            np.savez_compressed(
                packet, lower_rgb=np.asarray(lower),
                actor_qpos=room.data.qpos.copy(),
                estimated_pose_xyyaw=room.estimated.copy(),
                capture_time_s=room.time_s,
                commanded_offset_xy_m=fraction * RETREAT_XY_M,
                commanded_shoulder_pitch_rad=room.data.qpos[shoulder])
            sample.update({"actor_packet_path": str(packet.relative_to(ROOT)),
                           "actor_packet_sha256": sha(packet)})
            video.append_data(owner_module.panel(
                room, lower, f"V360 seed{seed} {mode} {condition} "
                f"step {fraction * RETREAT_XY_M[0]:.2f}m "
                f"arm {room.data.qpos[shoulder]:.2f}rad"))
        samples.append(sample)
        if room.motion_interrupted:
            break
    return {
        "policy": "development_geometry_selected_retreat_then_far_request",
        "actor_command_retreat_xy_m": RETREAT_XY_M.tolist(),
        "actor_target_shoulder_pitch_rad": TARGET_SHOULDER_RAD,
        "actor_final_estimated_pose_xyyaw": room.estimated.round(5).tolist(),
        "preparation_duration_s": round(room.time_s - start_s, 3),
        "evaluator_min_person_gap_m": min(s["evaluator_person_gap_m"]
                                          for s in samples),
        "evaluator_min_prop_gap_m": min(s["evaluator_prop_gap_m"]
                                        for s in samples),
        "evaluator_min_scene_gap_m": min(s["evaluator_scene_gap_m"]
                                         for s in samples),
        "samples_20ms": samples,
        "native_gait_or_joint_skill_verified": False,
        "physical_contact_authorized": False,
    }


def main() -> None:
    live.main(out=OUT, prepare=prepare, variant="V360",
              allow_research_reposition=True)
    path = OUT / "live-handoff-zone-audit.json"
    report = json.loads(path.read_text())
    report["scope"] = (
        "V360 live blocked-route owner-help iteration: after the low-prop "
        "guard, directly translate the selected Cove-50 10cm back and 5cm "
        "sideways while its left shoulder moves to -0.4rad over0.4s. "
        "Estimated pose advances by the command, not evaluator correction. "
        "Then select an offer zone and replay compliant far/ignored near "
        "owner paths with the V353 RGB-D/bearing/person and fresh taught-"
        "object/owner-answer loop. 20ms person/prop/scene gaps are independent "
        "evaluation. This uses V359 revealed geometry and kinematic motion; "
        "native execution/stopping, owner response and grasp remain open.")
    report["v360_source_sha256"] = sha(SOURCE)
    report["retreat_xy_m"] = RETREAT_XY_M.tolist()
    report["posture_target_shoulder_rad"] = TARGET_SHOULDER_RAD
    path.write_text(json.dumps(report, indent=2) + "\n")
    (OUT / SOURCE.name).write_bytes(SOURCE.read_bytes())
    for row in report["rows"]:
        handoff = row["actor_handoff"]
        if handoff is None:
            print(row["condition"], row["case"][0], row["case"][1],
                  row["status"], "no invitation", flush=True)
            continue
        prep = handoff["actor_invitation"]["preparation"]
        motion = handoff["offer_motion"]
        print(row["condition"], row["case"][0], row["case"][1],
              row["status"], "zone", handoff["actor_invitation"][
                  "chosen_zone"], "prep scene", prep[
                  "evaluator_min_scene_gap_m"], "offer person", motion[
                  "evaluator_minimum_human_gap_m"], flush=True)


if __name__ == "__main__":
    main()
