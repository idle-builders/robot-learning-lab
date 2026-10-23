"""V357: challenge V355's fixed -0.5rad posture with person/prop changes.

Evaluator-only 20ms geometry. The posture and far request remain fixed from
development evidence; this does not train a policy or prove physical safety.
"""

from __future__ import annotations

import json
from pathlib import Path

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
from run_shin_intrusion_posture_v324 import actual_pose
from screen_handoff_posture_v355 import prop_gap


SOURCE = Path(__file__)
OUT = ROOT / "artifacts/ape-closed-loop-cove50-v357"
V353 = ROOT / "artifacts/ape-closed-loop-cove50-v353"
SEEDS = (170, 172)
VARIANTS = (
    {"name": "base", "offset_xyz_m": (0., 0., 0.), "arm_extra_radius_m": 0.},
    {"name": "left_shift", "offset_xyz_m": (0., .12, 0.), "arm_extra_radius_m": 0.},
    {"name": "right_shift", "offset_xyz_m": (0., -.12, 0.), "arm_extra_radius_m": 0.},
    {"name": "farther", "offset_xyz_m": (.12, 0., 0.), "arm_extra_radius_m": 0.},
    {"name": "thicker_arm", "offset_xyz_m": (0., 0., 0.),
     "arm_extra_radius_m": .02},
)
PHASE_SHIFTS_S = (-.2, 0., .2)
REACHES = {"near": NEAR_HAND_XYZ_M, "far": FAR_HAND_XYZ_M}
TARGET_SHOULDER_RAD = -.5


def run_case(seed: int, variant: dict, phase_shift_s: float,
             reach: str, *, target_shoulder_rad: float = TARGET_SHOULDER_RAD,
             out: Path = OUT) -> dict:
    label = (f"seed{seed}-{variant['name']}-{reach}-"
             f"phase{phase_shift_s:+.1f}-shoulder{target_shoulder_rad:+.2f}")
    branch = out / label
    branch.mkdir(parents=True, exist_ok=True)
    prior = (V353 / "compliant_far" / f"seed{seed}-present" /
             f"seed{seed}-present")
    trace_path = prior / "episode-trace.json"
    invite_path = prior / "handoff-invitation.npz"
    trace = json.loads(trace_path.read_text())
    with np.load(invite_path) as packet:
        qpos = packet["actor_qpos"].copy()
        estimate = packet["estimated_pose_xyyaw"].copy()
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
        room.estimated[:] = estimate
        room.data.qpos[:] = qpos
        room.head = .8
        shoulder = room.model.joint("left_shoulder_pitch").qposadr[0]
        start_shoulder = float(qpos[shoulder])
        robot_geoms = RobotGeometry(room.model).geom_ids
        offset = np.asarray(variant["offset_xyz_m"], dtype=float)
        room.model.body_pos[room.model.body("loop_person").id] = offset
        extra = variant["arm_extra_radius_m"]
        for name in ("loop_person_upper_arm_geom",
                     "loop_person_forearm_geom", "loop_person_hand_geom"):
            room.model.geom_size[room.model.geom(name).id, 0] += extra
        samples = []
        for step in range(41):
            if step <= 20:
                fraction = 0.
                room.data.qpos[shoulder] = (start_shoulder + step / 20 *
                    (target_shoulder_rad - start_shoulder))
                phase = "prepare"
            else:
                fraction = (step - 20) / 20
                room.data.qpos[shoulder] = target_shoulder_rad
                phase = "offer"
            # Change the moving prop's phase while holding owner timing fixed.
            room.time_s = 16.2 + .02 * step + phase_shift_s
            room._update_moving_obstacle()
            hand = START_HAND_XYZ_M + fraction * (
                REACHES[reach] - START_HAND_XYZ_M)
            set_person_pose(room.model, hand)
            room.model.body_pos[room.model.body(
                "loop_TomatoSauce").id] += offset
            room._forward()
            person = gap_to_person(room, robot_geoms)
            prop = prop_gap(room, robot_geoms)
            scene = room._current_obstacle_clearance()
            samples.append({"relative_time_s": round(.02 * step, 3),
                            "phase": phase,
                            "evaluator_person_gap_m": round(person[0], 5),
                            "evaluator_prop_gap_m": round(prop[0], 5),
                            "evaluator_scene_gap_m": round(scene[0], 5)})
        min_person = min(s["evaluator_person_gap_m"] for s in samples)
        min_prop = min(s["evaluator_prop_gap_m"] for s in samples)
        return {"case": label, "seed": seed, "variant": variant,
                "target_shoulder_rad": target_shoulder_rad,
                "prop_phase_shift_s": phase_shift_s, "reach": reach,
                "source_invitation_packet": str(invite_path.relative_to(ROOT)),
                "source_invitation_sha256": sha(invite_path),
                "source_episode_trace_sha256": sha(trace_path),
                "evaluator_min_person_gap_m": min_person,
                "evaluator_min_prop_gap_m": min_prop,
                "evaluator_min_scene_gap_m": min(
                    s["evaluator_scene_gap_m"] for s in samples),
                "evaluator_under_5cm_person": min_person < .05,
                "evaluator_under_5cm_prop": min_prop < .05,
                "samples_20ms": samples}
    finally:
        room.close()


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    rows = [run_case(seed, variant, phase, reach)
            for seed in SEEDS for variant in VARIANTS
            for phase in PHASE_SHIFTS_S for reach in REACHES]
    report = {
        "scope": "Evaluator-only 20ms selected-body posture/reach geometry from the V353 invitation qpos. Freeze shoulder target -0.5rad from V355; vary person root by 12cm, increase upper/forearm/hand radii by2cm, and phase the moving low prop by +/-0.2s. Owner waits0.4s then reaches near or far over0.4s. Robot/person are kinematic, no actor prediction, native joint execution, contact force or stop model. Seed170/172 share essentially the same local room geometry.",
        "source_sha256": sha(SOURCE),
        "selected_visual_sha256": sha(selected_visual()),
        "rows": rows,
        "score": {
            "total": len(rows),
            "person_under_5cm": sum(r["evaluator_under_5cm_person"]
                                     for r in rows),
            "prop_under_5cm": sum(r["evaluator_under_5cm_prop"]
                                   for r in rows),
        },
    }
    path = OUT / "prepared-posture-challenge.json"
    path.write_text(json.dumps(report, indent=2) + "\n")
    (OUT / SOURCE.name).write_bytes(SOURCE.read_bytes())
    print(report["score"], flush=True)
    for row in rows:
        if row["evaluator_under_5cm_person"] or row["evaluator_under_5cm_prop"]:
            print(row["case"], "person", row["evaluator_min_person_gap_m"],
                  "prop", row["evaluator_min_prop_gap_m"], flush=True)


if __name__ == "__main__":
    main()
