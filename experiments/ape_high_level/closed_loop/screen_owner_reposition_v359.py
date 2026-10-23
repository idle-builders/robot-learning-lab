"""V359: can a pre-invitation body retreat clear the revealed hard reach?

Evaluator-only geometry from the selected Cove-50 invitation state. The
owner waits while a kinematic base/shoulder transition occurs. No native
walking, stop path, actor scene map or social compliance is claimed.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

import direct_structural_rooms_v270 as structural
from articulated_person_scene import (FAR_HAND_XYZ_M, START_HAND_XYZ_M,
                                      add_person, set_person_pose)
from audit_articulated_owner_v342 import gap_to_person
from direct_structural_rooms_v270 import selected_visual
from robot_geometry import RobotGeometry
from run import ROOT, sha
from run_late_corridor_intrusion_v323 import LateCorridorRoom
from run_shin_intrusion_posture_v324 import actual_pose
from screen_handoff_posture_v355 import prop_gap


SOURCE = Path(__file__)
OUT = ROOT / "artifacts/ape-closed-loop-cove50-v359"
V353 = ROOT / "artifacts/ape-closed-loop-cove50-v353"
SEED = 170
NEAR_CLOSER = np.asarray((.16, .08, .20))
SHOULDER_TARGET_RAD = -.4
BASE_RETREATS_XY_M = ((-.05, 0.), (-.10, 0.), (-.15, 0.),
                      (-.05, -.05), (-.10, -.05), (-.15, -.05),
                      (-.05, .05), (-.10, .05), (-.15, .05))
VARIANTS = (
    {"name": "base", "offset_xyz_m": (0., 0., 0.), "extra_arm_radius_m": 0.},
    {"name": "left_thick", "offset_xyz_m": (0., .12, 0.),
     "extra_arm_radius_m": .02},
)
REACHES = {"near_closer": NEAR_CLOSER, "far": FAR_HAND_XYZ_M}


def run_case(retreat_xy: tuple[float, float], variant: dict,
             reach: str, *, out: Path = OUT) -> dict:
    label = (f"retreat{retreat_xy[0]:+.2f}_{retreat_xy[1]:+.2f}-"
             f"{variant['name']}-{reach}")
    branch = out / label
    branch.mkdir(parents=True, exist_ok=True)
    prior = (V353 / "compliant_far" / f"seed{SEED}-present" /
             f"seed{SEED}-present")
    trace_path = prior / "episode-trace.json"
    packet_path = prior / "handoff-invitation.npz"
    trace = json.loads(trace_path.read_text())
    with np.load(packet_path) as packet:
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
        room.layout(SEED, absent=False, furnished=True)
        room.actual[:] = actual_pose(trace)
        room.estimated[:] = estimate
        room.data.qpos[:] = qpos
        room.head = .8
        base_xy = room.actual[:2].copy()
        shoulder = room.model.joint("left_shoulder_pitch").qposadr[0]
        initial_shoulder = float(qpos[shoulder])
        robot_geoms = RobotGeometry(room.model).geom_ids
        offset = np.asarray(variant["offset_xyz_m"], dtype=float)
        room.model.body_pos[room.model.body("loop_person").id] = offset
        for name in ("loop_person_upper_arm_geom", "loop_person_forearm_geom",
                     "loop_person_hand_geom"):
            room.model.geom_size[room.model.geom(name).id, 0] += variant[
                "extra_arm_radius_m"]
        samples = []
        for step in range(41):
            if step <= 20:
                fraction = 0.
                progress = step / 20
                room.actual[:2] = base_xy + progress * np.asarray(retreat_xy)
                room.data.qpos[shoulder] = (initial_shoulder + progress *
                    (SHOULDER_TARGET_RAD - initial_shoulder))
                phase = "prepare"
            else:
                fraction = (step - 20) / 20
                phase = "offer"
            room.time_s = 16.2 + .02 * step
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
                            "base_xy_m": room.actual[:2].round(5).tolist(),
                            "evaluator_person_gap_m": round(person[0], 5),
                            "evaluator_prop_gap_m": round(prop[0], 5),
                            "evaluator_scene_gap_m": round(scene[0], 5)})
        person_min = min(s["evaluator_person_gap_m"] for s in samples)
        prop_min = min(s["evaluator_prop_gap_m"] for s in samples)
        scene_min = min(s["evaluator_scene_gap_m"] for s in samples)
        return {"case": label, "retreat_xy_m": retreat_xy,
                "variant": variant, "reach": reach,
                "source_invitation_sha256": sha(packet_path),
                "source_episode_trace_sha256": sha(trace_path),
                "evaluator_min_person_gap_m": person_min,
                "evaluator_min_prop_gap_m": prop_min,
                "evaluator_min_scene_gap_m": scene_min,
                "evaluator_meets_5cm_margin": min(person_min, prop_min,
                                                   scene_min) >= .05,
                "samples_20ms": samples}
    finally:
        room.close()


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    rows = [run_case(retreat, variant, reach)
            for retreat in BASE_RETREATS_XY_M for variant in VARIANTS
            for reach in REACHES]
    summary = []
    for retreat in BASE_RETREATS_XY_M:
        selected = [r for r in rows if r["retreat_xy_m"] == retreat]
        summary.append({"retreat_xy_m": retreat,
                        "all_four_meet_5cm_margin": all(
                            r["evaluator_meets_5cm_margin"]
                            for r in selected),
                        "worst_scene_gap_m": min(
                            r["evaluator_min_scene_gap_m"]
                            for r in selected),
                        "worst_person_gap_m": min(
                            r["evaluator_min_person_gap_m"]
                            for r in selected),
                        "worst_prop_gap_m": min(
                            r["evaluator_min_prop_gap_m"]
                            for r in selected)})
    report = {
        "scope": "Evaluator-only 20ms selected-body screen of a 5-15cm kinematic base retreat in nine directions while shoulder moves to -0.4rad over0.4s, then normal far or closer ignored near owner reach. Base and shifted-left/thicker-arm variants use the V353 guarded invitation state. Scene/person/prop gap includes preparation and offer. No actor route discovery, human compliance, native walking/stop or contact dynamics; revealed V358 hard case, not a held-out pass.",
        "source_sha256": sha(SOURCE),
        "selected_visual_sha256": sha(selected_visual()),
        "summary": summary, "rows": rows}
    (OUT / "reposition-screen.json").write_text(
        json.dumps(report, indent=2) + "\n")
    (OUT / SOURCE.name).write_bytes(SOURCE.read_bytes())
    for item in summary:
        print(item, flush=True)


if __name__ == "__main__":
    main()
