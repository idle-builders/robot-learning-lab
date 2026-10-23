"""V369: challenge no-contact owner show under shifted people and lighting.

The owner holds the item at the original hand pose; the selected ape uses
the existing taught-object vision in bounded lower-head looks and its upper
social camera. Person/can geometry scores only after the actor choice.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

import challenge_person_belief_v346 as variants
import direct_structural_rooms_v270 as structural
from articulated_person_scene import (START_HAND_XYZ_M, add_person,
                                      set_person_pose)
from audit_articulated_owner_v342 import gap_to_person
from audit_direct_structural_entry_v271 import V149, V150
from direct_structural_rooms_v270 import selected_visual
from robot_geometry import RobotGeometry
from run import ROOT, Vision, iou, sha
from run_color_grounded_entry_v151 import infer
from run_late_corridor_intrusion_v323 import LateCorridorRoom
from run_shin_intrusion_posture_v324 import actual_pose


SOURCE = Path(__file__)
OUT = ROOT / "artifacts/ape-closed-loop-cove50-v369"
V353 = ROOT / "artifacts/ape-closed-loop-cove50-v353"
SEEDS = (170, 172)
VIEWS = ((.4, "lower_rgb"), (.8, "lower_rgb"),
         (1.2, "lower_rgb"), (.8, "upper_rgb"))


def run_case(seed: int, variant: dict, vision: Vision,
             exemplar: np.ndarray, owner_hist: np.ndarray) -> dict:
    label = f"seed{seed}-{variant['name']}"
    branch = OUT / label
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
        lambda kind, side: add_person(original_model(kind, side),
                                      shirt_rgba=variant["shirt_rgba"]))
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
        room.time_s = 16.2
        room._update_moving_obstacle()
        room.model.light_diffuse[:] *= variant["light_scale"]
        offset = np.asarray(variant["offset_xyz_m"], dtype=float)
        room.model.body_pos[room.model.body("loop_person").id] = offset
        set_person_pose(room.model, START_HAND_XYZ_M,
                        carried_item="loop_TomatoSauce")
        room.model.body_pos[room.model.body(
            "loop_TomatoSauce").id] += offset
        robot_geoms = RobotGeometry(room.model).geom_ids
        views = []
        for yaw, camera in VIEWS:
            room.head = yaw
            room._forward()
            rgb, depth = room.capture(camera, depth=True)
            depth[(depth < .08) | (depth > 1.5)] = np.nan
            pose = room.camera_pose_estimate(camera)
            choice = infer(vision, exemplar, owner_hist, rgb, depth,
                           pose, None, None)
            truth = room.truth_box("TomatoSauce", camera)
            correct = (choice["accepted_box"] is not None and
                       truth is not None and iou(
                           choice["accepted_box"], truth) >= .25)
            packet_path = branch / f"actor-{camera}-yaw{yaw:.1f}.npz"
            np.savez_compressed(
                packet_path, rgb=np.asarray(rgb), depth_m=depth,
                camera_origin_estimated_m=pose[0],
                camera_rotation_estimated=pose[1],
                actor_qpos=room.pose_data.qpos.copy(),
                estimated_pose_xyyaw=room.estimated.copy(),
                capture_time_s=room.time_s)
            views.append({"head_yaw_rad": yaw, "camera": camera,
                          "actor_packet_path": str(packet_path.relative_to(ROOT)),
                          "actor_packet_sha256": sha(packet_path),
                          "actor_choice": choice,
                          "evaluator_truth_box": truth,
                          "evaluator_accepted_presented_can": correct,
                          "evaluator_person_gap_m": round(gap_to_person(
                              room, robot_geoms)[0], 5)})
        return {"case": label, "seed": seed,
                "evaluator_variant": variant,
                "source_invitation_sha256": sha(invite_path),
                "source_episode_trace_sha256": sha(trace_path),
                "views": views,
                "evaluator_any_lower_accept": any(
                    v["evaluator_accepted_presented_can"] and
                    v["camera"] == "lower_rgb" for v in views),
                "evaluator_upper_accept": any(
                    v["evaluator_accepted_presented_can"] and
                    v["camera"] == "upper_rgb" for v in views)}
    finally:
        room.close()


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    vision = Vision("generic")
    exemplar = np.load(V149)
    owner_hist = np.asarray(json.loads(Path(V150).read_text())[
        "owner_color"]["hue_histogram"])
    rows = [run_case(seed, variant, vision, exemplar, owner_hist)
            for seed in SEEDS for variant in variants.POSITIVE_VARIANTS]
    report = {
        "scope": "Frozen V367 no-contact show challenge from the selected Cove-50 guarded blocked-route pose. Two local seeds x four V346 shifted/clothed/lit articulated owners hold the can at the original non-reaching hand. Lower RGB-D head yaw .4/.8/1.2 and upper camera at .8 use the existing DINO/hue taught-object pipeline; true item box and person gap score afterward. Direct-qpos head and static person, no audio comprehension/owner motion, real optics or sensor purchase conclusion.",
        "source_sha256": sha(SOURCE),
        "selected_visual_sha256": sha(selected_visual()),
        "v346_variant_source_sha256": sha(Path(variants.__file__)),
        "rows": rows,
        "score": {"lower_any": sum(r["evaluator_any_lower_accept"]
                                    for r in rows),
                  "upper": sum(r["evaluator_upper_accept"] for r in rows),
                  "total": len(rows)}}
    (OUT / "distant-show-challenge.json").write_text(
        json.dumps(report, indent=2) + "\n")
    (OUT / SOURCE.name).write_bytes(SOURCE.read_bytes())
    print(report["score"], flush=True)
    for row in rows:
        print(row["case"], [(v["head_yaw_rad"], v["camera"])
              for v in row["views"] if v["evaluator_accepted_presented_can"]],
              flush=True)


if __name__ == "__main__":
    main()
