"""V364: challenge the fresh RGB-D item-position offer-lane check.

The V363 10cm horizontal tolerance is frozen before shifted/clothed/lit
person endpoint renders. Actor sees only the usual candidate image/depth,
estimated pose and requested far item point. True item placement scores it.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

import challenge_person_belief_v346 as variants
import direct_structural_rooms_v270 as structural
from articulated_person_scene import (FAR_HAND_XYZ_M, ITEM_SIDE_OFFSET_M,
                                      NEAR_HAND_XYZ_M, add_person,
                                      set_person_pose)
from audit_direct_structural_entry_v271 import V149, V150
from direct_structural_rooms_v270 import selected_visual
from run import ROOT, Vision, iou, sha
from run_color_grounded_entry_v151 import infer
from run_late_corridor_intrusion_v323 import LateCorridorRoom
from run_shin_intrusion_posture_v324 import actual_pose


SOURCE = Path(__file__)
OUT = ROOT / "artifacts/ape-closed-loop-cove50-v364"
V360 = ROOT / "artifacts/ape-closed-loop-cove50-v360"
SEEDS = (170, 172)
REACHES = {"near": NEAR_HAND_XYZ_M, "far": FAR_HAND_XYZ_M}
FAR_ITEM_XY_M = (FAR_HAND_XYZ_M + ITEM_SIDE_OFFSET_M)[:2]
FROZEN_TOLERANCE_M = .10


def run_case(seed: int, variant: dict, reach: str, vision: Vision,
             exemplar: np.ndarray, owner_hist: np.ndarray) -> dict:
    label = f"seed{seed}-{variant['name']}-{reach}"
    branch = OUT / label
    branch.mkdir(parents=True, exist_ok=True)
    prior = (V360 / "compliant_far" / f"seed{seed}-present" /
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
        room.head = 0.
        room.time_s = 17.0
        room._update_moving_obstacle()
        room.model.light_diffuse[:] *= variant["light_scale"]
        offset = np.asarray(variant["offset_xyz_m"], dtype=float)
        room.model.body_pos[room.model.body("loop_person").id] = offset
        set_person_pose(room.model, REACHES[reach])
        room.model.body_pos[room.model.body(
            "loop_TomatoSauce").id] += offset
        room._forward()
        rgb, depth = room.capture("lower_rgb", depth=True)
        depth[(depth < .08) | (depth > 1.5)] = np.nan
        camera_pose = room.camera_pose_estimate("lower_rgb")
        choice = infer(vision, exemplar, owner_hist, rgb, depth,
                       camera_pose, None, None)
        actor_xyz = choice["accepted_world_xyz_m"]
        actor_error = (None if actor_xyz is None else float(np.linalg.norm(
            np.asarray(actor_xyz[:2]) - FAR_ITEM_XY_M)))
        actor_compatible = bool(actor_error is not None and
                                actor_error <= FROZEN_TOLERANCE_M)
        truth = room.truth_box("TomatoSauce", "lower_rgb")
        correct = (choice["accepted_box"] is not None and truth is not None
                   and iou(choice["accepted_box"], truth) >= .25)
        true_item_xy = room.model.body_pos[
            room.model.body("loop_TomatoSauce").id, :2].copy()
        true_error = float(np.linalg.norm(true_item_xy - FAR_ITEM_XY_M))
        packet_path = branch / "actor-offer-lane.npz"
        np.savez_compressed(
            packet_path, lower_rgb=np.asarray(rgb), lower_depth_m=depth,
            camera_origin_estimated_m=camera_pose[0],
            camera_rotation_estimated=camera_pose[1],
            actor_qpos=room.pose_data.qpos.copy(),
            estimated_pose_xyyaw=room.estimated.copy(),
            capture_time_s=room.time_s)
        return {"case": label, "seed": seed, "evaluator_variant": variant,
                "evaluator_scripted_reach": reach,
                "source_invitation_sha256": sha(invite_path),
                "source_episode_trace_sha256": sha(trace_path),
                "actor_packet_path": str(packet_path.relative_to(ROOT)),
                "actor_packet_sha256": sha(packet_path),
                "actor_choice": choice,
                "actor_requested_item_xy_m": FAR_ITEM_XY_M.tolist(),
                "actor_zone_error_m": None if actor_error is None else round(
                    actor_error, 5),
                "actor_zone_compatible": actor_compatible,
                "evaluator_accepted_box_is_presented_item": correct,
                "evaluator_true_item_xy_m": true_item_xy.round(5).tolist(),
                "evaluator_true_zone_error_m": round(true_error, 5),
                "evaluator_within_requested_zone":
                    true_error <= FROZEN_TOLERANCE_M}
    finally:
        room.close()


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    vision = Vision("generic")
    exemplar = np.load(V149)
    owner_hist = np.asarray(json.loads(Path(V150).read_text())[
        "owner_color"]["hue_histogram"])
    rows = [run_case(seed, variant, reach, vision, exemplar, owner_hist)
            for seed in SEEDS for variant in variants.POSITIVE_VARIANTS
            for reach in REACHES]
    report = {
        "scope": "Frozen V363 actor 10cm requested-far-item XY tolerance challenged on two local selected-body seeds and four V346 shifted/clothed/lit articulated-person endpoint variants, near/far scripted reach. Reuses V360 post-reposition qpos and estimated pose; renders lower RGB-D at head0 and the released DINO/hue taught-object pipeline. Actor lane decision uses only its accepted 3D item point; true can position and box overlap score afterward. Endpoint only: does not predict arm sweep, certify rear step, or prove human/robot stopping. Seeds share local geometry, person variants are procedural.",
        "source_sha256": sha(SOURCE),
        "selected_visual_sha256": sha(selected_visual()),
        "v346_variant_source_sha256": sha(Path(variants.__file__)),
        "frozen_tolerance_m": FROZEN_TOLERANCE_M,
        "rows": rows,
        "score": {
            "presented_item_boxes_correct": sum(
                r["evaluator_accepted_box_is_presented_item"] for r in rows),
            "actor_zone_matches_true_zone": sum(
                r["actor_zone_compatible"] ==
                r["evaluator_within_requested_zone"] for r in rows),
            "false_zone_compatible": sum(
                r["actor_zone_compatible"] and
                not r["evaluator_within_requested_zone"] for r in rows),
            "false_zone_reask": sum(
                not r["actor_zone_compatible"] and
                r["evaluator_within_requested_zone"] for r in rows),
        }}
    (OUT / "offer-lane-challenge.json").write_text(
        json.dumps(report, indent=2) + "\n")
    (OUT / SOURCE.name).write_bytes(SOURCE.read_bytes())
    print(report["score"], flush=True)
    for row in rows:
        print(row["case"], row["actor_zone_error_m"],
              row["evaluator_true_zone_error_m"],
              row["actor_zone_compatible"], flush=True)


if __name__ == "__main__":
    main()
