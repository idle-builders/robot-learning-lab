"""V354: challenge a frozen early offer warning across shifted people.

The 34 cm noisy-long-depth threshold was chosen from V353 base seed170 at
t16.3; this run applies it unchanged to shifted pose/clothes/light and near/
far paths. Actor uses RGB-D, arm proposal, own pose/joints. Future clearance
is evaluator-only. No owner stop or human dynamics are simulated here.
"""

from __future__ import annotations

import json
from pathlib import Path

import mujoco
import numpy as np
import torch
from transformers import AutoModelForZeroShotObjectDetection, AutoProcessor

import challenge_person_belief_v346 as variants
import direct_structural_rooms_v270 as structural
from active_scan import DETECTOR
from articulated_person_scene import (FAR_HAND_XYZ_M, NEAR_HAND_XYZ_M,
                                      START_HAND_XYZ_M, add_person,
                                      set_person_pose)
from audit_articulated_owner_v342 import gap_to_person
from audit_direct_localization_v297 import depth_variant
from audit_person_grounding_v344 import detect
from direct_structural_rooms_v270 import selected_visual
from replay_endpoint_depth_gap_v350 import box_depth_points, score_points
from robot_geometry import RobotGeometry
from run import ROOT, sha
from run_late_corridor_intrusion_v323 import LateCorridorRoom
from run_shin_intrusion_posture_v324 import actual_pose, estimated_pose
from self_depth_mask import SelfDepthMask


SOURCE = Path(__file__)
OUT = ROOT / "artifacts/ape-closed-loop-cove50-v354"
V323 = ROOT / "artifacts/ape-closed-loop-cove50-v323"
SEEDS = (170, 172)
REACHES = {"near": NEAR_HAND_XYZ_M, "far": FAR_HAND_XYZ_M}
LONG_EARLY_WARNING_M = .34
SAMPLE_TIME_S = 16.3


def run_case(seed: int, variant: dict, reach: str,
             processor, detector, device: str) -> dict:
    name = f"seed{seed}-{variant['name']}-{reach}"
    branch = OUT / name
    branch.mkdir(parents=True, exist_ok=True)
    trace = json.loads((V323 / f"seed{seed}-present" /
                        f"seed{seed}-present/episode-trace.json").read_text())
    original_model = structural.structural_model
    structural.structural_model = (
        lambda kind, side: add_person(
            original_model(kind, side),
            shirt_rgba=variant["shirt_rgba"]))
    try:
        room = LateCorridorRoom(case=tuple(trace["case"]),
                                pose_source="command", branch=branch)
    finally:
        structural.structural_model = original_model
    mask = None
    try:
        room.layout(seed, absent=False, furnished=True)
        room.actual[:] = actual_pose(trace)
        room.estimated[:] = estimated_pose(trace)
        room.head = .8
        room.data.qpos[room.model.joint(
            "left_shoulder_pitch").qposadr[0]] = -.8
        room.model.light_diffuse[:] *= variant["light_scale"]
        offset = np.asarray(variant["offset_xyz_m"])
        room.model.body_pos[room.model.body("loop_person").id] = offset
        robot = RobotGeometry(room.model)
        mask = SelfDepthMask(room.model, camera="lower_rgb",
                             width=640, height=480, agreement_m=.04)
        minimum_human = float("inf")
        samples = []
        for step in range(21):
            fraction = step / 20
            room.time_s = 16.2 + .4 * fraction
            room._update_moving_obstacle()
            hand = START_HAND_XYZ_M + fraction * (
                REACHES[reach] - START_HAND_XYZ_M)
            set_person_pose(room.model, hand)
            room.model.body_pos[room.model.body(
                "loop_TomatoSauce").id] += offset
            room._forward()
            human = gap_to_person(room, robot.geom_ids)[0]
            minimum_human = min(minimum_human, human)
            if step != 5:
                continue
            lower, ideal = room.capture("lower_rgb", depth=True)
            detections = detect(processor, detector, device, lower,
                                "hand . arm .")
            short = depth_variant(ideal, "short", seed, 1700)
            long = depth_variant(ideal, "noisy_long", seed, 1700)
            camera_pose = room.camera_pose_estimate("lower_rgb")
            actor_qpos = room.pose_data.qpos.copy()
            robot.set_state(actor_qpos)
            box = (max(detections, key=lambda d: d["score"])["box_xyxy"]
                   if detections else None)
            fovy = float(room.model.cam_fovy[
                room.model.camera("lower_rgb").id])
            scored = {}
            for mode, measured in (("short", short), ("noisy_long", long)):
                filtered, info = mask.apply(
                    measured, actor_qpos, max_depth_m=2.5)
                points = box_depth_points(filtered, box, camera_pose, fovy)
                scored[mode] = {**score_points(points, robot),
                                "self_mask": info}
            packet = branch / "actor-early.npz"
            np.savez_compressed(
                packet, lower_rgb=np.asarray(lower),
                lower_short_depth_m=short,
                lower_noisy_long_depth_m=long,
                camera_origin_estimated_m=camera_pose[0],
                camera_rotation_estimated=camera_pose[1],
                estimated_pose_xyyaw=room.estimated.copy(),
                actor_qpos=actor_qpos, capture_time_s=room.time_s)
            samples.append({
                "actor_packet_path": str(packet.relative_to(ROOT)),
                "actor_packet_sha256": sha(packet),
                "actor_arm_detections": detections,
                "actor_depth_body_gap": scored,
                "evaluator_person_gap_at_sample_m": round(human, 5)})
        early = samples[0]
        gap = early["actor_depth_body_gap"]["noisy_long"]["min_body_gap_m"]
        warning = bool(gap is not None and gap < LONG_EARLY_WARNING_M)
        return {"case": name, "seed": seed, "reach": reach,
                "evaluator_variant": variant,
                "actor_sample_time_s": SAMPLE_TIME_S,
                **early,
                "actor_early_warning": warning,
                "actor_early_unknown": gap is None,
                "evaluator_minimum_person_gap_m": round(minimum_human, 5),
                "evaluator_unsafe_under_5cm": minimum_human < .05}
    finally:
        if mask is not None:
            mask.close()
        room.close()


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    device = "mps" if torch.backends.mps.is_available() else "cpu"
    processor = AutoProcessor.from_pretrained(DETECTOR)
    detector = AutoModelForZeroShotObjectDetection.from_pretrained(
        DETECTOR).to(device).eval()
    rows = [run_case(seed, variant, reach, processor, detector, device)
            for seed in SEEDS for variant in variants.POSITIVE_VARIANTS
            for reach in REACHES]
    report = {
        "scope": "Frozen V353 seed170 base early 34cm long-depth own-body warning at t16.3s, challenged on two procedural furnished seeds, four shifted/clothed/lit articulated people and near/far scripted paths. Actor receives only DINO arm box, hypothetical short/noisy-long lower depth, own calibrated model/joints and estimated pose; future person gap scores after decision. This is an early-warning specificity screen, not a commanded owner/robot stop or qualified hardware sensor.",
        "source_sha256": sha(SOURCE),
        "v346_variant_source_sha256": sha(Path(variants.__file__)),
        "selected_visual_sha256": sha(selected_visual()),
        "detector_weights_sha256": sha(DETECTOR / "model.safetensors"),
        "early_warning_threshold_m": LONG_EARLY_WARNING_M,
        "rows": rows,
        "score": {
            "unsafe_warned": sum(r["evaluator_unsafe_under_5cm"] and
                                  r["actor_early_warning"] for r in rows),
            "unsafe_missed": sum(r["evaluator_unsafe_under_5cm"] and
                                  not r["actor_early_warning"] for r in rows),
            "safe_warned": sum(not r["evaluator_unsafe_under_5cm"] and
                                r["actor_early_warning"] for r in rows),
            "safe_quiet": sum(not r["evaluator_unsafe_under_5cm"] and
                               not r["actor_early_warning"] for r in rows),
        },
    }
    (OUT / "early-offer-challenge.json").write_text(
        json.dumps(report, indent=2) + "\n")
    (OUT / SOURCE.name).write_bytes(SOURCE.read_bytes())
    print(report["score"], flush=True)
    for row in rows:
        print(row["case"], row["actor_depth_body_gap"]["noisy_long"][
            "min_body_gap_m"], "unsafe", row["evaluator_unsafe_under_5cm"],
              "warn", row["actor_early_warning"], flush=True)


if __name__ == "__main__":
    main()
