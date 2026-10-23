"""V351: time-resolved owner reach with a bounded hypothetical head look.

Compare actor short/long RGB-D body-gap warnings at 10 Hz while a scripted
head moves from +1.2 to +0.8 rad over 0.2 s. The human continues regardless
of warnings; this screen measures available lead time, not a safe stop.
"""

from __future__ import annotations

import json
from pathlib import Path

import imageio.v2 as imageio
import numpy as np
import torch
from transformers import AutoModelForZeroShotObjectDetection, AutoProcessor

import direct_structural_rooms_v270 as structural
from active_scan import DETECTOR
from articulated_person_scene import (FAR_HAND_XYZ_M, NEAR_HAND_XYZ_M,
                                      START_HAND_XYZ_M, add_person,
                                      set_person_pose)
from audit_articulated_owner_v342 import gap_to_person
from audit_direct_localization_v297 import depth_variant
from audit_lateral_clutter_depth_v285 import project
from audit_person_grounding_v344 import detect, review_pair
from direct_structural_rooms_v270 import selected_visual
from person_belief import PersonBelief
from replay_endpoint_depth_gap_v350 import box_depth_points, score_points
from robot_geometry import RobotGeometry
from run import ROOT, sha
from run_late_corridor_intrusion_v323 import LateCorridorRoom
from run_shin_intrusion_posture_v324 import actual_pose, estimated_pose
from self_depth_mask import SelfDepthMask


SOURCE = Path(__file__)
OUT = ROOT / "artifacts/ape-closed-loop-cove50-v351"
V323 = ROOT / "artifacts/ape-closed-loop-cove50-v323"
SEEDS = (170, 172)
REACHES = {"near": NEAR_HAND_XYZ_M, "far": FAR_HAND_XYZ_M}
HEAD_SCHEDULES = ("swept", "prepositioned")
SHIRTS = {170: ".13 .38 .65 1", 172: ".70 .42 .13 1"}
BODY_WARNING_M = .10  # 5 cm development margin + 5 cm provisional stop budget.


def run_case(seed: int, reach: str, head_schedule: str,
             processor, model, device: str) -> dict:
    label = f"seed{seed}-{reach}-{head_schedule}"
    branch = OUT / label
    branch.mkdir(parents=True, exist_ok=True)
    trace = json.loads((V323 / f"seed{seed}-present" /
                        f"seed{seed}-present/episode-trace.json").read_text())
    original_model = structural.structural_model
    structural.structural_model = (
        lambda kind, side: add_person(original_model(kind, side),
                                      shirt_rgba=SHIRTS[seed]))
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
        room.data.qpos[room.model.joint(
            "left_shoulder_pitch").qposadr[0]] = -.8
        robot = RobotGeometry(room.model)
        mask = SelfDepthMask(room.model, camera="lower_rgb",
                             width=640, height=480, agreement_m=.04)
        fovy = float(room.model.cam_fovy[room.model.camera("lower_rgb").id])
        belief = PersonBelief()
        evaluator = []
        captures = []
        video = branch / "active-owner-look.mp4"
        with imageio.get_writer(video, fps=5, codec="libx264", quality=7) as writer:
            for step in range(21):
                fraction = step / 20
                room.time_s = 16.2 + .4 * fraction
                room.head = (.8 if head_schedule == "prepositioned" else
                             1.2 - .4 * min(step / 10, 1.))
                room._update_moving_obstacle()
                hand = START_HAND_XYZ_M + fraction * (
                    REACHES[reach] - START_HAND_XYZ_M)
                set_person_pose(room.model, hand)
                room._forward()
                actual_gap = gap_to_person(room, robot.geom_ids)[0]
                evaluator.append({"time_s": round(room.time_s, 3),
                                  "person_gap_m": round(actual_gap, 5)})
                if step % 5:
                    continue
                index = step // 5
                lower, ideal = room.capture("lower_rgb", depth=True)
                upper, _ = room.capture("upper_rgb", depth=False)
                lower_arm = detect(processor, model, device, lower,
                                   "hand . arm .")
                upper_person = detect(processor, model, device, upper,
                                      "person .")
                state = belief.ingest(room.time_s, upper_person, lower_arm)
                short = depth_variant(ideal, "short", seed, 1400 + index)
                long = depth_variant(ideal, "noisy_long", seed, 1400 + index)
                camera_pose = room.camera_pose_estimate("lower_rgb")
                actor_qpos = room.pose_data.qpos.copy()
                robot.set_state(actor_qpos)
                box = (max(lower_arm, key=lambda d: d["score"])["box_xyxy"]
                       if lower_arm else None)
                distance = {}
                for mode, measured in (("short", short), ("noisy_long", long)):
                    filtered, info = mask.apply(
                        measured, actor_qpos, max_depth_m=2.5)
                    points = box_depth_points(filtered, box, camera_pose, fovy)
                    distance[mode] = {**score_points(points, robot),
                                      "self_mask": info}
                packet = branch / f"actor-{index:02d}.npz"
                np.savez_compressed(
                    packet, lower_rgb=np.asarray(lower),
                    upper_rgb=np.asarray(upper),
                    lower_short_depth_m=short,
                    lower_noisy_long_depth_m=long,
                    camera_origin_estimated_m=camera_pose[0],
                    camera_rotation_estimated=camera_pose[1],
                    actor_qpos=actor_qpos,
                    estimated_pose_xyyaw=room.estimated.copy(),
                    capture_time_s=room.time_s)
                warnings = {mode: (result["min_body_gap_m"] is not None and
                                   result["min_body_gap_m"] < BODY_WARNING_M)
                            for mode, result in distance.items()}
                captures.append({
                    "time_s": round(room.time_s, 3),
                    "head_yaw_rad": round(room.head, 3),
                    "actor_packet_path": str(packet.relative_to(ROOT)),
                    "actor_packet_sha256": sha(packet),
                    "actor_upper_person_detections": upper_person,
                    "actor_lower_arm_detections": lower_arm,
                    "actor_person_belief": state,
                    "actor_depth_body_gap": distance,
                    "actor_warning_under_10cm": warnings,
                    "evaluator_person_gap_m": round(actual_gap, 5)})
                writer.append_data(review_pair(
                    lower, upper, lower_arm, upper_person,
                    f"V351 {label} t{room.time_s:.1f} yaw{room.head:+.1f} "
                    f"short:{int(warnings['short'])} long:{int(warnings['noisy_long'])}"))
        margin_time = next((e["time_s"] for e in evaluator
                            if e["person_gap_m"] < .05), None)
        first_warnings = {mode: next((c["time_s"] for c in captures
                                      if c["actor_warning_under_10cm"][mode]), None)
                          for mode in ("short", "noisy_long")}
        return {"case": label, "seed": seed, "reach": reach,
                "head_schedule": head_schedule,
                "actor_samples": captures,
                "evaluator_20ms_person_gap": evaluator,
                "evaluator_first_below_5cm_s": margin_time,
                "actor_first_10cm_warning_s": first_warnings,
                "video_path": str(video.relative_to(ROOT)),
                "video_sha256": sha(video)}
    finally:
        if mask is not None:
            mask.close()
        room.close()


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    device = "mps" if torch.backends.mps.is_available() else "cpu"
    processor = AutoProcessor.from_pretrained(DETECTOR)
    model = AutoModelForZeroShotObjectDetection.from_pretrained(
        DETECTOR).to(device).eval()
    rows = [run_case(seed, reach, schedule, processor, model, device)
            for seed in SEEDS for reach in REACHES
            for schedule in HEAD_SCHEDULES]
    report = {
        "scope": "Paired 10Hz actor RGB-D and frozen DINO/person belief over 0.4s articulated near/far owner reaches in two furnished selected-Cove scenes. One hypothetical head look shifts +1.2 to +0.8 rad over0.2s (2rad/s), while the other starts prepositioned at +0.8 rad. Actor point-to-body warning uses 10cm development threshold on top DINO arm box after own-model self masking; evaluator scores human gap every20ms. Human motion is not interrupted, head/joint control is kinematic, stop response unmodeled. This is a timing screen, not a safety controller.",
        "source_sha256": sha(SOURCE),
        "person_scene_source_sha256": sha(Path(add_person.__code__.co_filename)),
        "selected_visual_sha256": sha(selected_visual()),
        "detector_weights_sha256": sha(DETECTOR / "model.safetensors"),
        "actor_body_warning_m": BODY_WARNING_M,
        "rows": rows,
    }
    (OUT / "active-owner-look-audit.json").write_text(
        json.dumps(report, indent=2) + "\n")
    (OUT / SOURCE.name).write_bytes(SOURCE.read_bytes())
    for row in rows:
        print(row["case"], "first<5cm", row["evaluator_first_below_5cm_s"],
              "warnings", row["actor_first_10cm_warning_s"], flush=True)


if __name__ == "__main__":
    main()
