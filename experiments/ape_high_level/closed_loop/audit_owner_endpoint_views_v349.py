"""V349: selected-head viewpoint screen at a near/far articulated hand endpoint.

Fixed robot/person poses isolate camera coverage. The actor sees RGB-D and
frozen prompt-detector proposals; simulator masks/clearance score only after.
This does not establish that the head can turn safely during a live reach.
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
                                      PERSON_GEOMS, add_person,
                                      set_person_pose)
from audit_articulated_owner_v342 import gap_to_person
from audit_direct_localization_v297 import depth_variant
from audit_person_grounding_v344 import (detect, evaluator_boxes,
                                         review_pair, score)
from direct_structural_rooms_v270 import selected_visual
from robot_geometry import RobotGeometry
from run import ROOT, sha
from run_late_corridor_intrusion_v323 import LateCorridorRoom
from run_shin_intrusion_posture_v324 import actual_pose, estimated_pose


SOURCE = Path(__file__)
OUT = ROOT / "artifacts/ape-closed-loop-cove50-v349"
V323 = ROOT / "artifacts/ape-closed-loop-cove50-v323"
SEEDS = (170, 172)
REACHES = {"near": NEAR_HAND_XYZ_M, "far": FAR_HAND_XYZ_M}
HEAD_YAWS = (1.2, .8, .4, 0., -.4)
SHIRTS = {170: ".13 .38 .65 1", 172: ".70 .42 .13 1"}


def depth_in_box(depth: np.ndarray, box: list[float] | None) -> dict | None:
    if box is None:
        return None
    x0, y0, x1, y1 = box
    crop = depth[max(0, int(y0)):min(depth.shape[0], int(np.ceil(y1))),
                 max(0, int(x0)):min(depth.shape[1], int(np.ceil(x1)))]
    valid = crop[np.isfinite(crop) & (crop > 0)]
    return {"valid_pixels": int(len(valid)), "roi_pixels": int(crop.size),
            "depth_q20_m": round(float(np.quantile(valid, .2)), 4)
            if len(valid) else None}


def run_case(seed: int, reach: str, processor, model, device: str) -> dict:
    case = f"seed{seed}-{reach}"
    branch = OUT / case
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
    try:
        room.layout(seed, absent=False, furnished=True)
        room.actual[:] = actual_pose(trace)
        room.estimated[:] = estimated_pose(trace)
        room.data.qpos[room.model.joint(
            "left_shoulder_pitch").qposadr[0]] = -.8
        room.time_s = 16.6
        room._update_moving_obstacle()
        set_person_pose(room.model, REACHES[reach])
        robot_geoms = RobotGeometry(room.model).geom_ids
        results = []
        video = branch / "endpoint-view-sweep.mp4"
        with imageio.get_writer(video, fps=2, codec="libx264", quality=7) as writer:
            for index, yaw in enumerate(HEAD_YAWS):
                room.head = yaw
                room._forward()
                lower, ideal = room.capture("lower_rgb", depth=True)
                upper, _ = room.capture("upper_rgb", depth=False)
                lower_arm = detect(processor, model, device, lower,
                                   "hand . arm .")
                upper_person = detect(processor, model, device, upper,
                                      "person .")
                lower_truth = evaluator_boxes(room, "lower_rgb")
                upper_truth = evaluator_boxes(room, "upper_rgb")
                short = depth_variant(ideal, "short", seed, 1200 + index)
                long = depth_variant(ideal, "noisy_long", seed, 1200 + index)
                camera_position, camera_rotation = room.camera_pose_estimate(
                    "lower_rgb")
                packet = branch / f"actor-yaw{index:02d}.npz"
                np.savez_compressed(
                    packet, lower_rgb=np.asarray(lower),
                    upper_rgb=np.asarray(upper),
                    lower_short_depth_m=short,
                    lower_noisy_long_depth_m=long,
                    camera_origin_estimated_m=camera_position,
                    camera_rotation_estimated=camera_rotation,
                    estimated_pose_xyyaw=room.estimated.copy(),
                    actor_qpos=room.data.qpos.copy(),
                    capture_time_s=room.time_s)
                box = max(lower_arm, key=lambda d: d["score"])["box_xyxy"] \
                    if lower_arm else None
                row = {"head_yaw_rad": yaw,
                       "actor_packet_path": str(packet.relative_to(ROOT)),
                       "actor_packet_sha256": sha(packet),
                       "actor_lower_arm_detections": lower_arm,
                       "actor_upper_person_detections": upper_person,
                       "actor_short_depth_in_top_arm_box": depth_in_box(short, box),
                       "actor_long_depth_in_top_arm_box": depth_in_box(long, box),
                       "evaluator_arm_box_lower": lower_truth["hand_arm"],
                       "evaluator_person_box_upper": upper_truth["person"],
                       "evaluator_lower_arm_localized": score(
                           lower_arm, lower_truth["hand_arm"])[
                               "localized_iou_ge_0p25"],
                       "evaluator_upper_person_localized": score(
                           upper_person, upper_truth["person"])[
                               "localized_iou_ge_0p25"],
                       "evaluator_person_gap_m": round(
                           gap_to_person(room, robot_geoms)[0], 5)}
                results.append(row)
                writer.append_data(review_pair(
                    lower, upper, lower_arm, upper_person,
                    f"V349 {case} yaw {yaw:+.1f} | lower arm / upper person"))
        return {"case": case, "seed": seed, "reach": reach,
                "results": results,
                "video_path": str(video.relative_to(ROOT)),
                "video_sha256": sha(video)}
    finally:
        room.close()


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    device = "mps" if torch.backends.mps.is_available() else "cpu"
    processor = AutoProcessor.from_pretrained(DETECTOR)
    model = AutoModelForZeroShotObjectDetection.from_pretrained(
        DETECTOR).to(device).eval()
    rows = [run_case(seed, reach, processor, model, device)
            for seed in SEEDS for reach in REACHES]
    report = {"scope": "Selected Cove-50 head-yaw static-view screen at t16.6s in two procedural furnished seeds and near/far articulated owner endpoints. Actor sees retained lower/upper RGB, candidate short/noisy-long depth, estimated pose/joints and DINO proposals. Evaluator boxes and modeled person gap score only after capture. Head yaw is set kinematically at the endpoint, not an executed live skill or guarantee of mid-reach tracking.",
              "source_sha256": sha(SOURCE),
              "person_scene_source_sha256": sha(Path(add_person.__code__.co_filename)),
              "selected_visual_sha256": sha(selected_visual()),
              "detector_weights_sha256": sha(DETECTOR / "model.safetensors"),
              "rows": rows}
    (OUT / "endpoint-view-audit.json").write_text(
        json.dumps(report, indent=2) + "\n")
    (OUT / SOURCE.name).write_bytes(SOURCE.read_bytes())
    for row in rows:
        print(row["case"], [(r["head_yaw_rad"],
                            r["evaluator_lower_arm_localized"],
                            r["evaluator_upper_person_localized"])
                           for r in row["results"]], flush=True)


if __name__ == "__main__":
    main()
