"""Teach a real captured furniture instance, turn away, re-find and approach it.

The owner input is a click on a photographed-in-place object. The actor uses
RGB detections/features, mesh-ray depth, two range heights and either ideal
command feedback or gated RGB-D motion, optionally with ideal turn feedback.
The mesh produces simulated depth and range; its navigation map is evaluator-
only for motion acceptance and scoring. This remains a static-object,
kinematic 50 cm proxy experiment.
"""

from __future__ import annotations

import argparse
import gzip
import json
import math
from pathlib import Path

import cv2
import imageio.v2 as imageio
import numpy as np
from PIL import Image
import torch
from transformers import (AutoImageProcessor, AutoModel,
                          AutoModelForZeroShotObjectDetection, AutoProcessor)
from huggingface_hub import hf_hub_download

from run_habitat_gs_probe import make_scene, observe, read_nav_raster
from run_mesh_probe import segment_free
from screen_habitat_gs import ASSETS, REPO, ROOT
from fan_view_route import FanViewRoute
from measure_real_scan_visual_odometry import feature_packet, visual_transform, angle_deg


MODEL = ROOT / ".cache/ape-high-level/models"
PROMPT = "chair . sofa . bed . table . bookshelf ."
TEACH = {
    14: {"click_xy": [50, 180], "category": "chair"},
    # Frozen from scene32's first published ape-height camera view before
    # evaluating its route: the far end table is visible beyond the bed.
    32: {"click_xy": [390, 180], "category": "table"},
}


class FurnitureVision:
    def __init__(self) -> None:
        self.processor = AutoProcessor.from_pretrained(MODEL / "grounding-dino-tiny")
        self.detector = AutoModelForZeroShotObjectDetection.from_pretrained(
            MODEL / "grounding-dino-tiny").eval()
        self.feature_processor = AutoImageProcessor.from_pretrained(
            MODEL / "dinov2-small")
        self.features = AutoModel.from_pretrained(MODEL / "dinov2-small").eval()

    def detect(self, rgb: np.ndarray) -> list[dict]:
        picture = Image.fromarray(rgb)
        packet = self.processor(images=picture, text=[PROMPT],
                                return_tensors="pt")
        with torch.no_grad():
            raw = self.detector(**packet)
        result = self.processor.post_process_grounded_object_detection(
            raw, packet.input_ids, threshold=.25, text_threshold=.25,
            target_sizes=[(picture.height, picture.width)])[0]
        rows = []
        for name, score, box in zip(result["text_labels"], result["scores"],
                                    result["boxes"]):
            xyxy = [int(round(float(v))) for v in box]
            if (xyxy[2] - xyxy[0]) * (xyxy[3] - xyxy[1]) < 250:
                continue
            rows.append({"category": name, "score": float(score),
                         "box_xyxy": xyxy})
        return rows

    def feature_many(self, rgb: np.ndarray,
                     boxes: list[list[int]]) -> np.ndarray:
        if not boxes:
            return np.empty((0, 384), dtype=np.float32)
        picture = Image.fromarray(rgb)
        crops = [picture.crop(tuple(box)) for box in boxes]
        pixels = self.feature_processor(images=crops,
                                        return_tensors="pt")["pixel_values"]
        with torch.no_grad():
            hidden = self.features(pixel_values=pixels).last_hidden_state[:, 0]
            unit = torch.nn.functional.normalize(hidden, p=2, dim=-1)
        return unit.numpy()

    def feature(self, rgb: np.ndarray, box: list[int]) -> np.ndarray:
        return self.feature_many(rgb, [box])[0]


def image_point(box: list[int], depth: np.ndarray, pose: np.ndarray,
                heading_deg: float) -> dict | None:
    u = int((box[0] + box[2]) / 2)
    v = int((box[1] + box[3]) / 2)
    if not (0 <= v < depth.shape[0] and 0 <= u < depth.shape[1]):
        return None
    distance = float(depth[v, u])
    if not .35 < distance < 5.:
        return None
    focal = depth.shape[0] / (2 * math.tan(math.radians(75) / 2))
    side_m = (u - (depth.shape[1] - 1) / 2) * distance / focal
    yaw = math.radians(heading_deg)
    forward = np.array([math.sin(yaw), math.cos(yaw)])
    right = np.array([math.cos(yaw), -math.sin(yaw)])
    point = pose + distance * forward + side_m * right
    return {"pixel_uv": [u, v], "depth_m": distance,
            "bearing_deg": math.degrees(math.atan2(side_m, distance)),
            "point_xy_m": point.tolist()}


def signed_turn(target_xy: np.ndarray, pose_xy: np.ndarray,
                heading_deg: float) -> float:
    vector = target_xy - pose_xy
    direction = math.degrees(math.atan2(vector[0], vector[1]))
    return (direction - heading_deg + 180) % 360 - 180


def visual_rgbd_delta(previous: dict, current: dict, k: np.ndarray,
                      matcher: cv2.BFMatcher) -> tuple[np.ndarray | None, dict]:
    """Camera motion in the preceding camera frame, using no route pose truth."""
    forward, a = visual_transform(previous, current, k, matcher)
    reverse, b = visual_transform(current, previous, k, matcher)
    evidence = {"forward": a, "reverse": b, "accepted": False}
    if forward is None or reverse is None:
        return None, evidence
    closure = reverse @ forward
    evidence["reciprocal_translation_m"] = float(np.linalg.norm(closure[:3, 3]))
    evidence["reciprocal_rotation_deg"] = angle_deg(closure[:3, :3])
    evidence["accepted"] = bool(
        a["inliers"] >= 30 and b["inliers"] >= 30 and
        a["inlier_ratio"] >= .25 and b["inlier_ratio"] >= .25 and
        a["median_reprojection_px"] <= 2.5 and
        b["median_reprojection_px"] <= 2.5 and
        evidence["reciprocal_translation_m"] <= .08 and
        evidence["reciprocal_rotation_deg"] <= 5.)
    if not evidence["accepted"]:
        return None, evidence
    motion = np.linalg.inv(forward)
    evidence["camera_right_m"] = float(motion[0, 3])
    evidence["camera_forward_m"] = float(motion[2, 3])
    evidence["camera_vertical_m"] = float(motion[1, 3])
    evidence["yaw_delta_deg"] = math.degrees(math.atan2(
        motion[0, 2], motion[2, 2]))
    return motion, evidence


def run(scene_number: int = 14, steps: int = 45,
        recover_blocked: bool = False, active_relook: bool = False,
        pose_mode: str = "command",
        no_progress_frames: int | None = None) -> dict:
    if pose_mode not in ("command", "visual_rgbd", "visual_rgbd_yaw_feedback"):
        raise ValueError(f"Unknown pose source: {pose_mode}")
    cv2.setRNGSeed(0)
    torch.manual_seed(0)
    if scene_number not in TEACH:
        raise ValueError(f"No owner click frozen for scene{scene_number:02d}")
    split = "train" if scene_number <= 55 else "val"
    scene_id = f"scene{scene_number:02d}"
    folder = ASSETS / split / scene_id
    episode_path = Path(hf_hub_download(
        repo_id=REPO, repo_type="dataset",
        filename=f"episodes/pointnav/{split}/content/{scene_id}.json.gz",
        local_dir=ASSETS))
    nav_path = Path(hf_hub_download(
        repo_id=REPO, repo_type="dataset",
        filename=f"{split}/{scene_id}/{scene_id}.navmesh", local_dir=ASSETS))
    episode = json.load(gzip.open(episode_path, "rt"))["episodes"][0]
    start = np.asarray(episode["start_position"], float)
    goal = np.asarray(episode["goals"][0]["position"], float)
    nav = read_nav_raster(nav_path, start[1], anchor_xz=start[[0, 2]])
    actual = start[[0, 2]].copy()
    actor = np.zeros(2)
    heading = math.atan2(start[0] - goal[0], goal[2] - start[2])
    actor_heading = 0.
    renderer, ray_scene, _ = make_scene(scene_number, None, start[1])
    vision = FurnitureVision()
    rgb, depth, _, _ = observe(renderer, ray_scene, actual, heading, start[1])
    click = TEACH[scene_number]["click_xy"]
    category = TEACH[scene_number]["category"]
    proposals = vision.detect(rgb)
    clicked = [row for row in proposals if category in row["category"].split()
               and row["box_xyxy"][0] <= click[0] <= row["box_xyxy"][2]
               and row["box_xyxy"][1] <= click[1] <= row["box_xyxy"][3]]
    if not clicked:
        raise ValueError("Owner click has no native furniture proposal")
    taught = max(clicked, key=lambda row: row["score"])
    taught_point = image_point(taught["box_xyxy"], depth, actor, actor_heading)
    if taught_point is None:
        raise ValueError("Owner-clicked object lacks measured depth")
    exemplar = vision.feature(rgb, taught["box_xyxy"])
    reference = np.asarray(taught_point["point_xy_m"])
    # An owner asks for the item after the ape has looked away. This turn is
    # commanded and integrated by the actor; no simulator pose is handed in.
    heading += math.radians(100.)
    actor_heading += 100.
    output_name = ("native_object_active_relook_probe" if active_relook else
                   "native_object_recovery_probe" if recover_blocked else
                   "native_object_probe")
    if pose_mode != "command":
        output_name = output_name.replace("_probe", f"_{pose_mode}_probe")
    if no_progress_frames is not None:
        if no_progress_frames <= 0:
            raise ValueError("no-progress bound must be positive")
        output_name = output_name.replace("_probe", "_guarded_probe")
    output = folder / output_name
    output.mkdir(exist_ok=True)
    cv2.circle(rgb, tuple(click), 7, (255, 255, 0), 2)
    cv2.rectangle(rgb, tuple(taught["box_xyxy"][:2]),
                  tuple(taught["box_xyxy"][2:]), (255, 255, 0), 2)
    Image.fromarray(rgb).save(output / "teach_click.jpg")
    writer = imageio.get_writer(str(output / "first_person.mp4"), fps=5,
                                codec="libx264", quality=7, macro_block_size=8)
    trace = []
    planner = FanViewRoute(proxy_radius_m=.17, required_free_planes=(0, 1))
    recent_target = None
    no_detection = 0
    previous_command = previous_measured = 0.
    rejected_motion_count = 0
    unsupported_route_streak = 0
    start_heading = heading - math.radians(100.)
    orb = cv2.ORB_create(nfeatures=2500) if pose_mode != "command" else None
    matcher = cv2.BFMatcher(cv2.NORM_HAMMING) if orb is not None else None
    height, width = rgb.shape[:2]
    focal = height / (2 * math.tan(math.radians(75.) / 2))
    k = np.array([[focal, 0, (width - 1) / 2],
                  [0, focal, (height - 1) / 2], [0, 0, 1]], dtype=np.float64)
    # observe() casts unnormalized pinhole rays with forward component one;
    # its t_hit is already optical-axis depth, which is what PnP expects.
    previous_packet = None
    last_action_turn_only = False
    last_accepted_step_frame = None
    try:
        for index in range(steps):
            rgb, depth, ranges, angles = observe(
                renderer, ray_scene, actual, heading, start[1])
            pose_evidence = None
            pose_available = True
            if orb is not None:
                packet = feature_packet(rgb, depth, orb)
                if previous_packet is not None and pose_mode == "visual_rgbd_yaw_feedback" and last_action_turn_only:
                    pose_evidence = {"source": "idealized_yaw_feedback",
                                     "accepted": True}
                elif previous_packet is not None:
                    motion, pose_evidence = visual_rgbd_delta(
                        previous_packet, packet, k, matcher)
                    pose_evidence["source"] = "rgbd_pnp"
                    if motion is None:
                        pose_available = False
                    else:
                        old_yaw = math.radians(actor_heading)
                        actor += motion[0, 3] * np.array([
                            math.cos(old_yaw), -math.sin(old_yaw)])
                        actor += motion[2, 3] * np.array([
                            math.sin(old_yaw), math.cos(old_yaw)])
                        actor_heading += pose_evidence["yaw_delta_deg"]
                previous_packet = packet
            truth_at_capture = actual.copy()
            truth_heading_at_capture = heading
            actor_at_capture = actor.copy()
            actor_heading_at_capture = actor_heading
            displacement = truth_at_capture - start[[0, 2]]
            truth_local = np.array([
                -displacement[0] * math.cos(start_heading) -
                displacement[1] * math.sin(start_heading),
                -displacement[0] * math.sin(start_heading) +
                displacement[1] * math.cos(start_heading)])
            pose_error_m = float(np.linalg.norm(actor_at_capture - truth_local))
            yaw_error_deg = abs((actor_heading_at_capture -
                math.degrees(truth_heading_at_capture - start_heading) + 180) % 360 - 180)
            proposals = vision.detect(rgb)
            scored = []
            for proposal in proposals if pose_available else []:
                if category in proposal["category"].split():
                    point = image_point(proposal["box_xyxy"], depth,
                                        actor, actor_heading)
                    if point is not None:
                        scored.append((proposal, point))
            features = vision.feature_many(
                rgb, [row["box_xyxy"] for row, _ in scored])
            candidates = []
            for (proposal, point), feature in zip(scored, features):
                similarity = float(exemplar @ feature)
                memory_error = float(np.linalg.norm(
                    np.asarray(point["point_xy_m"]) - reference))
                if similarity >= .55 and memory_error <= .65:
                    candidates.append({**proposal, **point,
                                       "appearance_similarity": similarity,
                                       "memory_error_m": memory_error})
            chosen = (max(candidates, key=lambda row: row["appearance_similarity"] -
                          .5 * row["memory_error_m"]) if candidates else None)
            if chosen is not None:
                recent_target = np.asarray(chosen["point_xy_m"])
                no_detection = 0
            else:
                no_detection += 1
            blocked_feedback = None
            if not pose_available:
                decision = {"intent": "hold_request_help",
                            "reason": "visual_pose_unavailable"}
            elif previous_command > .02 and previous_measured < .5 * previous_command:
                rejected_motion_count += 1
                if recover_blocked:
                    blocked_feedback = planner.observe_rejected_step(
                        actor, actor_heading, previous_command, index * .2)
                if recover_blocked and rejected_motion_count < 3:
                    decision = {"intent": "hold_reobserve",
                                "reason": "rejected_step_update_route"}
                else:
                    decision = {"intent": "hold_request_help",
                                "reason": "repeated_blocked_motion" if recover_blocked
                                else "blocked_motion"}
            elif recent_target is None:
                error = signed_turn(reference, actor, actor_heading)
                decision = {"intent": "turn_to_taught_view",
                            "yaw_delta_deg": float(np.clip(error if abs(error) > 8 else 15., -20., 20.)),
                            "reason": "native_object_not_reacquired"}
            elif active_relook and no_detection >= 2:
                if no_detection > 7:
                    decision = {"intent": "hold_request_help",
                                "reason": "taught_object_not_reacquired_after_view_sweep"}
                else:
                    # Rotate around the last accepted RGB-D object bearing.
                    # No translation is licensed while its image match is stale.
                    offset = {2: 0., 3: 15., 4: -15., 5: 30., 6: -30., 7: 0.}[no_detection]
                    yaw = signed_turn(recent_target, actor, actor_heading) + offset
                    decision = {"intent": "turn_reacquire_taught_object",
                                "yaw_delta_deg": float(np.clip(yaw, -20., 20.)),
                                "view_offset_deg": offset,
                                "reason": "bounded_active_relook_after_missing_match"}
            elif no_detection > 5:
                decision = {"intent": "hold_request_help",
                            "reason": "stale_native_object_observation"}
            else:
                target_distance = float(np.linalg.norm(recent_target - actor))
                intermediate = target_distance > 2.2
                local_target = (actor + (recent_target - actor) *
                                (1.8 / target_distance)) if intermediate else recent_target
                decision = planner.plan(ranges, angles, actor, actor_heading,
                                        local_target, index * .2)
                if intermediate and decision["intent"] == "hold_viewpoint":
                    decision["intent"] = "reobserve_at_intermediate_goal"
                if decision["intent"] == "hold_reobserve" and no_detection >= 5:
                    decision = {"intent": "hold_request_help",
                                "reason": "no_supported_furniture_route"}
            if recover_blocked and decision["intent"] == "hold_reobserve":
                unsupported_route_streak += 1
                if unsupported_route_streak >= 5:
                    decision = {"intent": "hold_request_help",
                                "reason": "route_still_unsupported_after_reobservation"}
            else:
                unsupported_route_streak = 0
            if (no_progress_frames is not None and
                    last_accepted_step_frame is not None and
                    index - last_accepted_step_frame >= no_progress_frames and
                    decision["intent"] not in ("hold_viewpoint", "hold_request_help")):
                decision = {"intent": "hold_request_help",
                            "reason": "no_accepted_route_progress"}
            if (no_progress_frames is not None and index == steps - 1 and
                    decision["intent"] not in ("hold_viewpoint", "hold_request_help")):
                decision = {"intent": "hold_request_help",
                            "reason": "episode_action_budget_exhausted"}
            command_yaw = float(decision.get("yaw_delta_deg", 0.))
            command_step = float(decision.get("forward_step_m", 0.))
            accepted = True
            if command_step:
                candidate = actual + command_step * np.array([
                    -math.sin(heading), math.cos(heading)])
                accepted = segment_free(nav, actual, candidate)
                if accepted:
                    actual = candidate
                    last_accepted_step_frame = index
                    if pose_mode == "command":
                        actor += command_step * np.array([
                            math.sin(math.radians(actor_heading)),
                            math.cos(math.radians(actor_heading))])
            previous_command = command_step
            previous_measured = command_step if accepted else 0.
            heading += math.radians(command_yaw)
            if pose_mode == "command" or (pose_mode == "visual_rgbd_yaw_feedback" and
                                          command_step == 0.):
                actor_heading += command_yaw
            last_action_turn_only = bool(command_yaw and not command_step)
            frame = rgb.copy()
            if chosen is not None:
                box = chosen["box_xyxy"]
                cv2.rectangle(frame, tuple(box[:2]), tuple(box[2:]),
                              (70, 255, 70), 2)
            cv2.putText(frame, f"{index:02d} {decision['intent']} | {category}: {'seen' if chosen else 'uncertain'}",
                        (8, 24), cv2.FONT_HERSHEY_SIMPLEX, .5,
                        (255, 255, 255), 2)
            writer.append_data(frame)
            trace.append({"frame": index, "time_s": index * .2,
                          "raw_proposal_count": len(proposals),
                          "accepted_candidate_count": len(candidates),
                          "frames_since_accepted_match": no_detection,
                          "chosen": chosen, "decision": decision,
                          "range_low_m": ranges[0].tolist(),
                          "range_mid_m": ranges[1].tolist(),
                          "angles_deg": angles.tolist(),
                          "valid_depth_fraction": float(np.mean(np.isfinite(depth))),
                          "command_accepted": accepted,
                          "blocked_feedback": blocked_feedback,
                          "rejected_motion_count": rejected_motion_count,
                          "unsupported_route_streak": unsupported_route_streak,
                          "frames_since_accepted_step": (index - last_accepted_step_frame
                              if last_accepted_step_frame is not None else None),
                          "pose_source": pose_mode,
                          "pose_available_at_capture": pose_available,
                          "pose_evidence": pose_evidence,
                          "actor_pose_at_capture_xy_m": actor_at_capture.tolist(),
                          "actor_heading_at_capture_deg": actor_heading_at_capture,
                          "evaluator_pose_error_at_capture_m": pose_error_m,
                          "evaluator_yaw_error_at_capture_deg": yaw_error_deg,
                          "actor_odometry_xy_m": actor.tolist(),
                          "truth_xz_m": actual.tolist(),
                          "truth_heading_rad": heading})
            if decision["intent"] in ("hold_viewpoint", "hold_request_help"):
                break
    finally:
        writer.close()
    Image.fromarray(frame).save(output / "last_frame.jpg")
    # The reference surface is inferred from the owner-clicked first RGB-D
    # packet; there is no independent instance mesh label in Habitat-GS.
    true_reference_xz = start[[0, 2]] + np.array([
        -(reference[0] * math.cos(math.atan2(start[0] - goal[0], goal[2] - start[2])) +
          reference[1] * math.sin(math.atan2(start[0] - goal[0], goal[2] - start[2]))),
        -reference[0] * math.sin(math.atan2(start[0] - goal[0], goal[2] - start[2])) +
          reference[1] * math.cos(math.atan2(start[0] - goal[0], goal[2] - start[2]))])
    final_intent = trace[-1]["decision"]["intent"]
    final_distance = float(np.linalg.norm(actual - true_reference_xz))
    report = {"schema": "ape.habitat-gs-native-furniture-probe.v1",
              "scene_id": scene_id, "source": f"https://huggingface.co/datasets/{REPO}",
              "owner_click_xy": click, "taught_category": category,
              "teach_proposal": taught, "teach_depth_point": taught_point,
              "published_start_episode_id": episode["episode_id"],
              "opencv_and_torch_rng_seed": 0,
              "frames": len(trace),
              "pose_source": pose_mode,
              "no_progress_frame_bound": no_progress_frames,
              "pose_gate": "bidirectional ORB RGB-D PnP: >=30 inliers, >=0.25 ratio, <=2.5px median reprojection, <=0.08m/5deg reciprocal closure" if orb is not None else None,
              "pose_rejected_frames": sum(not row["pose_available_at_capture"] for row in trace),
              "max_evaluator_pose_error_m": max(row["evaluator_pose_error_at_capture_m"] for row in trace),
              "recovery_enabled": recover_blocked,
              "active_relook_enabled": active_relook,
              "rejected_motion_count": rejected_motion_count,
              "outcome": final_intent if final_intent in (
                  "hold_viewpoint", "hold_request_help") else "time_limit",
              "last_intent": final_intent,
              "reference_standoff_within_band": bool(
                  final_intent == "hold_viewpoint" and 1.20 <= final_distance <= 1.35),
              "final_reference_surface_distance_m": final_distance,
              "first_reacquire_frame": next((row["frame"] for row in trace
                  if row["chosen"] is not None), None),
              "accepted_motion_m": float(sum(
                  row["decision"].get("forward_step_m", 0.) for row in trace
                  if row["command_accepted"])),
              "limitations": ["Static native furniture, not a moved small object",
                              "Scripted owner click and look-away; no rendered person",
                              "The clicked RGB-D point is a reference, not independent instance ground truth",
                              "Kinematic 17 cm proxy and idealized command feedback",
                              "Scripted initial 100deg look-away uses commanded yaw, not a visual estimate",
                              "Yaw-feedback branch uses ideal commanded turn, not simulated IMU noise or calibrated joint sensing" if pose_mode == "visual_rgbd_yaw_feedback" else "No independent yaw sensor",
                              "Mesh depth is raycast from a separately supplied scan, not a measured depth camera",
                              "Model inference delay is not advanced on the simulated clock; no concurrent hazard monitor"],
              "trace": trace}
    (output / "trace.json").write_text(json.dumps(report, indent=2) + "\n")
    return {key: value for key, value in report.items() if key != "trace"}


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--scene-number", type=int, default=14)
    parser.add_argument("--steps", type=int, default=45)
    parser.add_argument("--recover-blocked", action="store_true")
    parser.add_argument("--active-relook", action="store_true")
    parser.add_argument("--pose-mode", choices=("command", "visual_rgbd",
                                                 "visual_rgbd_yaw_feedback"),
                        default="command")
    parser.add_argument("--no-progress-frames", type=int)
    args = parser.parse_args()
    print(json.dumps(run(args.scene_number, args.steps, args.recover_blocked,
                         args.active_relook, args.pose_mode,
                         args.no_progress_frames), indent=2), flush=True)
