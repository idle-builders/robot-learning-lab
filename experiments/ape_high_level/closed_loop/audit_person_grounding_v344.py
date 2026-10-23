"""V344: released prompt detector on actor RGB from an articulated owner.

Grounding DINO sees only upper/lower actor images. Re-rendered simulator masks
label person/arm localization after inference and verify packet alignment.
"""

from __future__ import annotations

import json
from pathlib import Path

import imageio.v2 as imageio
import mujoco
import numpy as np
from PIL import Image, ImageDraw
import torch
from transformers import (AutoModelForZeroShotObjectDetection,
                          AutoProcessor)

import direct_structural_rooms_v270 as structural
from active_scan import DETECTOR
from articulated_person_scene import (FAR_HAND_XYZ_M, NEAR_HAND_XYZ_M,
                                      PERSON_GEOMS, START_HAND_XYZ_M,
                                      add_person, set_person_pose)
from direct_structural_rooms_v270 import selected_visual
from run import ROOT, sha
from run_late_corridor_intrusion_v323 import LateCorridorRoom
from run_shin_intrusion_posture_v324 import actual_pose, estimated_pose


SOURCE = Path(__file__)
OUT = ROOT / "artifacts/ape-closed-loop-cove50-v344"
V323 = ROOT / "artifacts/ape-closed-loop-cove50-v323"
V342 = ROOT / "artifacts/ape-closed-loop-cove50-v342"
SEEDS = (170, 172)
REACHES = {"near": NEAR_HAND_XYZ_M, "far": FAR_HAND_XYZ_M}
SHIRTS = {170: ".13 .38 .65 1", 172: ".70 .42 .13 1"}
SAMPLES = (0, 2, 4)
PROMPTS = ("person .", "hand . arm .")
CAMERAS = ("lower_rgb", "upper_rgb")


def box_of(mask: np.ndarray) -> list[int] | None:
    ys, xs = np.nonzero(mask)
    return ([int(xs.min()), int(ys.min()), int(xs.max() + 1),
             int(ys.max() + 1)] if len(xs) else None)


def iou(a: list[float] | None, b: list[int] | None) -> float:
    if a is None or b is None:
        return 0.
    overlap = (max(0., min(a[2], b[2]) - max(a[0], b[0])) *
               max(0., min(a[3], b[3]) - max(a[1], b[1])))
    area_a = max(0., a[2] - a[0]) * max(0., a[3] - a[1])
    area_b = max(0., b[2] - b[0]) * max(0., b[3] - b[1])
    return float(overlap / max(1., area_a + area_b - overlap))


def evaluator_boxes(room, camera: str) -> dict:
    room.renderer.enable_segmentation_rendering()
    room.renderer.update_scene(room.data, camera=camera)
    segments = room.renderer.render().copy()
    room.renderer.disable_segmentation_rendering()
    is_geom = segments[:, :, 1] == int(mujoco.mjtObj.mjOBJ_GEOM)
    person_ids = [room.model.geom(name).id for name in PERSON_GEOMS]
    arm_ids = [room.model.geom(name).id for name in PERSON_GEOMS[2:]]
    return {"person": box_of(is_geom & np.isin(segments[:, :, 0], person_ids)),
            "hand_arm": box_of(is_geom & np.isin(segments[:, :, 0], arm_ids))}


def detect(processor, model, device: str,
           rgb: Image.Image, prompt: str) -> list[dict]:
    tensors = processor(images=rgb, text=[prompt],
                        return_tensors="pt").to(device)
    with torch.no_grad():
        output = model(**tensors)
    found = processor.post_process_grounded_object_detection(
        output, tensors.input_ids, threshold=.25, text_threshold=.25,
        target_sizes=[rgb.size[::-1]])[0]
    return [{"label": str(label), "score": round(float(score), 4),
             "box_xyxy": [round(float(value), 2) for value in box]}
            for label, score, box in zip(found["text_labels"],
                                         found["scores"], found["boxes"])]


def score(detections: list[dict], truth: list[int] | None) -> dict:
    overlaps = [iou(item["box_xyxy"], truth) for item in detections]
    return {"detections": len(detections),
            "best_iou": round(max(overlaps, default=0.), 4),
            "localized_iou_ge_0p25": any(value >= .25 for value in overlaps),
            "off_target_boxes_iou_lt_0p1": sum(value < .1 for value in overlaps),
            "top_score_box": max(detections, key=lambda x: x["score"])
            if detections else None}


def review_pair(lower: Image.Image, upper: Image.Image,
                lower_detections: list[dict], upper_detections: list[dict],
                title: str) -> np.ndarray:
    image = Image.new("RGB", (1280, 510), (240, 240, 238))
    image.paste(lower, (0, 30))
    image.paste(upper, (640, 30))
    draw = ImageDraw.Draw(image)
    draw.text((10, 7), title, fill=(20, 20, 20))
    for offset, detections in ((0, lower_detections), (640, upper_detections)):
        for item in detections:
            x1, y1, x2, y2 = item["box_xyxy"]
            draw.rectangle((offset + x1, 30 + y1, offset + x2, 30 + y2),
                           outline=(255, 220, 0), width=2)
            draw.text((offset + x1, 30 + y1),
                      f"{item['label']} {item['score']:.2f}",
                      fill=(20, 20, 20))
    return np.asarray(image)


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    device = "mps" if torch.backends.mps.is_available() else "cpu"
    processor = AutoProcessor.from_pretrained(DETECTOR)
    model = AutoModelForZeroShotObjectDetection.from_pretrained(
        DETECTOR).to(device).eval()
    rows = []
    original_model = structural.structural_model
    try:
        for seed in SEEDS:
            trace_path = (V323 / f"seed{seed}-present" /
                          f"seed{seed}-present/episode-trace.json")
            trace = json.loads(trace_path.read_text())
            for reach, end_hand in REACHES.items():
                structural.structural_model = (
                    lambda kind, side, shade=SHIRTS[seed]:
                    add_person(original_model(kind, side), shirt_rgba=shade))
                case = f"seed{seed}-{reach}-reach0.4s"
                room = LateCorridorRoom(case=tuple(trace["case"]),
                                        pose_source="command",
                                        branch=OUT / case)
                try:
                    room.layout(seed, absent=False, furnished=True)
                    room.actual[:] = actual_pose(trace)
                    room.estimated[:] = estimated_pose(trace)
                    room.head = 1.2
                    room.data.qpos[room.model.joint(
                        "left_shoulder_pitch").qposadr[0]] = -.8
                    for index in SAMPLES:
                        room.time_s = 16.2 + .1 * index
                        room._update_moving_obstacle()
                        hand = (START_HAND_XYZ_M + index / 4 *
                                (end_hand - START_HAND_XYZ_M))
                        set_person_pose(room.model, hand)
                        room._forward()
                        packet = (V342 / case / f"actor-{index:02d}.npz")
                        with np.load(packet) as saved:
                            actor_rgb = {camera: saved[camera].copy()
                                         for camera in CAMERAS}
                        views = {}
                        for camera in CAMERAS:
                            rgb, _ = room.capture(camera, depth=False)
                            if not np.array_equal(actor_rgb[camera],
                                                  np.asarray(rgb)):
                                raise AssertionError(
                                    f"V342 actor image changed: {case} {index} {camera}")
                            views[camera] = {"rgb": rgb,
                                             "evaluator_boxes": evaluator_boxes(
                                                 room, camera)}
                        results = {}
                        for camera in CAMERAS:
                            for prompt in PROMPTS:
                                prediction = detect(processor, model, device,
                                                    views[camera]["rgb"], prompt)
                                target = ("person" if prompt == "person ."
                                          else "hand_arm")
                                truth = views[camera]["evaluator_boxes"][target]
                                results[f"{camera}|{prompt}"] = {
                                    "actor_detections": prediction,
                                    "evaluator_target_box": truth,
                                    "evaluator_score": score(prediction, truth)}
                        row = {"case": case, "seed": seed, "reach": reach,
                               "sample": index,
                               "time_s": round(room.time_s, 3),
                               "positive_person": True,
                               "actor_packet_path": str(packet.relative_to(ROOT)),
                               "actor_packet_sha256": sha(packet),
                               "results": results}
                        rows.append(row)
                        if seed == SEEDS[0] and reach == "near":
                            lower = results["lower_rgb|hand . arm ."]["actor_detections"]
                            upper = results["upper_rgb|person ."]["actor_detections"]
                            frame = review_pair(
                                views["lower_rgb"]["rgb"],
                                views["upper_rgb"]["rgb"], lower, upper,
                                f"V344 {case} t{room.time_s:.1f}s | "
                                "yellow = actor prompt boxes")
                            path = OUT / f"review-{index:02d}.png"
                            Image.fromarray(frame).save(path)
                        print(case, index,
                              "upper-person", results[
                                  "upper_rgb|person ."]["evaluator_score"][
                                      "localized_iou_ge_0p25"],
                              "lower-arm", results[
                                  "lower_rgb|hand . arm ."]["evaluator_score"][
                                      "localized_iou_ge_0p25"], flush=True)
                finally:
                    room.close()
            # Same room/camera pose with no person supplies a true negative.
            structural.structural_model = original_model
            room = LateCorridorRoom(case=tuple(trace["case"]),
                                    pose_source="command",
                                    branch=OUT / f"seed{seed}-no-person")
            try:
                room.layout(seed, absent=False, furnished=True)
                room.actual[:] = actual_pose(trace)
                room.estimated[:] = estimated_pose(trace)
                room.head = 1.2
                room.data.qpos[room.model.joint(
                    "left_shoulder_pitch").qposadr[0]] = -.8
                for index in SAMPLES:
                    room.time_s = 16.2 + .1 * index
                    room._update_moving_obstacle()
                    room._forward()
                    results = {}
                    actor_frames = []
                    for camera in CAMERAS:
                        rgb, _ = room.capture(camera, depth=False)
                        source = OUT / f"seed{seed}-no-person-{index:02d}-{camera}.png"
                        rgb.save(source)
                        actor_frames.append({"path": str(source.relative_to(ROOT)),
                                             "sha256": sha(source)})
                        for prompt in PROMPTS:
                            prediction = detect(processor, model, device,
                                                rgb, prompt)
                            results[f"{camera}|{prompt}"] = {
                                "actor_detections": prediction,
                                "evaluator_target_box": None,
                                "evaluator_score": score(prediction, None)}
                    rows.append({"case": f"seed{seed}-no-person",
                                 "seed": seed, "positive_person": False,
                                 "sample": index,
                                 "time_s": round(room.time_s, 3),
                                 "actor_frames": actor_frames,
                                 "results": results})
            finally:
                room.close()
    finally:
        structural.structural_model = original_model
    report = {"scope": "V344 unchanged released Grounding DINO tiny prompts person. and hand.arm. on V342 selected-Cove actor upper/lower RGB at three reach times, two near/far trajectories and two shirt/scene seeds, plus no-person paired rooms at the same three times. The detector sees only actor images. Evaluator segmentation boxes are reconstructed from the same pose after pixel-exact image match and used only for scoring. Prompt detector boxes are not a calibrated safety monitor or identity estimator; contact, real person appearance and motor blur remain untested.",
              "source_sha256": sha(SOURCE),
              "person_scene_source_sha256": sha(Path(add_person.__code__.co_filename)),
              "selected_visual_sha256": sha(selected_visual()),
              "detector_weights_sha256": sha(DETECTOR / "model.safetensors"),
              "detector_config_sha256": sha(DETECTOR / "config.json"),
              "device": device, "rows": rows}
    (OUT / "person-grounding-audit.json").write_text(
        json.dumps(report, indent=2) + "\n")
    (OUT / SOURCE.name).write_bytes(SOURCE.read_bytes())


if __name__ == "__main__":
    main()
