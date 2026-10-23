"""Diagnostic owner-click appearance association on a rigged-person clip.

Uses the project's frozen Grounding DINO and DINOv2 models. The actor receives
only the first-frame click and video frames. World positions score afterward;
there is no owner ID, pose, or person label in the actor input. This does not
perform navigation or claim a finished social identity system.
"""

from __future__ import annotations

import argparse
import hashlib
import itertools
import json
import math
from pathlib import Path

import cv2
import numpy as np
from PIL import Image
import torch
from transformers import (AutoImageProcessor, AutoModel,
                          AutoModelForZeroShotObjectDetection, AutoProcessor)


ROOT = Path(__file__).resolve().parents[3]
DETECTOR = ROOT / ".cache/ape-high-level/models/grounding-dino-tiny"
FEATURES = ROOT / ".cache/ape-high-level/models/dinov2-small"
DEFAULT_CLICK_XY = (150, 220)  # human-selected first-frame owner in V9–V19
MIN_APPEARANCE_MARGIN = 0.05  # diagnostic abstention gate, not calibrated


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def frames(path: Path) -> list[Image.Image]:
    capture = cv2.VideoCapture(str(path))
    rows = []
    while True:
        okay, bgr = capture.read()
        if not okay:
            break
        rows.append(Image.fromarray(cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)))
    capture.release()
    if len(rows) != 12:
        raise RuntimeError(f"Expected 12 video frames, got {len(rows)}")
    return rows


def person_detections(images, device):
    processor = AutoProcessor.from_pretrained(DETECTOR)
    model = AutoModelForZeroShotObjectDetection.from_pretrained(DETECTOR).to(device).eval()
    all_rows, crops = [], []
    for image in images:
        inputs = processor(images=image, text=["person ."], return_tensors="pt").to(device)
        with torch.no_grad():
            output = model(**inputs)
        found = processor.post_process_grounded_object_detection(
            output, inputs.input_ids, threshold=.25, text_threshold=.25,
            target_sizes=[(image.height, image.width)])[0]
        rows = []
        for score, box in zip(found["scores"], found["boxes"]):
            xyxy = [float(v) for v in box]
            x1, y1, x2, y2 = xyxy
            crop = image.crop((max(0, int(x1)), max(0, int(y1)),
                               min(image.width, math.ceil(x2)),
                               min(image.height, math.ceil(y2))))
            if crop.width < 8 or crop.height < 8:
                continue
            rows.append({"score": float(score), "box_xyxy": xyxy,
                         "feature_index": len(crops)})
            crops.append(crop)
        all_rows.append(rows)
    return all_rows, crops


def feature_vectors(crops, device):
    processor = AutoImageProcessor.from_pretrained(FEATURES)
    model = AutoModel.from_pretrained(FEATURES).to(device).eval()
    chunks = []
    with torch.no_grad():
        for start in range(0, len(crops), 8):
            pixels = processor(images=crops[start:start + 8],
                               return_tensors="pt")["pixel_values"].to(device)
            hidden = model(pixel_values=pixels).last_hidden_state[:, 0]
            chunks.append(torch.nn.functional.normalize(hidden, p=2, dim=-1).cpu().numpy())
    return np.concatenate(chunks) if chunks else np.empty((0, 384))


def project_x(root_xyz, eye, target, fov_deg, width):
    eye = np.asarray(eye, dtype=float)
    target = np.asarray(target, dtype=float)
    forward = target - eye
    forward /= np.linalg.norm(forward)
    point = np.asarray([root_xyz[0], root_xyz[1], root_xyz[2] + 1.0])
    relative = point - eye
    depth = float(np.dot(relative, forward))
    fx = width / (2 * math.tan(math.radians(fov_deg / 2)))
    return width / 2 + fx * relative[0] / depth


def evaluator_labels(detections, world_x):
    if len(detections) != 2:
        return [None] * len(detections)
    centers = [(row["box_xyxy"][0] + row["box_xyxy"][2]) / 2
               for row in detections]
    assignments = list(itertools.permutations((0, 1)))
    assignment = min(assignments, key=lambda order: sum(
        abs(centers[i] - world_x[order[i]]) for i in range(2)))
    return list(assignment)


def box_iou(a, b):
    left = max(a[0], b[0])
    top = max(a[1], b[1])
    right = min(a[2], b[2])
    bottom = min(a[3], b[3])
    intersection = max(0., right - left) * max(0., bottom - top)
    area_a = max(0., a[2] - a[0]) * max(0., a[3] - a[1])
    area_b = max(0., b[2] - b[0]) * max(0., b[3] - b[1])
    union = area_a + area_b - intersection
    return intersection / union if union > 0 else 0.


def blue_shirt_centroid_x(image):
    """Evaluator-only visible anchor for the distinct blue police shirt."""
    rgb = np.asarray(image).astype(np.int16)
    region = rgb[90:320]
    blue = ((region[:, :, 2] > region[:, :, 0] + 35) &
            (region[:, :, 2] > region[:, :, 1] + 15) &
            (region[:, :, 2] > 80))
    _, xs = np.nonzero(blue)
    return float(np.mean(xs)) if len(xs) >= 100 else None


def depth_support(depth, box):
    """Image-local depth availability; a box is not a person-surface mask."""
    x1, y1, x2, y2 = box
    height, width = depth.shape
    scale_x, scale_y = width / 640, height / 480
    region = depth[max(0, int(y1 * scale_y)):min(height, math.ceil(y2 * scale_y)),
                   max(0, int(x1 * scale_x)):min(width, math.ceil(x2 * scale_x))]
    tx1, tx2 = x1 + .35 * (x2 - x1), x1 + .65 * (x2 - x1)
    ty1, ty2 = y1 + .25 * (y2 - y1), y1 + .55 * (y2 - y1)
    torso = depth[max(0, int(ty1 * scale_y)):min(height, math.ceil(ty2 * scale_y)),
                  max(0, int(tx1 * scale_x)):min(width, math.ceil(tx2 * scale_x))]
    box_valid = np.isfinite(region) & (region > 0) & (region < 10)
    torso_valid = np.isfinite(torso) & (torso > 0) & (torso < 10)
    return {"box_valid_fraction": float(box_valid.mean()) if region.size else 0.,
            "torso_patch_valid_fraction": float(torso_valid.mean()) if torso.size else 0.,
            "torso_patch_depth_m": float(np.median(torso[torso_valid]))
            if torso_valid.any() else None}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--evidence", type=Path, required=True)
    parser.add_argument("--click-x", type=int, default=DEFAULT_CLICK_XY[0])
    parser.add_argument("--click-y", type=int, default=DEFAULT_CLICK_XY[1])
    args = parser.parse_args()
    click_xy = (args.click_x, args.click_y)
    evidence = args.evidence.resolve()
    scene = json.loads((evidence / "scene-info.json").read_text())
    motion = json.loads((evidence / "character-motion-trace.json").read_text())
    device = "mps" if torch.backends.mps.is_available() else "cpu"
    result = {"scope": "first-frame user click and frozen appearance across 12 video frames; evaluator world pose only after decisions",
              "click_xy": list(click_xy), "margin_gate": MIN_APPEARANCE_MARGIN,
              "model_sha256": {"detector": sha(DETECTOR / "model.safetensors"),
                               "features": sha(FEATURES / "model.safetensors")},
              "views": []}
    for fov in (70, 90):
        video = evidence / f"ape_social_{fov}deg.mp4"
        images = frames(video)
        depth_file = evidence / f"ape_social_{fov}deg-depth-sequence-stride3.npz"
        depth_sequence = np.load(depth_file)["depth_m"] if depth_file.exists() else None
        if depth_sequence is not None and depth_sequence.shape[0] != len(images):
            raise RuntimeError(f"Depth/video frame count mismatch for {fov}")
        detections, crops = person_detections(images, device)
        features = feature_vectors(crops, device)
        first = [i for i, row in enumerate(detections[0]) if
                 row["box_xyxy"][0] <= click_xy[0] <= row["box_xyxy"][2] and
                 row["box_xyxy"][1] <= click_xy[1] <= row["box_xyxy"][3]]
        if len(first) != 1:
            raise RuntimeError(f"Owner click not in exactly one {fov} proposal: {first}")
        exemplar = features[detections[0][first[0]]["feature_index"]]
        rows = []
        for frame, proposals in enumerate(detections):
            scores = [float(features[item["feature_index"]] @ exemplar)
                      for item in proposals]
            order = np.argsort(-np.asarray(scores)) if scores else []
            best = int(order[0]) if len(order) else None
            margin = (scores[int(order[0])] - scores[int(order[1])]
                      if len(order) > 1 else None)
            chosen = (best if best is not None and
                      (margin is None or margin >= MIN_APPEARANCE_MARGIN) else None)
            world_x = [project_x(person["xyz_m"], scene["camera_eye_xyz_m"],
                                 scene["camera_target_xyz_m"], fov, images[0].width)
                       for person in motion[frame]["characters"]]
            labels = evaluator_labels(proposals, world_x)
            depth_health = None
            if depth_sequence is not None:
                depth = depth_sequence[frame]
                healthy = np.isfinite(depth) & (depth > 0) & (depth < 10)
                depth_health = {"whole_frame_valid_fraction": float(healthy.mean()),
                    "candidate_support": [depth_support(depth, item["box_xyxy"])
                                          for item in proposals]}
            rows.append({"frame": frame, "proposal_count": len(proposals),
                         "scores": scores, "appearance_margin": margin,
                         "chosen_proposal": chosen,
                         "chosen_evaluator_character_index": labels[chosen] if chosen is not None else None,
                         "projected_character_x_evaluator_px": world_x,
                         "blue_shirt_centroid_evaluator_px": blue_shirt_centroid_x(images[frame]),
                         "depth_health": depth_health,
                         "two_box_iou": box_iou(proposals[0]["box_xyxy"],
                                                proposals[1]["box_xyxy"])
                         if len(proposals) == 2 else None,
                         "boxes_xyxy": [item["box_xyxy"] for item in proposals]})
        expected_delta = (rows[-1]["projected_character_x_evaluator_px"][1] -
                          rows[0]["projected_character_x_evaluator_px"][1])
        blue_values = [row["blue_shirt_centroid_evaluator_px"] for row in rows]
        observed_delta = (blue_values[-1] - blue_values[0]
                          if blue_values[0] is not None and blue_values[-1] is not None else None)
        result["views"].append({"fov_deg": fov, "video_sha256": sha(video),
            "depth_sequence_sha256": sha(depth_file) if depth_sequence is not None else None,
            "depth_shape_t_h_w": list(depth_sequence.shape) if depth_sequence is not None else None,
            "whole_frame_depth_healthy_frames_at_half": sum(
                row["depth_health"]["whole_frame_valid_fraction"] >= .5 for row in rows)
                if depth_sequence is not None else None,
            "both_torso_patches_depth_healthy_frames_at_half": sum(
                len(row["depth_health"]["candidate_support"]) == 2 and
                all(item["torso_patch_valid_fraction"] >= .5
                    for item in row["depth_health"]["candidate_support"])
                for row in rows) if depth_sequence is not None else None,
            "frames_with_two_proposals": sum(row["proposal_count"] == 2 for row in rows),
            "correct_owner_frames_evaluator": sum(row["chosen_evaluator_character_index"] == 0 for row in rows),
            "wrong_owner_frames_evaluator": sum(row["chosen_evaluator_character_index"] == 1 for row in rows),
            "unscored_choice_frames_evaluator": sum(
                row["chosen_proposal"] is not None and
                row["chosen_evaluator_character_index"] is None for row in rows),
            "abstain_frames": sum(row["chosen_proposal"] is None for row in rows),
            "distractor_expected_projected_delta_px": expected_delta,
            "blue_shirt_observed_delta_px_evaluator": observed_delta,
            "render_motion_delta_mismatch_px": (abs(expected_delta - observed_delta)
                                                if observed_delta is not None else None),
            "trace": rows})
    (evidence / "owner-video-analysis.json").write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps([{key: value for key, value in view.items() if key != "trace"}
                      for view in result["views"]], indent=2))


if __name__ == "__main__":
    main()
