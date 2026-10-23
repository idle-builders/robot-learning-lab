"""Released person prompt plus calibrated depth for a lost owner range track.

This is image-grounded association evidence, not person identity or a safety
detector. The model is loaded only when the range owner track becomes unknown.
"""

from __future__ import annotations

import time
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from transformers import AutoModelForZeroShotObjectDetection, AutoProcessor

from screen_robocasa_owner_safe_stop_v494 import depth_point
from owner_depth_change_component_v516 import packet_health


ROOT = Path(__file__).resolve().parents[3]
DETECTOR = ROOT / ".cache/ape-high-level/models/grounding-dino-tiny"
PERSON_SCORE = .25
_LOADED = None


def _model():
    global _LOADED
    if _LOADED is None:
        device = "mps" if torch.backends.mps.is_available() else "cpu"
        processor = AutoProcessor.from_pretrained(DETECTOR)
        model = AutoModelForZeroShotObjectDetection.from_pretrained(
            DETECTOR).to(device).eval()
        _LOADED = processor, model, device
    return _LOADED


def observe(rgb, depth, data, model, camera, estimate_xy, actual_xy):
    if depth is None:
        return {"status": "depth_missing", "detections": []}
    healthy, fractions = packet_health(depth, depth)
    if not healthy:
        return {"status": "depth_unhealthy", "central_valid_fraction": fractions[0],
                "detections": []}
    start = time.perf_counter()
    processor, detector, device = _model()
    tensors = processor(images=Image.fromarray(rgb), text=["person ."],
                        return_tensors="pt").to(device)
    with torch.no_grad():
        output = detector(**tensors)
    found = processor.post_process_grounded_object_detection(
        output, tensors.input_ids, threshold=.25, text_threshold=.25,
        target_sizes=[rgb.shape[:2]])[0]
    detections = sorted(({"label": str(label), "score": float(score),
                          "box_xyxy": [float(value) for value in box]}
                         for label, score, box in zip(found["text_labels"],
                                                      found["scores"], found["boxes"])),
                        key=lambda row: row["score"], reverse=True)
    detector_wall = time.perf_counter() - start
    if not detections or detections[0]["score"] < PERSON_SCORE:
        return {"status": "person_unconfirmed", "central_valid_fraction": fractions[0],
                "detections": detections, "detector_wall_s": detector_wall}
    point = depth_point(detections[0]["box_xyxy"], depth, data, model, camera)
    if point is None:
        return {"status": "person_depth_unlocalized",
                "central_valid_fraction": fractions[0],
                "detections": detections, "detector_wall_s": detector_wall}
    point = np.asarray(point, dtype=float)
    point[:2] += np.asarray(estimate_xy) - np.asarray(actual_xy)
    return {"status": "person_surface_localized",
            "central_valid_fraction": fractions[0],
            "detections": detections, "point_xyz_est_m": point.tolist(),
            "detector_wall_s": detector_wall}
