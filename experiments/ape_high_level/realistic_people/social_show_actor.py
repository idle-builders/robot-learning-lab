"""V589-style new-show evidence checks on an Isaac RGB-D observation stream.

The script supplies a request and a prior taught crop. The actor sees only
images, radial depth, packet age and its own state; person/item USD paths and
the scripted response stay in the evaluator. A detected item is an
unattributed new-session observation, never restored old-owner identity.
"""

from __future__ import annotations

import math
import time

import cv2
import numpy as np
from PIL import Image
import torch


PROMPT = "bottle . can . jar . box ."
SLOW_SIMILARITY = .35  # V589 frozen object proposal/retrieval threshold
FAST_SIMILARITY = .60  # V403/V589 fast-template threshold
FAST_SCALES = (.8, .9, 1.0, 1.1, 1.2)
FAST_SEARCH_MARGIN_PX = 50
MAX_FRESH_AGE_S = .20


def local_point(depth: np.ndarray, box: list[float], image_wh: tuple[int, int],
                fov_deg: float) -> dict:
    """Use a central candidate patch, not all pixels in a detector rectangle."""
    width, height = image_wh
    x0, y0, x1, y1 = box
    cx, cy = (x0 + x1) / 2, (y0 + y1) / 2
    dx, dy = (x1 - x0) * .15, (y1 - y0) * .15
    dh, dw = depth.shape
    lo_x = max(0, int((cx - dx) * dw / width))
    hi_x = min(dw, math.ceil((cx + dx) * dw / width))
    lo_y = max(0, int((cy - dy) * dh / height))
    hi_y = min(dh, math.ceil((cy + dy) * dh / height))
    patch = depth[lo_y:hi_y, lo_x:hi_x]
    valid = np.isfinite(patch) & (patch >= .15) & (patch <= 2.8)
    fraction = float(valid.mean()) if patch.size else 0.0
    count = int(valid.sum())
    distance = float(np.median(patch[valid])) if count else None
    point = None
    if distance is not None:
        focal = width / (2 * math.tan(math.radians(fov_deg / 2)))
        ray = np.array([(cx - width / 2) / focal,
                        -(cy - height / 2) / focal, 1.0])
        point = (distance * ray / np.linalg.norm(ray)).tolist()
    return {"valid_fraction": fraction, "valid_pixels": count,
            "radial_distance_m": distance, "point_camera_xyz_m": point}


class SocialShowActor:
    def __init__(self, vision, taught_crop: Image.Image, *, object_detector_path=None):
        self.vision = vision
        if vision.person_detector == "grounding_dino":
            self.detector_processor = vision.detector_processor
            self.detector = vision.detector
        else:
            if object_detector_path is None:
                raise ValueError("An open-vocabulary object detector is required")
            from transformers import AutoModelForZeroShotObjectDetection, AutoProcessor
            self.detector_processor = AutoProcessor.from_pretrained(object_detector_path)
            self.detector = AutoModelForZeroShotObjectDetection.from_pretrained(
                object_detector_path).to(vision.device).eval()
        self.taught = self._features([taught_crop])[0]
        self.before_rgb: np.ndarray | None = None
        self.template: np.ndarray | None = None
        self.last_box: list[float] | None = None
        self.previous_point: np.ndarray | None = None
        self.streak = 0
        self.outcome = "new_show_unconfirmed_unknown"

    def _features(self, crops: list[Image.Image]) -> np.ndarray:
        pixels = self.vision.feature_processor(images=crops,
                                               return_tensors="pt")["pixel_values"].to(self.vision.device)
        with torch.no_grad():
            hidden = self.vision.feature_model(pixel_values=pixels).last_hidden_state[:, 0]
            unit = torch.nn.functional.normalize(hidden, p=2, dim=-1)
        return unit.cpu().numpy()

    def request(self, rgb: np.ndarray, *, time_s: float) -> dict:
        self.before_rgb = rgb.copy()
        return {"intent": "request_new_show", "time_s": time_s,
                "identity_state": "new_teacher_unattributed",
                "speech_intent": "Please show me the can again."}

    def _slow_proposals(self, image: Image.Image) -> list[dict]:
        inputs = self.detector_processor(images=image, text=[PROMPT],
                                                return_tensors="pt").to(self.vision.device)
        with torch.no_grad():
            output = self.detector(**inputs)
        found = self.detector_processor.post_process_grounded_object_detection(
            output, inputs.input_ids, threshold=.25, text_threshold=.25,
            target_sizes=[(image.height, image.width)])[0]
        rows, crops = [], []
        for box, score, label in zip(found["boxes"], found["scores"], found["text_labels"]):
            xyxy = [float(value) for value in box]
            x0, y0, x1, y1 = xyxy
            if (x1 - x0) * (y1 - y0) < 200:
                continue
            crop = image.crop((max(0, int(x0)), max(0, int(y0)),
                               min(image.width, math.ceil(x1)),
                               min(image.height, math.ceil(y1))))
            if crop.width < 8 or crop.height < 8:
                continue
            rows.append({"box_xyxy": xyxy, "detector_score": float(score),
                         "generic_label": str(label)})
            crops.append(crop)
        if crops:
            similarities = self._features(crops) @ self.taught
            for row, similarity in zip(rows, similarities):
                row["similarity"] = float(similarity)
        return sorted(rows, key=lambda row: row["similarity"], reverse=True)

    def _fast_match(self, rgb: np.ndarray) -> tuple[float, list[int]]:
        assert self.template is not None and self.last_box is not None
        bgr = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
        x0, y0, x1, y1 = self.last_box
        left = max(0, int(x0) - FAST_SEARCH_MARGIN_PX)
        top = max(0, int(y0) - FAST_SEARCH_MARGIN_PX)
        right = min(bgr.shape[1], math.ceil(x1) + FAST_SEARCH_MARGIN_PX)
        bottom = min(bgr.shape[0], math.ceil(y1) + FAST_SEARCH_MARGIN_PX)
        search = bgr[top:bottom, left:right]
        options = []
        for scale in FAST_SCALES:
            sample = cv2.resize(self.template, None, fx=scale, fy=scale,
                                interpolation=cv2.INTER_AREA)
            if search.shape[0] < sample.shape[0] or search.shape[1] < sample.shape[1]:
                continue
            response = cv2.matchTemplate(search, sample, cv2.TM_CCOEFF_NORMED)
            _, score, _, xy = cv2.minMaxLoc(response)
            options.append((float(score), [left + xy[0], top + xy[1],
                                           left + xy[0] + sample.shape[1],
                                           top + xy[1] + sample.shape[0]]))
        if not options:
            return 0.0, [left, top, left, top]
        return max(options, key=lambda option: option[0])

    def observe(self, rgb: np.ndarray, depth: np.ndarray, *, time_s: float,
                frame_age_s: float, fov_deg: float) -> dict:
        if self.before_rgb is None:
            raise RuntimeError("request must precede new-show observations")
        if self.outcome == "new_show_metric_item_seen_unattributed":
            return {"intent": "hold_item_seen", "outcome": self.outcome,
                    "time_s": time_s}
        if self.template is None:
            started = time.perf_counter()
            proposals = self._slow_proposals(Image.fromarray(rgb))
            elapsed = time.perf_counter() - started
            chosen = proposals[0] if proposals and proposals[0]["similarity"] >= SLOW_SIMILARITY else None
            if chosen is None:
                return {"intent": "wait_for_show", "reason": "taught_item_not_seen",
                        "time_s": time_s, "proposals": proposals,
                        "slow_inference_wall_s": elapsed}
            box = chosen["box_xyxy"]
            x0, y0, x1, y1 = [int(value) for value in box]
            region = np.abs(rgb[y0:y1, x0:x1].astype(float) -
                            self.before_rgb[y0:y1, x0:x1].astype(float))
            change = float(np.mean(region) / 255.0) if region.size else 0.0
            support = local_point(depth, box, (rgb.shape[1], rgb.shape[0]), fov_deg)
            common = {"time_s": time_s, "proposals": proposals,
                      "chosen_box_xyxy": box, "rgb_change_in_box": change,
                      "candidate_depth": support, "slow_inference_wall_s": elapsed}
            if change < .05 or support["valid_fraction"] < .7 or support["valid_pixels"] < 50:
                return {**common, "intent": "wait_for_show",
                        "reason": "new_show_evidence_weak"}
            crop = rgb[y0:y1, x0:x1]
            self.template = cv2.cvtColor(crop, cv2.COLOR_RGB2BGR)
            self.last_box = box
            return {**common, "intent": "check_fresh_show",
                    "reason": "candidate_needs_two_fresh_views"}
        started = time.perf_counter()
        similarity, box = self._fast_match(rgb)
        matching_wall = time.perf_counter() - started
        support = local_point(depth, box, (rgb.shape[1], rgb.shape[0]), fov_deg)
        point = support["point_camera_xyz_m"]
        fresh = (frame_age_s + matching_wall <= MAX_FRESH_AGE_S and
                 similarity >= FAST_SIMILARITY and
                 support["valid_fraction"] >= .7 and point is not None)
        if fresh:
            current = np.asarray(point)
            same = (self.previous_point is not None and
                    np.linalg.norm(current - self.previous_point) <= .10)
            self.streak = self.streak + 1 if same else 1
            self.previous_point = current
            self.last_box = box
        else:
            self.streak = 0
            self.previous_point = None
        common = {"time_s": time_s, "fast_similarity": similarity,
                  "candidate_box_xyxy": box, "candidate_depth": support,
                  "frame_age_s": frame_age_s + matching_wall,
                  "fast_inference_wall_s": matching_wall,
                  "fresh": fresh, "consistent_streak": self.streak}
        if self.streak >= 2:
            self.outcome = "new_show_metric_item_seen_unattributed"
            return {**common, "intent": "new_item_seen_unattributed",
                    "outcome": self.outcome,
                    "point_camera_xyz_m": self.previous_point.tolist()}
        return {**common, "intent": "check_fresh_show",
                "reason": "need_second_consistent_view" if fresh else "fresh_candidate_unavailable"}
