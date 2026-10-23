"""Streaming owner attention and bounded motion decisions for Isaac observations.

The actor sees pixels, radial depth, range beams and its own previous command.
It never receives character paths or simulator poses. A first-frame user click
teaches an appearance exemplar; later identity is only provisional.
"""

from __future__ import annotations

import math
from pathlib import Path
import time

import numpy as np
from PIL import Image
import torch
from transformers import (AutoImageProcessor, AutoModel,
                          AutoModelForZeroShotObjectDetection, AutoProcessor)

from fast_owner_flow import FastOwnerFlow


class OnlineOwnerActor:
    def __init__(self, detector: str | Path, features: str | Path, *,
                 click_xy: tuple[int, int] = (100, 220), device: str = "cpu",
                 person_detector: str = "grounding_dino",
                 fallback_detector: str | Path | None = None,
                 depth_separated_overlap: bool = False):
        self.device = device
        self.click_xy = click_xy
        self.person_detector = person_detector
        self.depth_separated_overlap = depth_separated_overlap
        if person_detector == "grounding_dino":
            self.detector_processor = AutoProcessor.from_pretrained(detector)
            self.detector = AutoModelForZeroShotObjectDetection.from_pretrained(
                detector).to(device).eval()
        elif person_detector == "ssdlite":
            from torchvision.models.detection import (
                SSDLite320_MobileNet_V3_Large_Weights,
                ssdlite320_mobilenet_v3_large)

            weights = SSDLite320_MobileNet_V3_Large_Weights.COCO_V1
            self.detector_processor = weights.transforms()
            self.detector = ssdlite320_mobilenet_v3_large(weights=weights).to(device).eval()
        else:
            raise ValueError(f"Unknown person detector: {person_detector}")
        self.fallback_processor = None
        self.fallback_model = None
        if fallback_detector is not None:
            if person_detector != "ssdlite":
                raise ValueError("Grounding DINO fallback requires SSDLite primary")
            self.fallback_processor = AutoProcessor.from_pretrained(fallback_detector)
            self.fallback_model = AutoModelForZeroShotObjectDetection.from_pretrained(
                fallback_detector).to(device).eval()
        self.feature_processor = AutoImageProcessor.from_pretrained(features)
        self.feature_model = AutoModel.from_pretrained(features).to(device).eval()
        self.owner_feature: np.ndarray | None = None
        self.identity_lost = False
        self.last_depth_m: float | None = None
        self.last_time_s: float | None = None
        self.last_bearing_deg: float | None = None
        self.last_proposal_timing: dict | None = None
        self.fast_flow = FastOwnerFlow()
        self.fast_flow_ready = False
        self.fast_frames_since_detection = 0
        self.consecutive_empty_full_checks = 0
        self.unconfirmed_appearance_checks = 0
        self.nested_overlap_retry_used = False
        self.appearance_margin_retry_used = False
        self.force_empty_next_proposal = False
        self.force_primary_empty_next_proposal = False

    def seed_fast_flow(self, rgb: np.ndarray, depth: np.ndarray, decision: dict,
                       *, time_s: float, fov_deg: float) -> dict:
        """Start a short track only after a depth-supported full appearance decision."""
        self.fast_flow_ready = False
        selected = decision.get("chosen_proposal")
        if (selected is None or decision.get("local_depth_m") is None or
                decision.get("intent") not in
                ("look_to_owner", "approach_bounded", "hold_near_person")):
            return {"status": "unseeded", "reason": "no_valid_full_detection"}
        seed = self.fast_flow.seed(rgb, depth,
            decision["proposals"][selected]["box_xyxy"],
            decision["local_depth_m"], time_s=time_s, fov_deg=fov_deg)
        self.fast_flow_ready = seed["status"] == "seeded"
        self.fast_frames_since_detection = 0
        return seed

    def observe_fast(self, rgb: np.ndarray, depth: np.ndarray, range_m: np.ndarray,
                     angles_deg: np.ndarray, *, time_s: float, fov_deg: float,
                     ego_yaw_delta_deg: float = 0.0) -> dict:
        """Use fresh geometry between full detections; never assign a new identity."""
        self.last_proposal_timing = None
        if self.identity_lost or not self.fast_flow_ready:
            return {"intent": "hold_reobserve", "reason": "fast_track_unavailable",
                    "observation_method": "rgbd_flow", "time_s": time_s}
        started = time.perf_counter()
        track = self.fast_flow.update(rgb, depth, time_s=time_s,
            fov_deg=fov_deg, ego_yaw_delta_deg=ego_yaw_delta_deg)
        self.last_proposal_timing = {"flow_s": time.perf_counter() - started,
                                     "total_s": time.perf_counter() - started}
        self.fast_frames_since_detection += 1
        if track["status"] != "tracked":
            self.fast_flow_ready = False
            if track.get("reason") == "track_time_gap":
                # The simulator advanced during a slow inference. A missing
                # flow update is not evidence that a different person took
                # the owner's place. Freeze and require a fresh full proposal
                # with the existing appearance, depth and bearing checks.
                self.consecutive_empty_full_checks = 1
                return {"intent": "hold_reobserve",
                        "reason": "fast_track_gap_retry_full_detector",
                        "observation_method": "rgbd_flow", "track": track,
                        "time_s": time_s}
            # Occlusion can leave a single foreground person proposal on the
            # next slow frame. Appearance, depth and bearing are insufficient
            # to silently restore the old social identity in that case.
            self.identity_lost = True
            return {"intent": "hold_ask_show_again",
                    "reason": "fast_track_uncertain_owner_identity_unbound",
                    "observation_method": "rgbd_flow", "track": track, "time_s": time_s}
        bearing_deg = track["bearing_deg"]
        depth_m = track["depth_m"]
        self.last_depth_m = depth_m
        self.last_time_s = time_s
        self.last_bearing_deg = bearing_deg
        forward = range_m[:, np.abs(angles_deg) <= 20]
        finite = forward[np.isfinite(forward)]
        nearest = float(finite.min()) if finite.size else None
        common = {"observation_method": "rgbd_flow", "track": track,
                  "owner_bearing_deg": bearing_deg, "local_depth_m": depth_m,
                  "nearest_forward_range_m": nearest, "time_s": time_s}
        if abs(bearing_deg) > 8:
            return {**common, "intent": "look_to_owner",
                    "yaw_delta_deg": float(np.clip(bearing_deg, -10, 10)),
                    "reason": "tracked_owner_off_center"}
        if nearest is None:
            return {**common, "intent": "hold_reobserve", "reason": "range_unknown"}
        if nearest < 1.25 or depth_m < 1.25:
            return {**common, "intent": "hold_near_person",
                    "reason": "near_body_clearance"}
        return {**common, "intent": "approach_bounded", "step_m": 0.10,
                "reason": "tracked_owner_centered_and_range_clear"}

    def _propose(self, image: Image.Image) -> tuple[list[dict], np.ndarray]:
        started = time.perf_counter()
        if self.force_empty_next_proposal:
            self.force_empty_next_proposal = False
            self.last_proposal_timing = {"detector_s": 0.0, "feature_s": 0.0,
                                         "detector_source": "forced_empty",
                                         "total_s": time.perf_counter() - started}
            return [], np.empty((0, 384), dtype=np.float32)
        if self.person_detector == "grounding_dino":
            inputs = self.detector_processor(images=image, text=["person ."],
                                             return_tensors="pt").to(self.device)
            with torch.no_grad():
                output = self.detector(**inputs)
            found = self.detector_processor.post_process_grounded_object_detection(
                output, inputs.input_ids, threshold=.25, text_threshold=.25,
                target_sizes=[(image.height, image.width)])[0]
            detections = zip(found["scores"], found["boxes"])
        elif self.force_primary_empty_next_proposal:
            # A labeled primary-detector outage exercises the optional
            # fallback without suppressing the full person proposal path.
            self.force_primary_empty_next_proposal = False
            detections = ()
        else:
            tensor = self.detector_processor(image).to(self.device)
            with torch.no_grad():
                found = self.detector([tensor])[0]
            detections = ((score, box) for score, label, box in zip(
                found["scores"], found["labels"], found["boxes"])
                if int(label) == 1 and float(score) >= .35)
        detector_s = time.perf_counter() - started
        detector_source = self.person_detector
        fallback_s = 0.0
        rows, crops = [], []
        for confidence, box in detections:
            x1, y1, x2, y2 = [float(value) for value in box]
            crop = image.crop((max(0, int(x1)), max(0, int(y1)),
                               min(image.width, math.ceil(x2)),
                               min(image.height, math.ceil(y2))))
            if crop.width < 8 or crop.height < 8:
                continue
            rows.append({"box_xyxy": [x1, y1, x2, y2],
                         "person_score": float(confidence)})
            crops.append(crop)
        if not crops and self.fallback_model is not None:
            fallback_started = time.perf_counter()
            inputs = self.fallback_processor(images=image, text=["person ."],
                                             return_tensors="pt").to(self.device)
            with torch.no_grad():
                output = self.fallback_model(**inputs)
            found = self.fallback_processor.post_process_grounded_object_detection(
                output, inputs.input_ids, threshold=.25, text_threshold=.25,
                target_sizes=[(image.height, image.width)])[0]
            for confidence, box in zip(found["scores"], found["boxes"]):
                x1, y1, x2, y2 = [float(value) for value in box]
                crop = image.crop((max(0, int(x1)), max(0, int(y1)),
                                   min(image.width, math.ceil(x2)),
                                   min(image.height, math.ceil(y2))))
                if crop.width < 8 or crop.height < 8:
                    continue
                rows.append({"box_xyxy": [x1, y1, x2, y2],
                             "person_score": float(confidence)})
                crops.append(crop)
            fallback_s = time.perf_counter() - fallback_started
            detector_source = "grounding_dino_fallback"
            detector_s += fallback_s
        if not crops:
            self.last_proposal_timing = {"detector_s": detector_s,
                "fallback_s": fallback_s, "detector_source": detector_source,
                "feature_s": 0.0, "total_s": time.perf_counter() - started}
            return rows, np.empty((0, 384), dtype=np.float32)
        feature_started = time.perf_counter()
        pixels = self.feature_processor(images=crops,
                                        return_tensors="pt")["pixel_values"].to(self.device)
        with torch.no_grad():
            hidden = self.feature_model(pixel_values=pixels).last_hidden_state[:, 0]
            unit = torch.nn.functional.normalize(hidden, p=2, dim=-1)
        self.last_proposal_timing = {"detector_s": detector_s,
            "fallback_s": fallback_s, "detector_source": detector_source,
            "feature_s": time.perf_counter() - feature_started,
            "total_s": time.perf_counter() - started}
        return rows, unit.cpu().numpy()

    @staticmethod
    def _local_depth(depth: np.ndarray, box: list[float], image_wh: tuple[int, int]) -> tuple[float, float | None]:
        x1, y1, x2, y2 = box
        width, height = image_wh
        dh, dw = depth.shape
        tx1, tx2 = x1 + .35 * (x2 - x1), x1 + .65 * (x2 - x1)
        ty1, ty2 = y1 + .25 * (y2 - y1), y1 + .55 * (y2 - y1)
        patch = depth[max(0, int(ty1 * dh / height)):min(dh, math.ceil(ty2 * dh / height)),
                      max(0, int(tx1 * dw / width)):min(dw, math.ceil(tx2 * dw / width))]
        healthy = np.isfinite(patch) & (patch > 0) & (patch < 10)
        fraction = float(healthy.mean()) if patch.size else 0.0
        return fraction, float(np.median(patch[healthy])) if healthy.any() else None

    @staticmethod
    def _selected_box_overlap(box: list[float], proposals: list[dict],
                              selected: int) -> float:
        """Fraction of the smaller box shared with another person proposal."""
        def area(rect: list[float]) -> float:
            return max(0., rect[2] - rect[0]) * max(0., rect[3] - rect[1])

        overlap = 0.0
        for index, row in enumerate(proposals):
            if index == selected:
                continue
            other = row["box_xyxy"]
            intersection = (max(0., min(box[2], other[2]) - max(box[0], other[0])) *
                            max(0., min(box[3], other[3]) - max(box[1], other[1])))
            overlap = max(overlap, intersection / max(1., min(area(box), area(other))))
        return overlap

    @staticmethod
    def _nested_overlap_only(box: list[float], proposals: list[dict],
                             selected: int) -> bool:
        """Identify a small contained person proposal, not a lateral crossing."""
        area = max(1., (box[2] - box[0]) * (box[3] - box[1]))
        nested = False
        for index, row in enumerate(proposals):
            if index == selected:
                continue
            other = row["box_xyxy"]
            other_area = max(1., (other[2] - other[0]) * (other[3] - other[1]))
            intersection = (max(0., min(box[2], other[2]) - max(box[0], other[0])) *
                            max(0., min(box[3], other[3]) - max(box[1], other[1])))
            if intersection / min(area, other_area) <= .25:
                continue
            if other_area > .55 * area or intersection / other_area < .9:
                return False
            nested = True
        return nested

    def _overlap_depth_layers(self, depth: np.ndarray, proposals: list[dict],
                              selected: int, image_wh: tuple[int, int]
                              ) -> tuple[list[dict], list[dict]]:
        """Keep nearby overlaps; set aside only small, clearly farther boxes.

        A distant portrait or a small distant person does not merge with the
        selected near owner surface. This is an actor-visible RGB-D distinction,
        not a claim that the other proposal is a photograph or safe to ignore.
        """
        selected_box = proposals[selected]["box_xyxy"]
        selected_fraction, selected_depth = self._local_depth(
            depth, selected_box, image_wh)
        if selected_fraction < .8 or selected_depth is None:
            return proposals, []
        selected_area = max(1., (selected_box[2] - selected_box[0]) *
                            (selected_box[3] - selected_box[1]))
        kept = [proposals[selected]]
        separated = []
        for index, row in enumerate(proposals):
            if index == selected:
                continue
            box = row["box_xyxy"]
            area = max(1., (box[2] - box[0]) * (box[3] - box[1]))
            fraction, other_depth = self._local_depth(depth, box, image_wh)
            if (area <= .25 * selected_area and fraction >= .8 and
                    other_depth is not None and
                    other_depth - selected_depth >= .6 and
                    other_depth >= 1.25 * selected_depth):
                separated.append({"proposal_index": index,
                    "selected_depth_m": selected_depth,
                    "other_depth_m": other_depth,
                    "other_area_fraction": area / selected_area})
            else:
                kept.append(row)
        return kept, separated

    def observe(self, rgb: np.ndarray, depth: np.ndarray, range_m: np.ndarray,
                angles_deg: np.ndarray, *, time_s: float, fov_deg: float,
                ego_yaw_delta_deg: float = 0.0) -> dict:
        if self.identity_lost:
            return {"intent": "hold_ask_show_again", "reason": "old_owner_identity_lost",
                    "time_s": time_s}
        image = Image.fromarray(rgb)
        rows, features = self._propose(image)
        if self.owner_feature is None:
            selected = [index for index, row in enumerate(rows) if
                        row["box_xyxy"][0] <= self.click_xy[0] <= row["box_xyxy"][2] and
                        row["box_xyxy"][1] <= self.click_xy[1] <= row["box_xyxy"][3]]
            if len(selected) != 1:
                return {"intent": "hold_ask_show_again", "reason": "click_ambiguous",
                        "proposals": rows, "time_s": time_s}
            self.owner_feature = features[selected[0]].copy()
        scores = features @ self.owner_feature if len(features) else np.empty(0)
        order = np.argsort(-scores)
        if not len(order):
            self.consecutive_empty_full_checks += 1
            if self.consecutive_empty_full_checks == 1 and self.owner_feature is not None:
                # One missed proposal is not evidence that the owner left.
                # Freeze motion and retry a new full image; no flow seed is
                # allowed from this frame.
                return {"intent": "hold_reobserve", "reason": "one_person_detector_miss",
                        "proposals": rows, "time_s": time_s}
            self.identity_lost = True
            return {"intent": "hold_ask_show_again", "reason": "repeated_no_person",
                    "proposals": rows, "time_s": time_s}
        recovering_from_miss = self.consecutive_empty_full_checks > 0
        self.consecutive_empty_full_checks = 0
        best = int(order[0])
        similarity = float(scores[best])
        minimum_similarity = .75 if recovering_from_miss else .65
        if similarity < minimum_similarity:
            self.unconfirmed_appearance_checks += 1
            # A singleton low-score bystander is not an owner. Keep the old
            # appearance/bearing untouched and request a bounded active look.
            # After two failed fresh checks, ask the person to reintroduce
            # themselves instead of silently binding the other candidate.
            if (self.unconfirmed_appearance_checks <= 2 and
                    self.last_bearing_deg is not None):
                expected = self.last_bearing_deg - ego_yaw_delta_deg
                return {"intent": "hold_reobserve",
                        "reason": "owner_appearance_unconfirmed",
                        "proposals": rows, "chosen_proposal": best,
                        "appearance_similarity": similarity,
                        "minimum_appearance_similarity": minimum_similarity,
                        "remembered_owner_bearing_deg": expected,
                        "requested_gaze_delta_deg": float(np.clip(expected, -10, 10)),
                        "unconfirmed_checks": self.unconfirmed_appearance_checks,
                        "time_s": time_s}
            self.identity_lost = True
            return {"intent": "hold_ask_show_again",
                    "reason": "owner_appearance_unconfirmed_after_retries",
                    "proposals": rows, "chosen_proposal": best,
                    "appearance_similarity": similarity,
                    "minimum_appearance_similarity": minimum_similarity,
                    "time_s": time_s}
        self.unconfirmed_appearance_checks = 0
        margin = float(scores[order[0]] - scores[order[1]]) if len(order) > 1 else None
        if margin is not None and margin < .05:
            if not self.appearance_margin_retry_used:
                self.appearance_margin_retry_used = True
                return {"intent": "hold_reobserve",
                        "reason": "appearance_margin_retry_full_detector",
                        "proposals": rows, "appearance_margin": margin,
                        "time_s": time_s}
            self.identity_lost = True
            return {"intent": "hold_ask_show_again", "reason": "appearance_ambiguous",
                    "proposals": rows, "appearance_margin": margin, "time_s": time_s}
        self.appearance_margin_retry_used = False
        box = rows[best]["box_xyxy"]
        overlap_rows = rows
        depth_separated = []
        overlap_selected = best
        if self.depth_separated_overlap:
            overlap_rows, depth_separated = self._overlap_depth_layers(
                depth, rows, best, (image.width, image.height))
            overlap_selected = 0
        box_overlap = self._selected_box_overlap(
            box, overlap_rows, overlap_selected)
        if self.last_time_s is not None and box_overlap > .25:
            if (not self.nested_overlap_retry_used and
                    self._nested_overlap_only(box, overlap_rows, overlap_selected)):
                # Detector duplicates often crop a person's upper body inside
                # its full-body proposal. Do not move or accept that crop as
                # a new identity; demand one more full frame before unbinding.
                self.nested_overlap_retry_used = True
                self.consecutive_empty_full_checks = 1
                return {"intent": "hold_reobserve",
                        "reason": "nested_person_proposal_retry",
                        "proposals": rows, "chosen_proposal": best,
                        "selected_box_overlap_fraction": box_overlap,
                        "depth_separated_proposals": depth_separated,
                        "time_s": time_s}
            # A crossing can make a detector's crop contain both people. Its
            # embedding and center-depth may still look plausible, so neither
            # may refresh the old owner binding or seed short-horizon flow.
            self.identity_lost = True
            return {"intent": "hold_ask_show_again", "reason": "person_boxes_overlap",
                    "proposals": rows, "chosen_proposal": best,
                    "selected_box_overlap_fraction": box_overlap,
                    "depth_separated_proposals": depth_separated,
                    "time_s": time_s}
        self.nested_overlap_retry_used = False
        fraction, depth_m = self._local_depth(depth, box, (image.width, image.height))
        if fraction < .5 or depth_m is None:
            return {"intent": "hold_reobserve", "reason": "local_depth_unhealthy",
                    "proposals": rows, "chosen_proposal": best, "time_s": time_s,
                    "local_depth_valid_fraction": fraction}
        if self.last_depth_m is not None and self.last_time_s is not None:
            limit = .10 + max(0., time_s - self.last_time_s) * 1.0
            if abs(depth_m - self.last_depth_m) > limit:
                self.identity_lost = True
                return {"intent": "hold_ask_show_again", "reason": "depth_discontinuity",
                        "proposals": rows, "chosen_proposal": best, "time_s": time_s,
                        "local_depth_m": depth_m, "last_depth_m": self.last_depth_m,
                        "allowed_depth_delta_m": limit}
        center_x = (box[0] + box[2]) / 2
        fx = image.width / (2 * math.tan(math.radians(fov_deg / 2)))
        bearing_deg = math.degrees(math.atan2(center_x - image.width / 2, fx))
        residual = None
        allowed = None
        if self.last_bearing_deg is not None and self.last_time_s is not None:
            elapsed_s = max(0.0, time_s - self.last_time_s)
            predicted = self.last_bearing_deg - ego_yaw_delta_deg
            residual = (bearing_deg - predicted + 180.0) % 360.0 - 180.0
            # A provisional 50 deg/s person-bearing allowance plus 3 deg
            # measurement slack; high angular innovations invalidate identity.
            allowed = 3.0 + 50.0 * elapsed_s
            if abs(residual) > allowed:
                self.identity_lost = True
                return {"intent": "hold_ask_show_again",
                        "reason": "bearing_inconsistent_with_ego_motion",
                        "proposals": rows, "chosen_proposal": best,
                        "time_s": time_s, "owner_bearing_deg": bearing_deg,
                        "predicted_owner_bearing_deg": predicted,
                        "bearing_residual_deg": residual,
                        "allowed_bearing_residual_deg": allowed,
                        "ego_yaw_delta_deg": ego_yaw_delta_deg}
        self.last_depth_m = depth_m
        self.last_time_s = time_s
        self.last_bearing_deg = bearing_deg
        in_front = np.abs(angles_deg) <= 20
        forward = range_m[:, in_front]
        finite = forward[np.isfinite(forward)]
        nearest_forward_m = float(finite.min()) if finite.size else None
        common = {"proposals": rows, "chosen_proposal": best,
                  "appearance_similarity": similarity,
                  "appearance_margin": margin, "local_depth_valid_fraction": fraction,
                  "depth_separated_proposals": depth_separated,
                  "local_depth_m": depth_m, "owner_bearing_deg": bearing_deg,
                  "bearing_residual_deg": residual,
                  "allowed_bearing_residual_deg": allowed,
                  "ego_yaw_delta_deg": ego_yaw_delta_deg,
                  "nearest_forward_range_m": nearest_forward_m, "time_s": time_s}
        if abs(bearing_deg) > 8:
            return {**common, "intent": "look_to_owner",
                    "yaw_delta_deg": float(np.clip(bearing_deg, -10, 10)),
                    "reason": "owner_off_center"}
        if nearest_forward_m is None:
            return {**common, "intent": "hold_reobserve",
                    "reason": "range_unknown"}
        if nearest_forward_m < 1.25 or depth_m < 1.25:
            return {**common, "intent": "hold_near_person",
                    "reason": "near_body_clearance"}
        return {**common, "intent": "approach_bounded",
                "step_m": 0.10, "reason": "owner_centered_and_range_clear"}
