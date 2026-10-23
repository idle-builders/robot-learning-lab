"""Run a small image-grounded, head-controlled teach/find probe.

This is a provisional 50 cm gaze rig with textured HOPE objects. A supplied
click selects a generic RGB proposal or a SAM 2 mask for teaching; search then
gets that crop, fresh RGB frames and camera forward kinematics. Ground-truth
segmentation is evaluator-only and search never receives the moved object's
position. The fixed base, synthetic room, exact pose and preselected click
remain limitations.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path

import numpy as np
from PIL import Image
import torch
from transformers import (AutoImageProcessor, AutoModel, Sam2Model, Sam2Processor,
                          AutoModelForZeroShotObjectDetection, AutoProcessor)

try:
    from .scene import HeadScene, HEIGHT, OBJECTS, ROOT, WIDTH
except ImportError:
    from scene import HeadScene, HEIGHT, OBJECTS, ROOT, WIDTH


DETECTOR = ROOT / ".cache/ape-high-level/models/grounding-dino-tiny"
RETRIEVER = ROOT / ".cache/ape-high-level/models/dinov2-small"
CLICK_SEGMENTER = ROOT / ".cache/ape-high-level/models/sam2-hiera-tiny"
PROMPT = "bottle . can . jar . box ."
TEACH_POSITIONS = {"TomatoSauce": (.45, 0), "Parmesan": (.50, .4),
                   "CreamCheese": (.50, -.35)}
DEFAULT_CLICKS = Path(__file__).with_name("clicks-example.json")


class Vision:
    def __init__(self, teach_selector: str, device: str | None = None):
        self.device = device or ("mps" if torch.backends.mps.is_available() else "cpu")
        self.detector_processor = AutoProcessor.from_pretrained(DETECTOR)
        self.detector = AutoModelForZeroShotObjectDetection.from_pretrained(
            DETECTOR).to(self.device).eval()
        self.feature_processor = AutoImageProcessor.from_pretrained(RETRIEVER)
        self.feature_model = AutoModel.from_pretrained(RETRIEVER).to(self.device).eval()
        self.segmenter = None
        if teach_selector == "sam2-click":
            self.segment_processor = Sam2Processor.from_pretrained(CLICK_SEGMENTER)
            self.segmenter = Sam2Model.from_pretrained(CLICK_SEGMENTER).to(self.device).eval()

    def select_click(self, image: Image.Image, click: list[int],
                     proposals: list[dict]) -> tuple[list[int] | None, dict]:
        if self.segmenter is None:
            chosen = _clicked_proposal(proposals, click)
            if chosen is None:
                return None, {"source": "generic proposal",
                              "reason": "No proposal enclosed click"}
            return [round(v) for v in chosen["box"]], {
                "source": "highest-scoring enclosing generic proposal",
                "model_score": chosen["detector_score"]}
        inputs = self.segment_processor(images=image, input_points=[[[click]]],
                                        input_labels=[[[1]]], return_tensors="pt").to(self.device)
        with torch.no_grad():
            output = self.segmenter(**inputs)
        masks = self.segment_processor.post_process_masks(
            output.pred_masks.cpu(), inputs["original_sizes"])[0][0]
        mask_idx = int(output.iou_scores[0, 0].argmax())
        mask = masks[mask_idx].numpy() > 0
        ys, xs = np.nonzero(mask)
        if len(xs) < 200 or not mask[click[1], click[0]]:
            return None, {"source": "SAM 2 click mask", "reason": "No usable mask at click"}
        return [int(xs.min()), int(ys.min()), int(xs.max() + 1), int(ys.max() + 1)], {
            "source": "SAM 2 highest predicted-IoU click mask",
            "model_score": float(output.iou_scores[0, 0, mask_idx]),
            "mask_area_px": int(len(xs))}

    def feature(self, images: list[Image.Image]) -> np.ndarray:
        pixels = self.feature_processor(images=images, return_tensors="pt")[
            "pixel_values"].to(self.device)
        with torch.no_grad():
            hidden = self.feature_model(pixel_values=pixels).last_hidden_state[:, 0]
            return torch.nn.functional.normalize(hidden, p=2, dim=-1).cpu().numpy()

    def detect(self, image: Image.Image) -> list[dict]:
        inputs = self.detector_processor(images=image, text=[PROMPT],
                                         return_tensors="pt").to(self.device)
        with torch.no_grad():
            output = self.detector(**inputs)
        result = self.detector_processor.post_process_grounded_object_detection(
            output, inputs.input_ids, threshold=.25, text_threshold=.25,
            target_sizes=[image.size[::-1]])[0]
        rows = []
        for box, score, label in zip(result["boxes"], result["scores"],
                                     result["text_labels"]):
            xyxy = box.cpu().tolist()
            if (xyxy[2] - xyxy[0]) * (xyxy[3] - xyxy[1]) < 200:
                continue
            rows.append({"box": xyxy, "detector_score": float(score),
                         "generic_label": label})
        return rows


def _ground_point(box: list[float], pose: tuple[np.ndarray, np.ndarray]) -> list[float] | None:
    """Use the bottom-center image ray and an assumed flat floor."""
    u, v = (box[0] + box[2]) / 2, box[3] - 2
    focal = HEIGHT / (2 * math.tan(math.radians(67) / 2))
    ray = pose[1] @ np.array([(u - WIDTH / 2) / focal,
                              -(v - HEIGHT / 2) / focal, -1])
    if ray[2] >= -1e-5:
        return None
    world = pose[0] - pose[0][2] / ray[2] * ray
    if np.linalg.norm(world[:2]) > 2:
        return None
    return world[:2].tolist()


def _iou(a: list[float], b: list[float] | None) -> float:
    if b is None:
        return 0.0
    x0, y0, x1, y1 = max(a[0], b[0]), max(a[1], b[1]), min(a[2], b[2]), min(a[3], b[3])
    intersection = max(0, x1 - x0) * max(0, y1 - y0)
    union = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - intersection
    return intersection / union if union else 0.0


def _clicked_proposal(proposals: list[dict], click: list[int]) -> dict | None:
    x, y = click
    enclosing = [p for p in proposals if p["box"][0] <= x <= p["box"][2]
                 and p["box"][1] <= y <= p["box"][3]]
    return max(enclosing, key=lambda p: p["detector_score"], default=None)


def _search_positions(target: str, absent: bool, seed: int) -> dict[str, tuple[float, float]]:
    others = [name for name in OBJECTS if name != target]
    rng = np.random.default_rng(seed)
    side = 1 if seed % 2 == 0 else -1
    positions = {target: (10, 0) if absent else
                 (.50 + float(rng.uniform(-.03, .03)), side * (.48 + float(rng.uniform(-.03, .03)))),
                 others[0]: (.47 + float(rng.uniform(-.03, .03)), float(rng.uniform(-.03, .03))),
                 others[1]: (.50 + float(rng.uniform(-.03, .03)), -side * .35)}
    return positions


def run_episode(scene: HeadScene, vision: Vision, target: str, absent: bool,
                seed: int, out: Path, teach_clicks: dict[str, list[int]],
                fault: str = "none") -> dict:
    if fault not in ("none", "drop_first_frame", "camera_unavailable",
                     "pose_unavailable", "name_audio_unavailable"):
        raise ValueError(f"Unknown sensor fault: {fault}")
    if fault == "name_audio_unavailable":
        return {"target": target, "target_absent": absent, "seed": seed,
                "sensor_fault": fault, "action": "ask_to_repeat_name",
                "reason": "No healthy name transcript was supplied for teaching",
                "observations": [], "correct_if_inspected": False}
    scene.place(TEACH_POSITIONS)
    scene.look(0, .75)
    teach_image = Image.fromarray(scene.image())
    teach_image.save(out / f"{target}-teach.png")
    click = teach_clicks[target]
    if not (0 <= click[0] < WIDTH and 0 <= click[1] < HEIGHT):
        raise ValueError(f"Teach click outside RGB frame: {target} {click}")
    all_teach_proposals = vision.detect(teach_image)
    selection, selection_info = vision.select_click(teach_image, click,
                                                    all_teach_proposals)
    if selection is None:
        return {"target": target, "target_absent": absent, "seed": seed,
                "teach_click_px": click, "teach_rgb_sha256": hashlib.sha256(
                    np.asarray(teach_image).tobytes()).hexdigest(),
                "teach_proposals": all_teach_proposals,
                "teach_selection_info": selection_info,
                "action": "ask_to_show_again", "reason": selection_info["reason"],
                "observations": [], "correct_if_inspected": False}
    exemplar = vision.feature([teach_image.crop(selection)])[0]
    teach_proposals = [p for p in all_teach_proposals
                       if _iou(p["box"], selection) < .25]
    teach_negative_scores = ([float(feature @ exemplar) for feature in vision.feature(
        [teach_image.crop(tuple(round(v) for v in p["box"]))
         for p in teach_proposals])] if teach_proposals else [])
    # The clicked proposal identifies which same-frame proposals are not the
    # taught object. This gives a local contrastive threshold, not a globally
    # calibrated probability of identity.
    acceptance = max(.5, max(teach_negative_scores, default=0) + .15)
    positions = _search_positions(target, absent, seed)
    rng = np.random.default_rng(seed + OBJECTS.index(target) * 41)
    headings = {name: float(rng.uniform(-.3, .3)) for name in OBJECTS}
    scene.place(positions, headings)
    observations: list[dict] = []
    pending_yaws = [0.0]
    observed_yaws = set()
    failed_first_frame = False
    while pending_yaws and len(observed_yaws) < 4:
        yaw = pending_yaws.pop(0)
        if round(yaw, 2) in observed_yaws:
            continue
        scene.look(yaw, .75)
        if fault == "camera_unavailable" or (fault == "drop_first_frame"
                                             and not failed_first_frame):
            observations.append({"yaw_rad": yaw, "pitch_rad": .75,
                                 "camera_healthy": False, "pose_healthy": fault != "pose_unavailable",
                                 "camera_xyz_m": None, "proposals": [],
                                 "requested_action": "retry_capture" if fault == "drop_first_frame"
                                 else "ask_camera_recovery"})
            if fault == "drop_first_frame":
                failed_first_frame = True
                pending_yaws.insert(0, yaw)
            if fault == "camera_unavailable":
                break
            continue
        observed_yaws.add(round(yaw, 2))
        image = Image.fromarray(scene.image())
        image.save(out / f"{target}-{'absent' if absent else 'moved'}-yaw{yaw:+.2f}.png")
        proposals = vision.detect(image)
        if proposals:
            crops = [image.crop(tuple(round(v) for v in p["box"])) for p in proposals]
            features = vision.feature(crops)
            pose = scene.camera_pose() if fault != "pose_unavailable" else None
            for proposal, feature in zip(proposals, features):
                box = proposal["box"]
                proposal["similarity"] = float(feature @ exemplar)
                proposal["floor_xy_m"] = _ground_point(box, pose) if pose is not None else None
                proposal["border_crop"] = box[0] < 8 or box[2] > WIDTH - 8
        observations.append({"yaw_rad": yaw, "pitch_rad": .75,
                             "camera_healthy": True, "pose_healthy": fault != "pose_unavailable",
                             "requested_action": "look_and_capture",
                             "camera_xyz_m": scene.camera_pose()[0].tolist()
                             if fault != "pose_unavailable" else None,
                             "proposals": proposals})
        if len(observed_yaws) == 1:
            # A truncated edge object causes an image-driven head turn to inspect it.
            borders = [p for p in proposals if p["border_crop"]]
            if borders:
                edge = max(borders, key=lambda p: p["detector_score"])
                pending_yaws.append(.7 if (edge["box"][0] + edge["box"][2]) / 2 < WIDTH / 2 else -.7)
            pending_yaws.extend([-.7, .7])

    candidates = [(obs["yaw_rad"], p) for obs in observations for p in obs["proposals"]
                  if not p["border_crop"] and p["floor_xy_m"] is not None]
    candidates.sort(key=lambda item: item[1]["similarity"], reverse=True)
    # Deliberately conservative, provisional acceptance; calibrated thresholds
    # require held-out ape-height captures and target-absent examples.
    best = candidates[0] if candidates else None
    runner_up = next((c for c in candidates[1:] if np.linalg.norm(
        np.subtract(c[1]["floor_xy_m"], best[1]["floor_xy_m"])) > .08), None) if best else None
    margin = best[1]["similarity"] - runner_up[1]["similarity"] if best and runner_up else None
    action = "inspect" if best and best[1]["similarity"] >= acceptance and (margin is None or margin >= .08) else "ask_or_continue_search"
    if fault == "camera_unavailable":
        action = "ask_camera_recovery"
    elif fault == "pose_unavailable":
        action = "ask_pose_recovery"
    verification = None
    if action == "inspect":
        yaw, proposal = best
        center = (proposal["box"][0] + proposal["box"][2]) / 2
        focal = HEIGHT / (2 * math.tan(math.radians(67) / 2))
        corrected_yaw = float(np.clip(yaw - math.atan((center - WIDTH / 2) / focal), -1.5, 1.5))
        scene.look(corrected_yaw, .75)
        image = Image.fromarray(scene.image())
        image.save(out / f"{target}-{'absent' if absent else 'moved'}-verify.png")
        found = vision.detect(image)
        if found:
            features = vision.feature([image.crop(tuple(round(v) for v in p["box"])) for p in found])
            scores = [float(f @ exemplar) for f in features]
            candidate = found[int(np.argmax(scores))]
            verification = {"yaw_rad": corrected_yaw, "box": candidate["box"],
                            "similarity": max(scores)}
            if max(scores) < acceptance:
                action = "ask_or_continue_search"
        else:
            action = "ask_or_continue_search"
    # Reveal simulator truth only after the search and verification decisions.
    scene.place(TEACH_POSITIONS)
    scene.look(0, .75)
    teach_truth_iou = _iou(selection, scene.evaluator_box(target))
    scene.place(positions, headings)
    for observation in observations:
        scene.look(observation["yaw_rad"], observation["pitch_rad"])
        truth_box = scene.evaluator_box(target)
        for proposal in observation["proposals"]:
            proposal["evaluator_target_iou"] = _iou(proposal["box"], truth_box)
    if verification is not None:
        scene.look(verification["yaw_rad"], .75)
        verification["evaluator_target_iou"] = _iou(
            verification["box"], scene.evaluator_box(target))
    return {"target": target, "target_absent": absent, "seed": seed,
            "sensor_fault": fault,
            "teach_click_px": click,
            "teach_rgb_sha256": hashlib.sha256(np.asarray(teach_image).tobytes()).hexdigest(),
            "teach_proposals": all_teach_proposals,
            "teach_selection_box": selection,
            "teach_selection_info": selection_info,
            "evaluator_teach_iou": teach_truth_iou,
            "teach_negative_similarities": teach_negative_scores,
            "acceptance_similarity": acceptance,
            "observations": observations,
            "candidate_count": len(candidates), "top_similarity": best[1]["similarity"] if best else None,
            "margin_to_different_position": margin, "action": action,
            "verification": verification,
            "correct_if_inspected": bool(action == "inspect" and verification
                                         and verification["evaluator_target_iou"] >= .25)}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--teach-clicks", type=Path, default=DEFAULT_CLICKS)
    parser.add_argument("--teach-selector", choices=["generic-proposal", "sam2-click"],
                        default="generic-proposal")
    parser.add_argument("--sensor-fault", choices=["none", "drop_first_frame",
                                                    "camera_unavailable", "pose_unavailable",
                                                    "name_audio_unavailable"],
                        default="none")
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    click_source = json.loads(args.teach_clicks.read_text())
    teach_clicks = click_source["clicks_px"]
    if set(teach_clicks) != set(OBJECTS):
        raise ValueError("Teach click file must have one click per object")
    scene, vision = HeadScene(), Vision(args.teach_selector)
    try:
        episodes = [run_episode(scene, vision, name, absent, args.seed, args.output,
                                teach_clicks, args.sensor_fault)
                    for name in OBJECTS for absent in (False, True)]
    finally:
        scene.close()
    result = {"scope": "50 cm static-base provisional head rig; textured HOPE object meshes; "
                       "synthetic lighting and floor; exact simulated camera pose. "
                       "Head yaw is image-driven for border inspection and final verification. "
                       "Teaching uses a supplied RGB click and generic object proposal; no physical person, "
                       "walking gait, contact or hardware camera. Sensor faults are injected status "
                       "ablations, not measured hardware failure rates.",
              "models": {"detector": str(DETECTOR), "retriever": str(RETRIEVER),
                         "generic_prompt": PROMPT, "teach_selector": args.teach_selector,
                         "click_segmenter": str(CLICK_SEGMENTER) if args.teach_selector == "sam2-click" else None},
              "teach_clicks_source": click_source.get("source"),
              "teach_clicks_file_sha256": hashlib.sha256(args.teach_clicks.read_bytes()).hexdigest(),
              "sensor_fault": args.sensor_fault,
              "head_camera": {"nominal_upright_eye_z_m": .446,
                              "vertical_fov_deg": 67, "image_size": [WIDTH, HEIGHT],
                              "search_pitch_rad": .75},
              "episodes": episodes}
    (args.output / "result.json").write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps([{"target": e["target"], "absent": e["target_absent"],
                       "capture_attempts": len(e["observations"]),
                       "healthy_views": sum(o["camera_healthy"] for o in e["observations"]),
                       "action": e["action"],
                       "correct_if_inspected": e["correct_if_inspected"]}
                      for e in episodes], indent=2))


if __name__ == "__main__":
    main()
