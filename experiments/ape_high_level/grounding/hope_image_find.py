"""Find an owner-taught object in real HOPE images with image-based proposals.

The released pose/mesh annotations crop the *teach* view, standing in for a
person selecting the object. They are used again only to score the finds. The
find side receives RGB and generic object-word proposals, never pose or class.
HOPE is tabletop grocery clutter, not ape-height floor or social footage.
"""

from __future__ import annotations

import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path

import numpy as np
from PIL import Image
import torch
from transformers import AutoModelForZeroShotObjectDetection, AutoProcessor

from hope_retrieval import DEFAULT_DATA, DEFAULT_MESHES, DEFAULT_MODEL, _features, _load_candidates


ROOT = Path(__file__).resolve().parents[3]
DEFAULT_DETECTOR = ROOT / ".cache/ape-high-level/models/grounding-dino-tiny"
DETECTOR_REVISION = "a2bb814dd30d776dcf7e30523b00659f4f141c71"
PROMPT = ["bottle", "can", "box", "jar", "bag"]


def _iou(a: list[float], b: list[float]) -> float:
    x0, y0 = max(a[0], b[0]), max(a[1], b[1])
    x1, y1 = min(a[2], b[2]), min(a[3], b[3])
    intersection = max(0, x1 - x0) * max(0, y1 - y0)
    area_a = max(0, a[2] - a[0]) * max(0, a[3] - a[1])
    area_b = max(0, b[2] - b[0]) * max(0, b[3] - b[1])
    union = area_a + area_b - intersection
    return intersection / union if union > 0 else 0.0


def _proposals(data: Path, scenes: list[str], detector_dir: Path,
               image_width: int) -> tuple[list[dict], list[Image.Image]]:
    device = "mps" if torch.backends.mps.is_available() else "cpu"
    processor = AutoProcessor.from_pretrained(detector_dir)
    model = AutoModelForZeroShotObjectDetection.from_pretrained(detector_dir).to(device).eval()
    rows, crops = [], []
    for scene in scenes:
        source = Image.open(data / scene / "0000_rgb.jpg").convert("RGB")
        scale = source.width / image_width
        image = source.resize((image_width, round(source.height / scale)), Image.Resampling.LANCZOS)
        inputs = processor(images=image, text=[PROMPT], return_tensors="pt").to(device)
        with torch.no_grad():
            output = model(**inputs)
        result = processor.post_process_grounded_object_detection(
            output, inputs.input_ids, threshold=0.25, text_threshold=0.25,
            target_sizes=[image.size[::-1]],
        )[0]
        found = 0
        for box, score, label in zip(result["boxes"], result["scores"], result["text_labels"]):
            xyxy = box.cpu().tolist()
            area_fraction = ((xyxy[2] - xyxy[0]) * (xyxy[3] - xyxy[1])) / (image.width * image.height)
            if not 0.0001 <= area_fraction <= 0.4:
                continue
            crop = image.crop(tuple(int(round(x)) for x in xyxy))
            if min(crop.size) < 8:
                continue
            rows.append({"scene": scene, "box": [x * scale for x in xyxy],
                         "detector_label": label,
                         "detector_score": float(score.item())})
            crops.append(crop)
            found += 1
        print(f"{scene}: {found} RGB proposals", flush=True)
    del model
    if device == "mps":
        torch.mps.empty_cache()
    return rows, crops


def _match_scene(query_id: int, proposal_ids: list[int], scene: str,
                 oracle: list[dict],
                 proposals: list[dict], features: np.ndarray,
                 teach_count: int, correction_proposal_id: int | None = None) -> dict:
    if not proposal_ids:
        return {"candidate_count": 0, "chosen_box": None,
                "target_present": any(row["scene"] == scene
                                      and row["class"] == oracle[query_id]["class"]
                                      for row in oracle),
                "correct_at_iou_025": False,
                "target_proposed_at_iou_025": False}
    query = features[query_id]
    if correction_proposal_id is not None:
        query = query + features[teach_count + correction_proposal_id]
        query = query / np.linalg.norm(query)
    sims = features[[teach_count + i for i in proposal_ids]] @ query
    order = np.argsort(-sims)
    best_id = proposal_ids[int(order[0])]
    target_name = oracle[query_id]["class"]
    targets = [row for row in oracle if row["scene"] == scene and row["class"] == target_name]
    chosen_box = proposals[best_id]["box"]
    best_iou = max((_iou(chosen_box, row["bbox"]) for row in targets), default=0.0)
    return {"candidate_count": len(proposal_ids), "chosen_box": chosen_box,
            "chosen_label": proposals[best_id]["detector_label"],
            "top_similarity": float(sims[order[0]]),
            "margin_to_second": float(sims[order[0]] - sims[order[1]]) if len(order) > 1 else None,
            "target_present": bool(targets), "chosen_target_iou": best_iou,
            "correct_at_iou_025": best_iou >= 0.25,
            "target_proposed_at_iou_025": any(
                _iou(proposals[i]["box"], target["bbox"]) >= 0.25
                for i in proposal_ids for target in targets),
            "max_proposal_target_iou": max((
                _iou(proposals[i]["box"], target["bbox"])
                for i in proposal_ids for target in targets), default=0.0)}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, default=DEFAULT_DATA)
    parser.add_argument("--meshes", type=Path, default=DEFAULT_MESHES)
    parser.add_argument("--retriever", type=Path, default=DEFAULT_MODEL)
    parser.add_argument("--detector", type=Path, default=DEFAULT_DETECTOR)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--image-width", type=int, default=1920,
                        help="Input width before RGB detection and retrieval; original is 1920")
    args = parser.parse_args()

    scenes = sorted(p.name for p in args.data.glob("scene_*"))
    if len(scenes) < 2:
        raise ValueError("Expected at least two HOPE scenes")
    oracle, teach_crops = _load_candidates(args.data, args.meshes, scenes, [0])
    teach_count = sum(row["scene"] == scenes[0] for row in oracle)
    teach_image = Image.open(args.data / scenes[0] / "0000_rgb.jpg").convert("RGB")
    teach_scale = args.image_width / teach_image.width
    teach_image = teach_image.resize((args.image_width, round(teach_image.height * teach_scale)),
                                    Image.Resampling.LANCZOS)
    teach_crops = [teach_image.crop(tuple(round(v * teach_scale) for v in oracle[i]["bbox"]))
                   for i in range(teach_count)]
    query_ids = list(range(teach_count))
    class_counts = Counter(oracle[i]["class"] for i in query_ids)
    query_ids = [i for i in query_ids if class_counts[oracle[i]["class"]] == 1]
    proposals, find_crops = _proposals(args.data, scenes[1:], args.detector,
                                      args.image_width)
    all_crops = teach_crops + find_crops
    features = _features(all_crops, args.retriever, batch_size=16)
    proposals_by_scene = {scene: [i for i, row in enumerate(proposals)
                                  if row["scene"] == scene] for scene in scenes[1:]}
    trials = []
    for query_id in query_ids:
        for scene in scenes[1:]:
            ids = proposals_by_scene[scene]
            result = _match_scene(query_id, ids, scene, oracle, proposals, features,
                                  teach_count)
            trials.append({"taught_class": oracle[query_id]["class"],
                           "teach_scene": scenes[0], "find_scene": scene, **result})
    correction_trials = []
    correction_scene = scenes[1]
    correction_ids = proposals_by_scene[correction_scene]
    for query_id in query_ids:
        target_name = oracle[query_id]["class"]
        targets = [row for row in oracle if row["scene"] == correction_scene
                   and row["class"] == target_name]
        ranked = sorted(((max(_iou(proposals[i]["box"], t["bbox"]) for t in targets), i)
                         for i in correction_ids), reverse=True) if targets else []
        if not ranked or ranked[0][0] < 0.25:
            continue
        # Evaluator picks the correct proposal in the correction scene. This
        # emulates a person tapping the item after the ape makes a mistake.
        correction_id = ranked[0][1]
        for scene in scenes[2:]:
            ids = proposals_by_scene[scene]
            before = _match_scene(query_id, ids, scene, oracle, proposals,
                                  features, teach_count)
            after = _match_scene(query_id, ids, scene, oracle, proposals,
                                 features, teach_count, correction_id)
            correction_trials.append({"taught_class": target_name,
                                      "find_scene": scene,
                                      "correction_scene": correction_scene,
                                      "correction_box": proposals[correction_id]["box"],
                                      "before": before, "after": after})
    present = [t for t in trials if t["target_present"]]
    absent = [t for t in trials if not t["target_present"]]
    result = {
        "scope": ("Real HOPE tabletop RGB; evaluator-only annotated teach crop and scoring boxes. "
                  "Find-side proposals come from RGB generic words; visual ranking uses one taught crop. "
                  "No user selection UI, robot camera movement, floor-height imagery or closed-loop control."),
        "dataset": {"name": "NVIDIA HOPE-Image validation", "scenes": scenes,
                    "archive_md5": "8175c04733f266c6aa98091fa0f1c879"},
        "rgb_input_width": args.image_width,
        "detector": {"name": "IDEA-Research/grounding-dino-tiny", "revision": DETECTOR_REVISION,
                     "weights_sha256": hashlib.sha256((args.detector / "model.safetensors").read_bytes()).hexdigest(),
                     "prompt": PROMPT, "score_threshold": 0.25,
                     "proposal_count": len(proposals)},
        "retriever": {"name": "facebook/dinov2-small",
                      "weights_sha256": hashlib.sha256((args.retriever / "model.safetensors").read_bytes()).hexdigest()},
        "score_iou_threshold": 0.25,
        "summary": {"taught_unique_objects": len(query_ids), "trials": len(trials),
                    "present_trials": len(present), "absent_trials": len(absent),
                    "present_target_proposed": sum(t.get("target_proposed_at_iou_025", False) for t in present),
                    "present_correct_choice": sum(t.get("correct_at_iou_025", False) for t in present),
                    "correction_cohort_trials": len(correction_trials),
                    "correction_cohort_present": sum(t["before"]["target_present"] for t in correction_trials),
                    "before_correction_correct": sum(t["before"]["correct_at_iou_025"] for t in correction_trials),
                    "after_correction_correct": sum(t["after"]["correct_at_iou_025"] for t in correction_trials)},
        "trials": trials,
        "correction_trials": correction_trials,
        "proposals": proposals,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result["summary"], indent=2))


if __name__ == "__main__":
    main()
