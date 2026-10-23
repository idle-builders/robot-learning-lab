"""Teach from supplied RGB clicks, then find objects in real HOPE images.

Compares a clicked generic proposal with a point-prompted SAM 2 mask for the
teach crop. The search images use the same saved RGB Grounding DINO proposals
as hope_image_find.py, so only teaching changes. Poses/meshes score results
after choices; they never select a teaching or search crop.
"""

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
from PIL import Image
import torch
from transformers import Sam2Model, Sam2Processor

from hope_image_find import _iou, _proposals
from hope_retrieval import DEFAULT_DATA, DEFAULT_MESHES, DEFAULT_MODEL, _features, _load_candidates


ROOT = Path(__file__).resolve().parents[3]
DEFAULT_CLICKS = Path(__file__).with_name("hope_teach_clicks.json")
DEFAULT_CORRECTIONS = Path(__file__).with_name("hope_correction_clicks.json")
DEFAULT_BASELINE = ROOT / "artifacts/ape-grounding-hope-20261005/image-find-640.json"
DEFAULT_SAM2 = ROOT / ".cache/ape-high-level/models/sam2-hiera-tiny"
DEFAULT_DETECTOR = ROOT / ".cache/ape-high-level/models/grounding-dino-tiny"
WIDTH = 640


def selected_generic(proposals, click):
    x, y = click
    enclosing = [proposal for proposal in proposals
                 if proposal["box"][0] <= x <= proposal["box"][2]
                 and proposal["box"][1] <= y <= proposal["box"][3]]
    return max(enclosing, key=lambda proposal: proposal["detector_score"], default=None)


def selected_sam2(processor, model, image, click, device):
    inputs = processor(images=image, input_points=[[[click]]],
                       input_labels=[[[1]]], return_tensors="pt").to(device)
    with torch.no_grad():
        output = model(**inputs)
    masks = processor.post_process_masks(output.pred_masks.cpu(),
                                         inputs["original_sizes"])[0][0]
    index = int(output.iou_scores[0, 0].argmax())
    mask = masks[index].numpy() > 0
    ys, xs = np.nonzero(mask)
    if len(xs) < 20 or not mask[click[1], click[0]]:
        return None
    return {"box": [int(xs.min()), int(ys.min()), int(xs.max() + 1), int(ys.max() + 1)],
            "mask_area_px": int(len(xs)),
            "predicted_mask_iou": float(output.iou_scores[0, 0, index])}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--clicks", type=Path, default=DEFAULT_CLICKS)
    parser.add_argument("--corrections", type=Path, default=DEFAULT_CORRECTIONS)
    parser.add_argument("--baseline", type=Path, default=DEFAULT_BASELINE)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    fixture = json.loads(args.clicks.read_text())
    baseline = json.loads(args.baseline.read_text())
    if baseline["rgb_input_width"] != WIDTH:
        raise ValueError("Expected 640-wide saved find proposals")
    scenes = baseline["dataset"]["scenes"]
    teach_path = DEFAULT_DATA / scenes[0] / "0000_rgb.jpg"
    if hashlib.sha256(teach_path.read_bytes()).hexdigest() != fixture["image_sha256"]:
        raise ValueError("Teach image does not match click fixture")
    original = Image.open(teach_path).convert("RGB")
    scale = WIDTH / original.width
    teach_image = original.resize((WIDTH, round(original.height * scale)),
                                  Image.Resampling.LANCZOS)
    clicks = {name: [round(x * scale), round(y * scale)]
              for name, (x, y) in fixture["clicks_px_original_1920x1080"].items()}
    # RGB teaching proposals; no mesh/pose is used for the selection.
    teach_proposals, _ = _proposals(DEFAULT_DATA, scenes[:1], DEFAULT_DETECTOR, WIDTH)
    teach_proposals = [{**proposal, "box": [v * scale for v in proposal["box"]]}
                       for proposal in teach_proposals]
    device = "mps" if torch.backends.mps.is_available() else "cpu"
    sam_processor = Sam2Processor.from_pretrained(DEFAULT_SAM2)
    sam_model = Sam2Model.from_pretrained(DEFAULT_SAM2).to(device).eval()
    selections = []
    for name, click in clicks.items():
        generic = selected_generic(teach_proposals, click)
        sam2 = selected_sam2(sam_processor, sam_model, teach_image, click, device)
        selections.append({"name": name, "click_px_640": click,
                           "generic": generic, "sam2": sam2})
        print(name, "generic", generic is not None, "sam2", sam2 is not None, flush=True)
    correction_fixture = json.loads(args.corrections.read_text())
    correction_scene = scenes[1]
    correction_path = DEFAULT_DATA / correction_scene / "0000_rgb.jpg"
    if hashlib.sha256(correction_path.read_bytes()).hexdigest() != correction_fixture["image_sha256"]:
        raise ValueError("Correction image does not match click fixture")
    correction_original = Image.open(correction_path).convert("RGB")
    correction_image = correction_original.resize(
        (WIDTH, round(correction_original.height * scale)), Image.Resampling.LANCZOS)
    correction_selections = []
    for name, original_click in correction_fixture["clicks_px_original_1920x1080"].items():
        click = [round(value * scale) for value in original_click]
        chosen = selected_sam2(sam_processor, sam_model, correction_image, click, device)
        correction_selections.append({"name": name, "click_px_640": click,
                                      "sam2": chosen})
        print("correction", name, "sam2", chosen is not None, flush=True)
    del sam_model
    if device == "mps":
        torch.mps.empty_cache()

    # Reuse identical saved search proposals so the comparison isolates the
    # teaching click/crop. Recreate crops from RGB, not annotation images.
    proposals = baseline["proposals"]
    scene_images = {}
    find_crops = []
    for proposal in proposals:
        scene = proposal["scene"]
        if scene not in scene_images:
            source = Image.open(DEFAULT_DATA / scene / "0000_rgb.jpg").convert("RGB")
            scene_images[scene] = source.resize((WIDTH, round(source.height * scale)),
                                                Image.Resampling.LANCZOS)
        box = [round(v * scale) for v in proposal["box"]]
        find_crops.append(scene_images[scene].crop(box))
    teach_crops = []
    feature_index = {}
    for method in ("generic", "sam2"):
        for selection in selections:
            chosen = selection[method]
            if chosen is None:
                continue
            feature_index[(method, selection["name"])] = len(teach_crops)
            teach_crops.append(teach_image.crop(chosen["box"]))
    correction_crops = []
    correction_feature_index = {}
    for selection in correction_selections:
        if selection["sam2"] is None:
            continue
        correction_feature_index[selection["name"]] = (
            len(teach_crops) + len(find_crops) + len(correction_crops))
        correction_crops.append(correction_image.crop(selection["sam2"]["box"]))
    features = _features(teach_crops + find_crops + correction_crops,
                         DEFAULT_MODEL, batch_size=16)
    oracle, _ = _load_candidates(DEFAULT_DATA, DEFAULT_MESHES, scenes, [0])
    teach_truth = {row["class"]: row["bbox"] for row in oracle
                   if row["scene"] == scenes[0] and row["class"] in clicks}
    for selection in selections:
        truth = teach_truth[selection["name"]]
        for method in ("generic", "sam2"):
            chosen = selection[method]
            if chosen is not None:
                chosen["teach_iou_with_projected_truth"] = _iou(
                    [v / scale for v in chosen["box"]], truth)
    for selection in correction_selections:
        chosen = selection["sam2"]
        if chosen is None:
            continue
        targets = [row["bbox"] for row in oracle if row["scene"] == correction_scene
                   and row["class"] == selection["name"]]
        chosen["correction_iou_with_projected_truth"] = max(
            (_iou([v / scale for v in chosen["box"]], truth) for truth in targets),
            default=0.0)
    ids_by_scene = {scene: [i for i, proposal in enumerate(proposals)
                            if proposal["scene"] == scene] for scene in scenes[1:]}
    trials = []
    for method in ("generic", "sam2"):
        for name in clicks:
            query_index = feature_index.get((method, name))
            for scene in scenes[1:]:
                targets = [row for row in oracle if row["scene"] == scene
                           and row["class"] == name]
                record = {"method": method, "name": name, "find_scene": scene,
                          "target_present": bool(targets),
                          "teach_selection_available": query_index is not None}
                if query_index is not None:
                    ids = ids_by_scene[scene]
                    sims = features[[len(teach_crops) + i for i in ids]] @ features[query_index]
                    order = np.argsort(-sims)
                    chosen_id = ids[int(order[0])]
                    record.update({
                        "candidate_count": len(ids),
                        "chosen_box": proposals[chosen_id]["box"],
                        "top_similarity": float(sims[order[0]]),
                        "chosen_target_iou": max((_iou(proposals[chosen_id]["box"], target["bbox"])
                                                  for target in targets), default=0.0),
                    })
                    record["correct_at_iou_025"] = record["chosen_target_iou"] >= 0.25
                trials.append(record)
    correction_trials = []
    for name, correction_index in correction_feature_index.items():
        original_index = feature_index.get(("sam2", name))
        if original_index is None:
            continue
        for method, query in (
            ("before", features[original_index]),
            ("replace", features[correction_index]),
            ("blend", (features[original_index] + features[correction_index]) /
             np.linalg.norm(features[original_index] + features[correction_index])),
        ):
            for scene in scenes[2:]:
                ids = ids_by_scene[scene]
                targets = [row for row in oracle if row["scene"] == scene
                           and row["class"] == name]
                sims = features[[len(teach_crops) + i for i in ids]] @ query
                order = np.argsort(-sims)
                chosen_id = ids[int(order[0])]
                chosen_iou = max((_iou(proposals[chosen_id]["box"], target["bbox"])
                                  for target in targets), default=0.0)
                correction_trials.append({
                    "name": name, "method": method, "find_scene": scene,
                    "target_present": bool(targets),
                    "chosen_box": proposals[chosen_id]["box"],
                    "top_similarity": float(sims[order[0]]),
                    "chosen_target_iou": chosen_iou,
                    "correct_at_iou_025": chosen_iou >= 0.25,
                })
    summary = {}
    for method in ("generic", "sam2"):
        chosen = [selection[method] for selection in selections]
        rows = [trial for trial in trials if trial["method"] == method]
        summary[method] = {
            "teach_selected": sum(item is not None for item in chosen),
            "teach_iou_025": sum(item is not None and item["teach_iou_with_projected_truth"] >= 0.25
                                  for item in chosen),
            "present_correct": sum(row.get("correct_at_iou_025", False) for row in rows
                                   if row["target_present"]),
            "present_trials": sum(row["target_present"] for row in rows),
            "absent_trials": sum(not row["target_present"] for row in rows),
        }
    correction_summary = {
        "selected": len(correction_feature_index),
        "iou_025": sum(selection["sam2"] is not None and
                       selection["sam2"]["correction_iou_with_projected_truth"] >= 0.25
                       for selection in correction_selections),
        "methods": {},
    }
    for method in ("before", "replace", "blend"):
        rows = [row for row in correction_trials if row["method"] == method]
        correction_summary["methods"][method] = {
            "trials": len(rows),
            "present": sum(row["target_present"] for row in rows),
            "present_correct": sum(row["correct_at_iou_025"] for row in rows),
            "absent": sum(not row["target_present"] for row in rows),
        }
    duplicate_present = sum(
        sum(row["scene"] == scene and row["class"] == name for row in oracle) > 1
        for name in clicks for scene in scenes[1:]
    )
    result = {"scope": (__doc__.strip() + " Scoring accepts any annotated instance of the "
                        "taught class, not proof of the same physical copy. Clicks were "
                        "development-selected from displayed RGB, not user-blinded."),
              "duplicate_class_present_trials": duplicate_present,
              "models_sha256": {
                  "sam2": hashlib.sha256((DEFAULT_SAM2 / "model.safetensors").read_bytes()).hexdigest(),
                  "teach_detector": hashlib.sha256((DEFAULT_DETECTOR / "model.safetensors").read_bytes()).hexdigest(),
                  "retriever": hashlib.sha256((DEFAULT_MODEL / "model.safetensors").read_bytes()).hexdigest(),
              },
              "click_fixture_sha256": hashlib.sha256(
        args.clicks.read_bytes()).hexdigest(), "baseline_result_sha256": hashlib.sha256(
        args.baseline.read_bytes()).hexdigest(), "correction_fixture_sha256": hashlib.sha256(
        args.corrections.read_bytes()).hexdigest(), "image_width": WIDTH,
        "teach_selections": selections, "correction_selections": correction_selections,
        "summary": summary, "correction_summary": correction_summary,
        "trials": trials, "correction_trials": correction_trials}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps({"teach": summary, "correction": correction_summary}, indent=2))


if __name__ == "__main__":
    main()
