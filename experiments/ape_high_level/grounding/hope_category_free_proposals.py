"""Screen category-free SAM 2 search proposals on real HOPE RGB scenes.

The existing find loop asks Grounding DINO for bottle/can/jar/box proposals.
This diagnostic prompts the released SAM 2 model on fixed image grids instead,
then compares proposal coverage with the saved detector on the *same* HOPE
target-present class/scene trials. Object annotations are used only after the
image proposals are fixed. This tests proposal coverage, not identity or an
ape-camera field of view.
"""

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
from PIL import Image
import torch
from transformers import Sam2Model, Sam2Processor

from hope_image_find import _iou
from hope_retrieval import DEFAULT_DATA, DEFAULT_MESHES, _load_candidates


ROOT = Path(__file__).resolve().parents[3]
MODEL = ROOT / ".cache/ape-high-level/models/sam2-hiera-tiny"
BASELINE = ROOT / "artifacts/ape-grounding-hope-20261005/image-find-640.json"
WIDTH = 640
GRID_SIZES = ((10, 6), (20, 12))


def grid_points(width, height, nx, ny):
    return [(round(x), round(y))
            for y in np.linspace(20, height - 20, ny)
            for x in np.linspace(20, width - 20, nx)]


def proposals(processor, model, image, points, device):
    inputs = processor(
        images=image,
        input_points=[[[list(point)] for point in points]],
        input_labels=[[[1] for _ in points]],
        return_tensors="pt",
    ).to(device)
    with torch.no_grad():
        prediction = model(**inputs)
    masks = processor.post_process_masks(
        prediction.pred_masks.cpu(), inputs["original_sizes"]
    )[0]
    width, height = image.size
    found = []
    for i, (px, py) in enumerate(points):
        choice = int(prediction.iou_scores[0, i].argmax())
        mask = masks[i, choice].numpy() > 0
        ys, xs = np.nonzero(mask)
        area = len(xs)
        if area < 40 or area > 0.4 * width * height or not mask[py, px]:
            continue
        box = [int(xs.min()), int(ys.min()), int(xs.max() + 1), int(ys.max() + 1)]
        if (box[2] - box[0]) * (box[3] - box[1]) < 100:
            continue
        found.append({
            "box_640": box,
            "prompt_point_640": [px, py],
            "mask_area_px": area,
            "predicted_mask_iou": float(prediction.iou_scores[0, i, choice]),
        })
    # Identical or near-identical boxes from neighbouring points do not add
    # a distinct search candidate. This suppression uses no object label.
    ordered = sorted(found, key=lambda row: row["predicted_mask_iou"], reverse=True)
    unique = []
    for row in ordered:
        if not any(_iou(row["box_640"], keep["box_640"]) >= 0.85 for keep in unique):
            unique.append(row)
    return unique


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    baseline_bytes = BASELINE.read_bytes()
    baseline = json.loads(baseline_bytes)
    assert baseline["rgb_input_width"] == WIDTH
    scenes = baseline["dataset"]["scenes"][1:]
    truth, _ = _load_candidates(DEFAULT_DATA, DEFAULT_MESHES,
                                baseline["dataset"]["scenes"], [0])
    truth_by_scene_class = {}
    for row in truth:
        truth_by_scene_class.setdefault((row["scene"], row["class"]), []).append(row["bbox"])
    device = "mps" if torch.backends.mps.is_available() else "cpu"
    processor = Sam2Processor.from_pretrained(MODEL)
    model = Sam2Model.from_pretrained(MODEL).to(device).eval()
    all_proposals = {}
    image_hashes = {}
    for scene in scenes:
        path = DEFAULT_DATA / scene / "0000_rgb.jpg"
        image_hashes[scene] = hashlib.sha256(path.read_bytes()).hexdigest()
        original = Image.open(path).convert("RGB")
        image = original.resize(
            (WIDTH, round(original.height * WIDTH / original.width)),
            Image.Resampling.LANCZOS,
        )
        for nx, ny in GRID_SIZES:
            key = f"{nx}x{ny}"
            all_proposals.setdefault(key, {})[scene] = proposals(
                processor, model, image,
                grid_points(image.width, image.height, nx, ny), device,
            )
        print(scene, {key: len(rows[scene]) for key, rows in all_proposals.items()},
              flush=True)

    present = [row for row in baseline["trials"] if row["target_present"]]
    evaluated = {}
    for key, scene_proposals in all_proposals.items():
        rows = []
        for trial in present:
            scene = trial["find_scene"]
            targets = truth_by_scene_class[(scene, trial["taught_class"])]
            image_width = Image.open(DEFAULT_DATA / scene / "0000_rgb.jpg").width
            scale = image_width / WIDTH
            best = max((
                _iou([v * scale for v in p["box_640"]], target)
                for p in scene_proposals[scene] for target in targets
            ), default=0.0)
            rows.append({
                "name": trial["taught_class"], "scene": scene,
                "best_target_iou": best,
                "target_proposed_at_iou_025": best >= 0.25,
                "generic_detector_proposed_at_iou_025": trial["target_proposed_at_iou_025"],
            })
        evaluated[key] = {
            "present_trials": len(rows),
            "sam2_proposed": sum(r["target_proposed_at_iou_025"] for r in rows),
            "generic_detector_proposed": sum(r["generic_detector_proposed_at_iou_025"] for r in rows),
            "both": sum(r["target_proposed_at_iou_025"] and
                        r["generic_detector_proposed_at_iou_025"] for r in rows),
            "sam2_only": sum(r["target_proposed_at_iou_025"] and
                             not r["generic_detector_proposed_at_iou_025"] for r in rows),
            "generic_only": sum(not r["target_proposed_at_iou_025"] and
                                r["generic_detector_proposed_at_iou_025"] for r in rows),
            "proposals_per_scene": {scene: len(scene_proposals[scene]) for scene in scenes},
            "rows": rows,
        }
    report = {
        "scope": "Fixed-grid category-free SAM2 proposal coverage on real HOPE RGB; published pose/mesh boxes score afterward. Projected targets may be occluded. No identity, confidence, compute-on-ape or owner test.",
        "model_sha256": hashlib.sha256((MODEL / "model.safetensors").read_bytes()).hexdigest(),
        "baseline_sha256": hashlib.sha256(baseline_bytes).hexdigest(),
        "image_sha256": image_hashes,
        "model_device": device,
        "grid_sizes": [list(size) for size in GRID_SIZES],
        "filter": "Best predicted mask per positive grid point; area 40 px to 40% image; box >=100 px; suppress box IoU >=0.85.",
        "evaluation": evaluated,
        "proposals": all_proposals,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps({key: {metric: value for metric, value in row.items() if metric != "rows"}
                      for key, row in evaluated.items()}, indent=2))


if __name__ == "__main__":
    main()
