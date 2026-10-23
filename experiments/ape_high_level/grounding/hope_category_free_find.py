"""Rank category-free search masks against clicked HOPE teaching crops.

Uses saved grid-prompted SAM 2 proposals and the same real RGB scenes, manually
clicked teaching boxes and DINOv2 model as hope_click_find.py. Ground-truth
mesh projections score the resulting choice only after RGB ranking.
"""

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
from PIL import Image

from hope_image_find import _iou
from hope_retrieval import DEFAULT_DATA, DEFAULT_MESHES, DEFAULT_MODEL, _features, _load_candidates


ROOT = Path(__file__).resolve().parents[3]
PROPOSALS = ROOT / "artifacts/ape-grounding-hope-20261005/category-free-proposals.json"
CLICKS = ROOT / "artifacts/ape-grounding-hope-20261005/click-find-640.json"


def score(rows):
    present = [row for row in rows if row["target_present"]]
    return {
        "trials": len(rows),
        "present": len(present),
        "absent": len(rows) - len(present),
        "right_present_choice": sum(row["correct_at_iou_025"] for row in present),
        "descriptive_score_gates": {
            str(threshold): {
                "claims": sum(row["top_similarity"] >= threshold for row in rows),
                "right": sum(row["top_similarity"] >= threshold and
                             row["correct_at_iou_025"] for row in rows),
                "wrong_present": sum(row["top_similarity"] >= threshold and
                                     row["target_present"] and
                                     not row["correct_at_iou_025"] for row in rows),
                "wrong_absent": sum(row["top_similarity"] >= threshold and
                                    not row["target_present"] for row in rows),
            }
            for threshold in (0.6, 0.65, 0.7)
        },
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    proposal_bytes = PROPOSALS.read_bytes()
    click_bytes = CLICKS.read_bytes()
    proposal_result = json.loads(proposal_bytes)
    clicked_result = json.loads(click_bytes)
    scenes = list(proposal_result["image_sha256"])
    teach_scene = "scene_0000"
    images = {}
    image_scale = {}
    for scene in [teach_scene] + scenes:
        source = Image.open(DEFAULT_DATA / scene / "0000_rgb.jpg").convert("RGB")
        image_scale[scene] = source.width / 640
        images[scene] = source.resize(
            (640, round(source.height / image_scale[scene])), Image.Resampling.LANCZOS
        )
    teach = {row["name"]: row["sam2"]["box"]
             for row in clicked_result["teach_selections"]}
    names = list(teach)
    grids = list(proposal_result["proposals"])
    crop_index = {}
    crops = []
    for name in names:
        crop_index[("teach", name)] = len(crops)
        crops.append(images[teach_scene].crop(teach[name]))
    for grid in grids:
        for scene in scenes:
            for i, row in enumerate(proposal_result["proposals"][grid][scene]):
                crop_index[(grid, scene, i)] = len(crops)
                crops.append(images[scene].crop(row["box_640"]))
    features = _features(crops, DEFAULT_MODEL, batch_size=16)
    truth, _ = _load_candidates(DEFAULT_DATA, DEFAULT_MESHES,
                                [teach_scene] + scenes, [0])
    target_boxes = {}
    for row in truth:
        target_boxes.setdefault((row["scene"], row["class"]), []).append(row["bbox"])
    baseline_rows = [row for row in clicked_result["trials"] if row["method"] == "sam2"]
    evaluated = {}
    for grid in grids:
        rows = []
        for trial in baseline_rows:
            name, scene = trial["name"], trial["find_scene"]
            proposed = proposal_result["proposals"][grid][scene]
            query = features[crop_index[("teach", name)]]
            sims = features[[crop_index[(grid, scene, i)] for i in range(len(proposed))]] @ query
            chosen_index = int(np.argmax(sims))
            chosen = proposed[chosen_index]
            scale = image_scale[scene]
            box = [value * scale for value in chosen["box_640"]]
            targets = target_boxes.get((scene, name), [])
            overlap = max((_iou(box, target) for target in targets), default=0.0)
            rows.append({
                "name": name, "scene": scene,
                "target_present": trial["target_present"],
                "candidate_count": len(proposed),
                "chosen_box": box,
                "top_similarity": float(sims[chosen_index]),
                "chosen_target_iou": overlap,
                "correct_at_iou_025": overlap >= 0.25,
                "generic_search_correct_at_iou_025": trial["correct_at_iou_025"],
            })
        evaluated[grid] = {"summary": score(rows), "trials": rows}
    report = {
        "scope": "Real HOPE RGB click teach/find; category-free SAM2 grid search proposals versus saved category-prompted Grounding DINO search, same DINOv2 features. Class-level projected-box score; no ape camera or owner.",
        "proposal_sha256": hashlib.sha256(proposal_bytes).hexdigest(),
        "click_result_sha256": hashlib.sha256(click_bytes).hexdigest(),
        "retriever_model_sha256": hashlib.sha256((DEFAULT_MODEL / "model.safetensors").read_bytes()).hexdigest(),
        "generic_search_sam2_teach_summary": score(baseline_rows),
        "grid_search": evaluated,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps({"generic": report["generic_search_sam2_teach_summary"],
                      **{grid: data["summary"] for grid, data in evaluated.items()}},
                     indent=2))


if __name__ == "__main__":
    main()
