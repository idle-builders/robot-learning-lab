"""Real-image diagnostic for re-finding a taught object using oracle crops.

HOPE's released object poses and meshes provide crop proposals here. Those are
evaluator inputs, not a deployable detector. This isolates visual retrieval on
real clutter and changed lighting before adding image-based proposals.
"""

from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import hashlib
import json
from pathlib import Path

import numpy as np
from PIL import Image
import torch
from transformers import AutoImageProcessor, AutoModel


ROOT = Path(__file__).resolve().parents[3]
DEFAULT_DATA = ROOT / ".cache/ape-high-level/hope_valid"
DEFAULT_MESHES = ROOT / ".cache/ape-high-level/hope_meshes_eval"
DEFAULT_MODEL = ROOT / ".cache/ape-high-level/models/dinov2-small"
MODEL_REVISION = "ed25f3a31f01632728cabb09d1542f84ab7b0056"
DATASET_MD5 = "8175c04733f266c6aa98091fa0f1c879"


def _vertices(path: Path) -> np.ndarray:
    points = []
    with path.open() as source:
        for line in source:
            if line.startswith("v "):
                points.append([float(n) for n in line.split()[1:4]])
    return np.asarray(points, dtype=np.float32)


def _project_box(vertices: np.ndarray, pose: np.ndarray, intrinsics: np.ndarray,
                 width: int, height: int) -> tuple[int, int, int, int] | None:
    camera = vertices @ pose[:3, :3].T + pose[:3, 3]
    camera = camera[camera[:, 2] > 0.01]
    if len(camera) == 0:
        return None
    u = intrinsics[0, 0] * camera[:, 0] / camera[:, 2] + intrinsics[0, 2]
    v = intrinsics[1, 1] * camera[:, 1] / camera[:, 2] + intrinsics[1, 2]
    x0, y0, x1, y1 = float(u.min()), float(v.min()), float(u.max()), float(v.max())
    pad = 0.04 * max(x1 - x0, y1 - y0)
    x0 = max(0, int(np.floor(x0 - pad)))
    y0 = max(0, int(np.floor(y0 - pad)))
    x1 = min(width, int(np.ceil(x1 + pad)))
    y1 = min(height, int(np.ceil(y1 + pad)))
    if x1 - x0 < 20 or y1 - y0 < 20 or (x1 - x0) * (y1 - y0) < 900:
        return None
    return x0, y0, x1, y1


def _load_candidates(data_dir: Path, mesh_dir: Path, scenes: list[str],
                     lighting: list[int]) -> tuple[list[dict], list[Image.Image]]:
    vertices = {}
    candidates, crops = [], []
    for scene in scenes:
        for light in lighting:
            source = data_dir / scene / f"{light:04d}"
            if not source.with_suffix(".json").exists():
                continue
            annotation = json.loads(source.with_suffix(".json").read_text())
            image = Image.open(source.parent / f"{source.name}_rgb.jpg").convert("RGB")
            width, height = image.size
            k = np.asarray(annotation["camera"]["intrinsics"], dtype=np.float32)
            for index, item in enumerate(annotation["objects"]):
                name = item["class"]
                if name not in vertices:
                    vertices[name] = _vertices(mesh_dir / f"{name}.obj")
                box = _project_box(vertices[name], np.asarray(item["pose"], dtype=np.float32),
                                   k, width, height)
                if box is None:
                    continue
                crop = image.crop(box)
                candidates.append({"scene": scene, "light": light,
                                   "object_index": index, "class": name,
                                   "bbox": box, "crop_pixels": crop.width * crop.height})
                crops.append(crop)
    return candidates, crops


def _features(crops: list[Image.Image], model_dir: Path, batch_size: int) -> np.ndarray:
    device = "mps" if torch.backends.mps.is_available() else "cpu"
    processor = AutoImageProcessor.from_pretrained(model_dir)
    model = AutoModel.from_pretrained(model_dir).to(device).eval()
    output = []
    with torch.no_grad():
        for start in range(0, len(crops), batch_size):
            batch = crops[start:start + batch_size]
            pixels = processor(images=batch, return_tensors="pt")["pixel_values"].to(device)
            hidden = model(pixel_values=pixels).last_hidden_state[:, 0, :]
            unit = torch.nn.functional.normalize(hidden, p=2, dim=-1)
            output.append(unit.cpu().numpy())
    return np.concatenate(output, axis=0)


def _rank(query_id: int, candidate_ids: list[int], features: np.ndarray) -> tuple[int, float, float]:
    similarities = features[candidate_ids] @ features[query_id]
    order = np.argsort(-similarities)
    best = candidate_ids[int(order[0])]
    margin = float(similarities[order[0]] - similarities[order[1]]) if len(order) > 1 else float("nan")
    return best, float(similarities[order[0]]), margin


def evaluate(candidates: list[dict], features: np.ndarray) -> dict:
    by_view = defaultdict(list)
    for i, row in enumerate(candidates):
        by_view[(row["scene"], row["light"])].append(i)
    scenes = sorted({row["scene"] for row in candidates})
    cross, light = [], []
    for source_scene in scenes:
        query_ids = by_view[(source_scene, 0)]
        query_counts = Counter(candidates[i]["class"] for i in query_ids)
        for target_scene in scenes:
            if target_scene == source_scene:
                continue
            target_ids = by_view[(target_scene, 0)]
            target_counts = Counter(candidates[i]["class"] for i in target_ids)
            for query in query_ids:
                name = candidates[query]["class"]
                # A duplicated class cannot establish which instance was taught.
                if query_counts[name] != 1 or target_counts[name] != 1:
                    continue
                best, score, margin = _rank(query, target_ids, features)
                cross.append({"teach_scene": source_scene, "find_scene": target_scene,
                              "requested": name, "chosen": candidates[best]["class"],
                              "correct": candidates[best]["class"] == name,
                              "top_score": score, "top_margin": margin,
                              "distractors": len(target_ids) - 1})
    source_scene = scenes[0]
    query_ids = by_view[(source_scene, 0)]
    for lighting in range(1, 5):
        target_ids = by_view[(source_scene, lighting)]
        for query in query_ids:
            matches = [i for i in target_ids if candidates[i]["object_index"] == candidates[query]["object_index"]]
            if len(matches) != 1:
                continue
            best, score, margin = _rank(query, target_ids, features)
            light.append({"lighting": lighting, "requested": candidates[query]["class"],
                          "object_index": candidates[query]["object_index"],
                          "chosen_index": candidates[best]["object_index"],
                          "correct": candidates[best]["object_index"] == candidates[query]["object_index"],
                          "top_score": score, "top_margin": margin,
                          "distractors": len(target_ids) - 1})
    return {
        "cross_scene_unique_class": {
            "correct": sum(r["correct"] for r in cross), "trials": len(cross),
            "rows": cross},
        "same_scene_changed_lighting_instance": {
            "correct": sum(r["correct"] for r in light), "trials": len(light),
            "rows": light},
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, default=DEFAULT_DATA)
    parser.add_argument("--meshes", type=Path, default=DEFAULT_MESHES)
    parser.add_argument("--model", type=Path, default=DEFAULT_MODEL)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--batch-size", type=int, default=16)
    args = parser.parse_args()
    scenes = sorted(p.name for p in args.data.glob("scene_*"))
    if len(scenes) < 2:
        raise ValueError("Expected at least two HOPE scenes")
    candidates, crops = _load_candidates(args.data, args.meshes, scenes,
                                         [0, 1, 2, 3, 4])
    features = _features(crops, args.model, args.batch_size)
    result = evaluate(candidates, features)
    result["scope"] = ("Real HOPE tabletop RGB with evaluator-only 6D poses and meshes "
                       "used to make oracle crops. This is re-identification, not "
                       "image-based detection, floor-height sensing or closed-loop control.")
    result["dataset"] = {"name": "NVIDIA HOPE-Image validation", "archive_md5": DATASET_MD5,
                         "scene_count": len(scenes), "candidate_crops": len(candidates)}
    result["model"] = {"name": "facebook/dinov2-small", "revision": MODEL_REVISION,
                       "weights_sha256": hashlib.sha256((args.model / "model.safetensors").read_bytes()).hexdigest()}
    result["crop_rule"] = "Projected released mesh/pose box, 4% padding, minimum 20x20 and 900 pixels"
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n")
    for key in ("cross_scene_unique_class", "same_scene_changed_lighting_instance"):
        entry = result[key]
        print(f"{key}: {entry['correct']}/{entry['trials']}")
    print(f"candidate crops: {len(candidates)}")


if __name__ == "__main__":
    main()
