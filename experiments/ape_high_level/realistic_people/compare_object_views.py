"""Frozen 70/90-degree object-view screen on saved Isaac RGB-D packets.

This measures candidate visibility; it does not close control on either view.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path

import cv2
import numpy as np
from PIL import Image
import torch
from transformers import (AutoImageProcessor, AutoModel,
                          AutoModelForZeroShotObjectDetection, AutoProcessor)

from social_show_actor import PROMPT, local_point


ROOT = Path(__file__).resolve().parents[3]


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("runs", nargs="+", type=Path)
    parser.add_argument("--frames", nargs="+", type=int, default=[13, 15])
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    detector_path = ROOT / ".cache/ape-high-level/models/grounding-dino-tiny-compatible"
    features_path = ROOT / ".cache/ape-high-level/models/dinov2-small"
    torch.set_num_threads(4)
    detector_processor = AutoProcessor.from_pretrained(detector_path)
    detector = AutoModelForZeroShotObjectDetection.from_pretrained(detector_path).eval()
    feature_processor = AutoImageProcessor.from_pretrained(features_path)
    feature_model = AutoModel.from_pretrained(features_path).eval()
    taught = Image.open(ROOT / "artifacts/ape-robocasa-kitchen-pilot/taught-rgb-crop.png")

    def feature(crops: list[Image.Image]) -> np.ndarray:
        pixels = feature_processor(images=crops,
                                   return_tensors="pt")["pixel_values"]
        with torch.no_grad():
            hidden = feature_model(pixel_values=pixels).last_hidden_state[:, 0]
            unit = torch.nn.functional.normalize(hidden, p=2, dim=-1)
        return unit.numpy()

    taught_feature = feature([taught])[0]
    rows = []
    for run in args.runs:
        evidence = run / "evidence"
        for fov in (70, 90):
            depth_path = evidence / f"ape_social_{fov}deg-depth-sequence-stride3.npz"
            video_path = evidence / f"ape_social_{fov}deg.mp4"
            depth = np.load(depth_path)["depth_m"]
            capture = cv2.VideoCapture(str(video_path))
            wanted = set(args.frames)
            images = {}
            for index in range(max(wanted) + 1):
                okay, frame = capture.read()
                if not okay:
                    raise RuntimeError(f"{run}: missing video frame {index}")
                if index in wanted:
                    images[index] = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
            capture.release()
            for index in args.frames:
                raw = (evidence / "capture" / f"ape_social_{fov}deg" /
                       "rgb" / f"rgb_{index:05d}.png")
                image = Image.open(raw).convert("RGB") if raw.exists() else images[index]
                inputs = detector_processor(images=image, text=[PROMPT],
                                            return_tensors="pt")
                with torch.no_grad():
                    output = detector(**inputs)
                found = detector_processor.post_process_grounded_object_detection(
                    output, inputs.input_ids, threshold=.25, text_threshold=.25,
                    target_sizes=[(image.height, image.width)])[0]
                proposals, crops = [], []
                for box, score, label in zip(found["boxes"], found["scores"],
                                             found["text_labels"]):
                    xyxy = [float(value) for value in box]
                    x0, y0, x1, y1 = xyxy
                    crop = image.crop((max(0, int(x0)), max(0, int(y0)),
                                       min(image.width, math.ceil(x1)),
                                       min(image.height, math.ceil(y1))))
                    if crop.width < 8 or crop.height < 8:
                        continue
                    proposals.append({"box_xyxy": xyxy,
                        "box_area_px": (x1 - x0) * (y1 - y0),
                        "detector_score": float(score), "label": str(label),
                        "central_depth_stride3": local_point(
                            depth[index], xyxy, image.size, fov)})
                    crops.append(crop)
                if crops:
                    similarities = feature(crops) @ taught_feature
                    for proposal, similarity in zip(proposals, similarities):
                        proposal["taught_similarity"] = float(similarity)
                rows.append({"run": run.name, "frame": index, "fov_deg": fov,
                    "rgb_source": "raw_png" if raw.exists() else "h264_video",
                    "rgb_source_sha256": sha256(raw if raw.exists() else video_path),
                    "depth_sequence_sha256": sha256(depth_path),
                    "proposals": sorted(proposals,
                        key=lambda row: row["taught_similarity"], reverse=True)})
    report = {"scope": "same-pose frozen 70/90 RGB-D comparison; no control feedback; "
               "central depth is stride-3 sampled so pixel counts are not full-resolution gates",
              "detector_threshold": .25,
              "model_sha256": {"detector": sha256(detector_path / "model.safetensors"),
                               "features": sha256(features_path / "model.safetensors")},
              "rows": rows}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps([{"run": row["run"], "frame": row["frame"],
        "fov_deg": row["fov_deg"], "proposal_count": len(row["proposals"]),
        "best": row["proposals"][0] if row["proposals"] else None}
        for row in rows], indent=2))


if __name__ == "__main__":
    main()
