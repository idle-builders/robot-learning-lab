"""Check rendered human visibility, person proposals, and depth in both views.

This diagnostic does not assign owner identity or evaluate an interaction.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import time

import cv2
import numpy as np
from PIL import Image, ImageDraw
import torch
from transformers import AutoModelForZeroShotObjectDetection, AutoProcessor


ROOT = Path(__file__).resolve().parents[3]
DETECTOR = ROOT / ".cache/ape-high-level/models/grounding-dino-tiny"


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--evidence", type=Path, required=True)
    args = parser.parse_args()
    evidence = args.evidence.resolve()
    device = "mps" if torch.backends.mps.is_available() else "cpu"
    processor = AutoProcessor.from_pretrained(DETECTOR)
    model = AutoModelForZeroShotObjectDetection.from_pretrained(DETECTOR).to(device).eval()
    views = []
    for fov in (70, 90):
        name = f"ape_social_{fov}deg"
        camera = evidence / "capture" / name
        rgb_path = camera / "rgb/rgb_00000.png"
        depth_path = camera / "distance_to_camera/distance_to_camera_00000.npy"
        video_path = evidence / f"{name}.mp4"
        image = Image.open(rgb_path).convert("RGB")
        rgb = np.asarray(image)
        depth = np.load(depth_path)
        assert depth.shape == rgb.shape[:2], (name, depth.shape, rgb.shape)
        start = time.perf_counter()
        tensors = processor(images=image, text=["person ."], return_tensors="pt").to(device)
        with torch.no_grad():
            output = model(**tensors)
        found = processor.post_process_grounded_object_detection(
            output, tensors.input_ids, threshold=.25, text_threshold=.25,
            target_sizes=[rgb.shape[:2]])[0]
        inference_s = time.perf_counter() - start
        valid = np.isfinite(depth) & (depth > 0) & (depth < 10)
        detections = []
        annotated = image.copy()
        draw = ImageDraw.Draw(annotated)
        for label, score, box in zip(found["text_labels"], found["scores"], found["boxes"]):
            x1, y1, x2, y2 = [float(v) for v in box]
            x1i, x2i = max(0, int(x1)), min(rgb.shape[1], int(x2) + 1)
            y1i, y2i = max(0, int(y1)), min(rgb.shape[0], int(y2) + 1)
            roi = depth[y1i:y2i, x1i:x2i]
            good = valid[y1i:y2i, x1i:x2i]
            detections.append({"label": str(label), "score": float(score),
                "box_xyxy": [x1, y1, x2, y2],
                "box_depth_valid_fraction": float(good.mean()) if good.size else 0,
                "box_depth_median_m": float(np.median(roi[good])) if good.any() else None})
            draw.rectangle((x1, y1, x2, y2), outline="lime", width=3)
            draw.text((x1, max(0, y1 - 14)), f"{label} {float(score):.2f}", fill="lime")
        annotated_path = evidence / f"{name}-person-proposals.png"
        annotated.save(annotated_path)
        capture = cv2.VideoCapture(str(video_path))
        frames = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
        fps = float(capture.get(cv2.CAP_PROP_FPS))
        okay, first = capture.read()
        last = first
        while True:
            success, frame = capture.read()
            if not success:
                break
            last = frame
        capture.release()
        frame_change = (float(np.mean(np.abs(first.astype(float) - last.astype(float))))
                        if okay and last is not None else None)
        views.append({"view": name, "rgb_sha256": sha(rgb_path),
            "video_sha256": sha(video_path), "video_frames": frames, "video_fps": fps,
            "first_last_frame_mean_abs_rgb_difference": frame_change,
            "depth_shape": list(depth.shape), "depth_finite_0_10m_fraction": float(valid.mean()),
            "person_inference_wall_s": inference_s, "detections": detections,
            "annotated_image": str(annotated_path)})
    report = {"scope": "one first frame per fixed low camera, no identity or tracking; actor model reused",
              "scene_info_sha256": sha(evidence / "scene-info.json"), "views": views}
    (evidence / "first-people-analysis.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
