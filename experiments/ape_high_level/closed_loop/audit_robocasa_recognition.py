"""Run the existing RGB proposal/retrieval stack on RoboCasa ape-height frames.

The actor gets only exposed RGB images, an owner click in the first frame, and
its taught crop. Simulator segmentation in the scene report is read after
inference solely to score whether a selected proposal covered that instance.
This is a perception transfer probe, not a navigation or full owner episode.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
import sys

import imageio.v2 as imageio
import numpy as np
from PIL import Image, ImageDraw


ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / "experiments/ape_high_level/teach_find"))
from active_scan import Vision  # noqa: E402

OUT = ROOT / "artifacts/ape-robocasa-kitchen-pilot"
SAMPLES = (*range(0, 48, 4), 35, 47)
OWNER_CLICK = [499, 249]  # an owner-supplied click on the visible can in step 0
SIMILARITY_THRESHOLD = 0.35  # frozen from the existing Cove-50 loop


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def iou(a: list[float], b: list[int] | None) -> float:
    if b is None:
        return 0.0
    x0, y0, x1, y1 = max(a[0], b[0]), max(a[1], b[1]), min(a[2], b[2]), min(a[3], b[3])
    intersection = max(0, x1 - x0) * max(0, y1 - y0)
    union = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - intersection
    return intersection / union if union else 0.0


def annotation(image: Image.Image, camera: str, step: int, row: dict,
               gt_box: list[int] | None) -> np.ndarray:
    canvas = Image.new("RGB", (640, 520), "#12171a")
    canvas.paste(image, (0, 40))
    draw = ImageDraw.Draw(canvas)
    status = row["outcome"]
    draw.text((12, 12), f"{camera} step {step:02d}: {status}", fill="white")
    if gt_box is not None:
        draw.rectangle((gt_box[0], gt_box[1] + 40, gt_box[2], gt_box[3] + 40),
                       outline="#36bbff", width=2)
    if row["selected_box"] is not None:
        box = row["selected_box"]
        color = "#39e47c" if status == "correct_visible" else "#ff563f"
        draw.rectangle((box[0], box[1] + 40, box[2], box[3] + 40),
                       outline=color, width=3)
    return np.asarray(canvas)


def main() -> None:
    survey = json.loads((OUT / "report.json").read_text())
    vision = Vision(teach_selector="generic-proposal")
    teach_image = Image.open(OUT / "step00-head.png").convert("RGB")
    teach_proposals = vision.detect(teach_image)
    taught_box, click_detail = vision.select_click(teach_image, OWNER_CLICK, teach_proposals)
    if taught_box is None:
        raise RuntimeError(f"Owner click has no generic proposal: {click_detail}")
    taught_crop = teach_image.crop(taught_box)
    taught_crop.save(OUT / "taught-rgb-crop.png")
    exemplar = vision.feature([taught_crop])[0]
    rows: list[dict] = []
    annotated: dict[str, list[np.ndarray]] = {"head": [], "chest": []}
    for step in SAMPLES:
        for camera in ("head", "chest"):
            image_path = OUT / f"step{step:02d}-{camera}.png"
            image = Image.open(image_path).convert("RGB")
            proposals = vision.detect(image)
            scores = (vision.feature([image.crop(p["box"]) for p in proposals]) @ exemplar
                      if proposals else np.empty(0))
            ranked = sorted(zip(proposals, scores), key=lambda pair: float(pair[1]), reverse=True)
            selected = ranked[0] if ranked and float(ranked[0][1]) >= SIMILARITY_THRESHOLD else None
            gt = survey["samples"][step]
            gt_pixels = gt["visible_pixels"][camera]["shown_can"]
            gt_box = gt["evaluator_boxes"][camera]["shown_can"]
            visible = gt_pixels >= 100
            selected_box = selected[0]["box"] if selected is not None else None
            overlap = iou(selected_box, gt_box) if selected_box is not None else 0.0
            outcome = ("correct_visible" if visible and selected_box is not None and overlap >= 0.3
                       else "wrong_visible" if visible and selected_box is not None
                       else "missed_visible" if visible
                       else "false_find_hidden" if selected_box is not None
                       else "abstain_hidden")
            row = {"step": step, "camera": camera, "image_sha256": sha256(image_path),
                   "proposals": len(proposals), "selected_box": selected_box,
                   "top_similarity": float(ranked[0][1]) if ranked else None,
                   "top_generic_label": ranked[0][0]["generic_label"] if ranked else None,
                   "evaluator_visible_pixels": gt_pixels, "evaluator_iou": overlap,
                   "outcome": outcome}
            rows.append(row)
            annotated[camera].append(annotation(image, camera, step, row, gt_box))
            print(f"{camera:5s} {step:02d}: {outcome:17s} props={len(proposals):2d} "
                  f"sim={row['top_similarity']} gt_px={gt_pixels} iou={overlap:.2f}", flush=True)
    videos = {}
    for camera, frames in annotated.items():
        path = OUT / f"recognition-{camera}.mp4"
        imageio.mimsave(path, frames, fps=4, codec="libx264", quality=7)
        videos[camera] = {"path": str(path), "sha256": sha256(path), "frames": len(frames)}
    report = {"scope": "RGB-only owner-click exemplar to generic proposal/DINOv2 transfer; no location or plan",
              "source_sha256": sha256(Path(__file__)),
              "survey_report_sha256": sha256(OUT / "report.json"),
              "owner_click": OWNER_CLICK, "taught_box": taught_box,
              "click_detail": click_detail, "similarity_threshold": SIMILARITY_THRESHOLD,
              "sampled_steps": SAMPLES, "rows": rows, "videos": videos,
              "summary": {camera: {outcome: sum(r["camera"] == camera and r["outcome"] == outcome for r in rows)
                                   for outcome in ("correct_visible", "wrong_visible", "missed_visible",
                                                   "false_find_hidden", "abstain_hidden")}
                          for camera in ("head", "chest")}}
    (OUT / "recognition-report.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report["summary"], indent=2))


if __name__ == "__main__":
    main()
