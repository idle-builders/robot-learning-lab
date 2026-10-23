"""Measure a fast RGB reacquisition frontend across two RoboCasa styles.

The exemplar is the owner-click / Grounding-DINO crop from the earlier style-1
survey. The tracker never sees evaluator segmentation. The fixed 0.60 match
threshold was chosen on style 1 before rendering style 2. This tests whether
a cheap current-frame search can complement slow semantic inference; it is
not a general object recognizer or an owner/locomotion episode.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
import statistics
import time

import cv2
import imageio.v2 as imageio
import mujoco
import numpy as np
from PIL import Image, ImageDraw

from probe_robocasa_eye_height import WIDTH, HEIGHT, capture, scene


ROOT = Path(__file__).resolve().parents[3]
BASE = ROOT / "artifacts/ape-robocasa-kitchen-pilot"
OUT = ROOT / "artifacts/ape-robocasa-fast-rgb-v402"
SAMPLES = (*range(0, 48, 4), 35, 47)
THRESHOLD = .60
SCALES = (.8, .9, 1.0, 1.1, 1.2)


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def iou(a: list[int], b: list[int] | None) -> float:
    if b is None:
        return 0.0
    x0, y0, x1, y1 = max(a[0], b[0]), max(a[1], b[1]), min(a[2], b[2]), min(a[3], b[3])
    overlap = max(0, x1 - x0) * max(0, y1 - y0)
    union = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - overlap
    return overlap / union if union else 0.0


def match(rgb: np.ndarray, template: np.ndarray) -> dict:
    start = time.perf_counter()
    image = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
    matches = []
    for scale in SCALES:
        scaled = cv2.resize(template, None, fx=scale, fy=scale,
                            interpolation=cv2.INTER_AREA)
        response = cv2.matchTemplate(image, scaled, cv2.TM_CCOEFF_NORMED)
        _, score, _, xy = cv2.minMaxLoc(response)
        matches.append((score, [xy[0], xy[1], xy[0] + scaled.shape[1],
                                xy[1] + scaled.shape[0]], scale))
    score, box, scale = max(matches, key=lambda row: row[0])
    return {"score": float(score), "box": box, "scale": scale,
            "latency_wall_s": time.perf_counter() - start}


def annotate(rgb: np.ndarray, result: dict, gt_box: list[int] | None,
             outcome: str, label: str) -> np.ndarray:
    canvas = Image.new("RGB", (WIDTH, HEIGHT + 48), "#10151a")
    canvas.paste(Image.fromarray(rgb), (0, 48))
    draw = ImageDraw.Draw(canvas)
    draw.text((10, 8), label, fill="white")
    draw.text((10, 27), f"{outcome}  score {result['score']:.3f}  {result['latency_wall_s'] * 1000:.1f} ms",
              fill="#e6ecee")
    if gt_box:
        draw.rectangle((gt_box[0], gt_box[1] + 48, gt_box[2], gt_box[3] + 48),
                       outline="#3cb7ef", width=2)
    if result["score"] >= THRESHOLD:
        box = result["box"]
        draw.rectangle((box[0], box[1] + 48, box[2], box[3] + 48),
                       outline="#44e683" if outcome == "correct_visible" else "#fc6042", width=3)
    return np.asarray(canvas)


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    template_path = BASE / "taught-rgb-crop.png"
    template = cv2.imread(str(template_path), cv2.IMREAD_COLOR)
    if template is None:
        raise FileNotFoundError(template_path)
    base_report = json.loads((BASE / "report.json").read_text())
    rows = []
    video_frames: dict[tuple[int, str], list[np.ndarray]] = {}

    # The held-out style receives no new exemplar or fitted threshold.
    model = scene(style_id=2)
    data = mujoco.MjData(model)
    renderer = mujoco.Renderer(model, height=HEIGHT, width=WIDTH)
    can_ids = {name: model.geom(name + "_geom").id for name in ("shown_can", "counter_can")}
    cameras = {name: model.camera(name).id for name in ("head", "chest")}
    for style in (1, 2):
        for camera in ("head", "chest"):
            frames = []
            for step in SAMPLES:
                if style == 1:
                    path = BASE / f"step{step:02d}-{camera}.png"
                    rgb = np.asarray(Image.open(path).convert("RGB"))
                    gt = base_report["samples"][step]
                    visible = gt["visible_pixels"][camera]["shown_can"]
                    box = gt["evaluator_boxes"][camera]["shown_can"]
                    image_sha = sha256(path)
                else:
                    model.cam_pos[cameras[camera], 0] = 1.05 + 3.30 * step / 47
                    mujoco.mj_forward(model, data)
                    rgb, _, pixels, boxes = capture(renderer, data, camera, can_ids)
                    visible = pixels["shown_can"]
                    box = boxes["shown_can"]
                    image_sha = hashlib.sha256(rgb.tobytes()).hexdigest()
                found = match(rgb, template)
                overlap = iou(found["box"], box)
                if visible >= 100:
                    outcome = ("correct_visible" if found["score"] >= THRESHOLD and overlap >= .3
                               else "wrong_visible" if found["score"] >= THRESHOLD
                               else "missed_visible")
                else:
                    outcome = "false_find_hidden" if found["score"] >= THRESHOLD else "abstain_hidden"
                row = {"style": style, "camera": camera, "step": step,
                       "image_sha256": image_sha, "visible_pixels": visible,
                       "evaluator_box": box, "iou": overlap, "outcome": outcome,
                       **found}
                rows.append(row)
                frames.append(annotate(rgb, found, box, outcome,
                                       f"style {style}, {camera}, aisle view {step:02d}"))
            video_frames[(style, camera)] = frames
    renderer.close()
    videos = []
    for (style, camera), frames in video_frames.items():
        path = OUT / f"style{style}-{camera}-template.mp4"
        imageio.mimsave(path, frames, fps=4, codec="libx264", quality=7)
        videos.append({"style": style, "camera": camera, "path": str(path),
                       "sha256": sha256(path), "frames": len(frames)})
    summary = {}
    for style in (1, 2):
        for camera in ("head", "chest"):
            selected = [r for r in rows if r["style"] == style and r["camera"] == camera]
            summary[f"style{style}-{camera}"] = {
                outcome: sum(r["outcome"] == outcome for r in selected)
                for outcome in ("correct_visible", "wrong_visible", "missed_visible",
                                "false_find_hidden", "abstain_hidden")}
            summary[f"style{style}-{camera}"]["median_latency_ms"] = round(
                1000 * statistics.median(r["latency_wall_s"] for r in selected), 2)
    report = {"scope": "RGB template speed and scene-style transfer; no action or detector retraining",
              "source_sha256": sha256(Path(__file__)),
              "template_sha256": sha256(template_path), "threshold": THRESHOLD,
              "scales": SCALES, "summary": summary, "rows": rows, "videos": videos}
    (OUT / "report.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
