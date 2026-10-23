"""V21: real RGB person proposals on the articulated kitchen head views.

SSDLite sees RGB only. The depth crop is actor-available if it proposes a box;
MuJoCo segmentation and scripted person XY are evaluator-only scoring data.
This probes whether the stylized capsule can support semantic camera fusion.
"""

from __future__ import annotations

import json
import math
from pathlib import Path
import time

import cv2
import mujoco
import numpy as np
from PIL import Image
import torch
import torchvision
from torchvision.models.detection import (
    SSDLite320_MobileNet_V3_Large_Weights,
    ssdlite320_mobilenet_v3_large,
)

import changed_speed_trial as held
import kitchen_native as kitchen
import kitchen_trial as trial


OUT = kitchen.OUT / "v21-camera-person-probe"
TIMES_S = (.16, .22, .30, .50, .90)
CASES = ("fast_clear_held", "slow_cross_held", "empty_room")
CAMERAS = ("ape_upper_rgb", "ape_lower_rgb")
IMAGE_WH = (320, 240)
PERSON_SCORE_GATE = .35  # Same SSDLite gate as the project's online Isaac actor.


def evaluator_bbox(segmented: np.ndarray, person_id: int) -> tuple[list[int] | None, int]:
    ys, xs = np.where(segmented[:, :, 0] == person_id)
    if len(xs) == 0:
        return None, 0
    return [int(xs.min()), int(ys.min()), int(xs.max()+1), int(ys.max()+1)], len(xs)


def box_iou(a: list[float], b: list[float]) -> float:
    overlap = max(0., min(a[2], b[2])-max(a[0], b[0])) * max(
        0., min(a[3], b[3])-max(a[1], b[1]))
    area_a = max(0., a[2]-a[0])*max(0., a[3]-a[1])
    area_b = max(0., b[2]-b[0])*max(0., b[3]-b[1])
    return overlap/(area_a+area_b-overlap) if area_a+area_b > overlap else 0.


def local_depth(depth: np.ndarray, box: list[float]) -> tuple[float, float | None]:
    x1, y1, x2, y2 = box
    sx1, sx2 = x1+.35*(x2-x1), x1+.65*(x2-x1)
    sy1, sy2 = y1+.25*(y2-y1), y1+.55*(y2-y1)
    patch = depth[max(0, int(sy1)):min(depth.shape[0], math.ceil(sy2)),
                  max(0, int(sx1)):min(depth.shape[1], math.ceil(sx2))]
    valid = np.isfinite(patch) & (patch > 0) & (patch < 10.)
    return (float(valid.mean()) if patch.size else 0.,
            float(np.median(patch[valid])) if valid.any() else None)


def overlay(rgb: np.ndarray, proposals: list[dict], truth: list[int] | None) -> np.ndarray:
    canvas = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
    if truth is not None:
        cv2.rectangle(canvas, tuple(truth[:2]), tuple(truth[2:]), (30, 210, 80), 2)
    for p in proposals:
        if p["score"] < PERSON_SCORE_GATE:
            continue
        box = [int(round(z)) for z in p["box_xyxy"]]
        cv2.rectangle(canvas, tuple(box[:2]), tuple(box[2:]), (30, 85, 235), 2)
        cv2.putText(canvas, f"person {p['score']:.2f}", (box[0]+3, max(14,box[1]+15)),
                    cv2.FONT_HERSHEY_SIMPLEX, .45, (30, 85, 235), 1, cv2.LINE_AA)
    return canvas


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    weights = SSDLite320_MobileNet_V3_Large_Weights.COCO_V1
    model_detector = ssdlite320_mobilenet_v3_large(weights=weights).eval()
    torch.set_num_threads(4)
    model = mujoco.MjModel.from_xml_path(str(kitchen.OUT / "scene.xml"))
    data = mujoco.MjData(model)
    renderer = mujoco.Renderer(model, height=IMAGE_WH[1], width=IMAGE_WH[0])
    person_id = model.geom("person_capsule").id
    rows = []
    images = []
    with np.load(kitchen.TRACE, allow_pickle=False) as saved:
        for world in range(4):
            for case in CASES:
                for t in TIMES_S:
                    model.body_pos[model.body("approaching_person").id, :2] = (
                        (10., -1.) if case == "empty_room" else held.hazard_xy(case, t))
                    model.body_pos[model.body("low_pet").id, :2] = (10., -1.)
                    pose = kitchen.put_native_pose(model, data, saved, world, t, 3.)
                    for camera in CAMERAS:
                        renderer.enable_segmentation_rendering()
                        renderer.update_scene(data, camera=camera)
                        segmented = renderer.render().copy()
                        renderer.disable_segmentation_rendering()
                        renderer.enable_depth_rendering()
                        renderer.update_scene(data, camera=camera)
                        depth = renderer.render().copy()
                        renderer.disable_depth_rendering()
                        renderer.update_scene(data, camera=camera)
                        rgb = renderer.render().copy()
                        started = time.perf_counter()
                        tensor = weights.transforms()(Image.fromarray(rgb))
                        with torch.no_grad():
                            detected = model_detector([tensor])[0]
                        inference_ms = 1000*(time.perf_counter()-started)
                        truth, pixels = evaluator_bbox(segmented, person_id)
                        proposals = []
                        for score, label, box in zip(detected["scores"],
                                                     detected["labels"],
                                                     detected["boxes"]):
                            if int(label) != 1 or float(score) < .05:
                                continue
                            bounds = [round(float(x), 3) for x in box]
                            fraction, median = local_depth(depth, bounds)
                            proposals.append({"score": round(float(score), 5),
                                              "box_xyxy": bounds,
                                              "local_depth_valid_fraction": round(fraction, 5),
                                              "local_depth_median_m": (
                                                  round(median, 5) if median is not None else None),
                                              "evaluator_iou": (round(box_iou(bounds, truth), 5)
                                                                if truth else None)})
                        accepted = [p for p in proposals if p["score"] >= PERSON_SCORE_GATE]
                        matched = [p for p in accepted if truth and p["evaluator_iou"] >= .25]
                        row = {"world": world, "case": case, "time_s": t,
                               "camera": camera, "root_xy_m": pose[:2].round(5).tolist(),
                               "inference_ms": round(inference_ms, 2),
                               "actor_proposals": proposals,
                               "accepted_person_count": len(accepted),
                               "evaluator_visible_pixels": pixels,
                               "evaluator_box_xyxy": truth,
                               "evaluator_cropped": bool(truth and
                                   (truth[0] == 0 or truth[1] == 0 or
                                    truth[2] == IMAGE_WH[0] or truth[3] == IMAGE_WH[1])),
                               "evaluator_matched_detection": bool(matched)}
                        if world == 1 and t in (.22, .90):
                            filename = f"{case}-{camera}-t{int(t*100):03d}.png"
                            path = OUT / "frames" / filename
                            path.parent.mkdir(parents=True, exist_ok=True)
                            cv2.imwrite(str(path), overlay(rgb, proposals, truth))
                            row["overlay"] = {"path": str(path), "sha256": trial.sha(path)}
                            images.append(row["overlay"])
                        rows.append(row)
            print("world", world, "views", len(rows), flush=True)
    renderer.close()
    checkpoint = (Path(torch.hub.get_dir()) / "checkpoints" /
                  "ssdlite320_mobilenet_v3_large_coco-a79551df.pth")
    summary = {}
    for camera in CAMERAS:
        scoped = [r for r in rows if r["camera"] == camera]
        positive = [r for r in scoped if r["case"] != "empty_room" and
                    r["evaluator_visible_pixels"] >= 500]
        negative = [r for r in scoped if r["case"] == "empty_room"]
        summary[camera] = {"visible_positive_views": len(positive),
                           "matched_visible_views": sum(r["evaluator_matched_detection"]
                                                        for r in positive),
                           "accepted_empty_views": sum(r["accepted_person_count"] > 0
                                                       for r in negative),
                           "empty_views": len(negative),
                           "median_inference_ms": round(float(np.median(
                               [r["inference_ms"] for r in scoped])), 2)}
    report = {"schema": "robot-learning-lab.ape-robocasa-head-person-detector-probe.v1",
              "task": "T-9v3p3n", "body_status": "historical_rejected_direct_frame",
              "purpose": "Gate actor-side RGB/depth fusion on a real person detector. SSDLite proposals from RGB and box-local rendered depth are actor data; segmentation box/pixels and scripted case are evaluator data. Cropped capsule and empty-room controls included; no detector tuning or guard fusion.",
              "detector": {"name": "torchvision SSDLite320 MobileNetV3 COCO_V1",
                           "gate": PERSON_SCORE_GATE,
                           "torch_version": torch.__version__,
                           "torchvision_version": torchvision.__version__,
                           "checkpoint_sha256": trial.sha(checkpoint)},
              "image_size_px": list(IMAGE_WH), "cases": list(CASES),
              "times_s": list(TIMES_S), "cameras": list(CAMERAS),
              "source_sha256": trial.sha(Path(__file__)),
              "scene_sha256": trial.sha(kitchen.OUT / "scene.xml"),
              "native_trace_sha256": trial.sha(kitchen.TRACE),
              "body_sha256": trial.sha(kitchen.BODY),
              "v11_case_source_sha256": trial.sha(Path(held.__file__)),
              "summary": summary, "frames": images, "rows": rows}
    (OUT / "report.json").write_text(json.dumps(report, indent=2)+"\n")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
