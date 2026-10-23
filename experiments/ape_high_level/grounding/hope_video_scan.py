"""Replay a real moving-camera sequence for visual re-observation.

The camera sweep is recorded, not controlled by this program. A selected
object in frame 0 stands in for a person's teaching click. Find frames use
only RGB for proposals and visual matching; HOPE poses score projected overlap
and are never used by the find algorithm. Projection does not prove visibility
through occlusion. This is tabletop, not an ape-height camera trial.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
from PIL import Image
import torch
from transformers import AutoModelForZeroShotObjectDetection, AutoProcessor

from hope_image_find import DEFAULT_DETECTOR, DETECTOR_REVISION, PROMPT, _iou
from hope_retrieval import DEFAULT_MESHES, DEFAULT_MODEL, _features, _project_box, _vertices


ROOT = Path(__file__).resolve().parents[3]
DEFAULT_VIDEO = ROOT / ".cache/ape-high-level/hope_video_scene_0000"
VIDEO_MD5 = "02bbfcdd9f610d638ec26ddd607777fb"
TEACH_NAMES = ("TomatoSauce", "Parmesan", "CreamCheese")
TABLE_Z_CM = -34.8  # Calibrated once from the sequence's tabletop depth.


def _truth_box(video_dir: Path, frame: int, name: str, mesh_dir: Path,
               vertices: dict[str, np.ndarray]) -> list[int] | None:
    data = json.loads((video_dir / f"{frame:04d}.json").read_text())
    item = next(row for row in data["objects"] if row["class"] == name)
    if name not in vertices:
        vertices[name] = _vertices(mesh_dir / f"{name}.obj")
    return _project_box(vertices[name], np.asarray(item["pose"], dtype=np.float32),
                        np.asarray(data["camera"]["intrinsics"], dtype=np.float32),
                        data["camera"]["width"], data["camera"]["height"])


def _depth_world_point(depth_mm: np.ndarray, box: list[float],
                       camera: dict) -> list[float] | None:
    """Estimate a surface point from the center patch using RGB-aligned depth.

    HOPE-Video's published extrinsic translation is in meters although object
    poses are in centimeters; its matrix maps world to camera. This follows the
    dataset correction and is an optimistic calibrated-pose input.
    """
    x0, y0, x1, y1 = box
    cx, cy = (x0 + x1) / 2, (y0 + y1) / 2
    half_w = max(2, (x1 - x0) * 0.08)
    half_h = max(2, (y1 - y0) * 0.08)
    left, right = max(0, int(cx - half_w)), min(depth_mm.shape[1], int(cx + half_w) + 1)
    top, bottom = max(0, int(cy - half_h)), min(depth_mm.shape[0], int(cy + half_h) + 1)
    values = depth_mm[top:bottom, left:right]
    values = values[(values > 70) & (values < 3000)]
    if len(values) < 3:
        return None
    depth_cm = float(np.median(values)) / 10
    k = np.asarray(camera["intrinsics"], dtype=np.float64)
    local = np.array([(cx - k[0, 2]) * depth_cm / k[0, 0],
                      (cy - k[1, 2]) * depth_cm / k[1, 1], depth_cm, 1.0])
    world_to_camera = np.asarray(camera["extrinsics"], dtype=np.float64).copy()
    world_to_camera[:3, 3] *= 100
    return (np.linalg.inv(world_to_camera) @ local)[:3].tolist()


def _plane_world_point(box: list[float], camera: dict,
                       plane_z_cm: float) -> list[float] | None:
    """Intersect the bbox bottom-center ray with a known horizontal plane."""
    u, v = (box[0] + box[2]) / 2, box[3] - 2
    k = np.asarray(camera["intrinsics"], dtype=np.float64)
    ray_camera = np.array([(u - k[0, 2]) / k[0, 0],
                           (v - k[1, 2]) / k[1, 1], 1.0])
    world_to_camera = np.asarray(camera["extrinsics"], dtype=np.float64).copy()
    world_to_camera[:3, 3] *= 100
    camera_to_world = np.linalg.inv(world_to_camera)
    ray_world = camera_to_world[:3, :3] @ ray_camera
    origin = camera_to_world[:3, 3]
    if abs(ray_world[2]) < 1e-5:
        return None
    scale = (plane_z_cm - origin[2]) / ray_world[2]
    if scale <= 0 or scale > 300:
        return None
    return (origin + scale * ray_world).tolist()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--video", type=Path, default=DEFAULT_VIDEO)
    parser.add_argument("--meshes", type=Path, default=DEFAULT_MESHES)
    parser.add_argument("--retriever", type=Path, default=DEFAULT_MODEL)
    parser.add_argument("--detector", type=Path, default=DEFAULT_DETECTOR)
    parser.add_argument("--stride", type=int, default=20)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    frames = sorted(int(p.stem.split("_")[0]) for p in args.video.glob("*_rgb.jpg"))
    frames = [frame for frame in frames if frame % args.stride == 0]
    if len(frames) < 3 or frames[0] != 0:
        raise ValueError("Need frame 0 and at least two later sampled frames")

    vertices: dict[str, np.ndarray] = {}
    teach_image = Image.open(args.video / "0000_rgb.jpg").convert("RGB")
    teach_boxes = {name: _truth_box(args.video, 0, name, args.meshes, vertices)
                   for name in TEACH_NAMES}
    if any(box is None for box in teach_boxes.values()):
        raise ValueError("Selected teach items must be projected in frame 0")
    crops = [teach_image.crop(teach_boxes[name]) for name in TEACH_NAMES]
    teach_annotation = json.loads((args.video / "0000.json").read_text())
    teach_depth = np.asarray(Image.open(args.video / "0000_depth.png"))
    teach_points = {name: _depth_world_point(teach_depth, teach_boxes[name],
                                            teach_annotation["camera"])
                    for name in TEACH_NAMES}
    teach_plane_points = {name: _plane_world_point(teach_boxes[name],
                                                 teach_annotation["camera"], TABLE_Z_CM)
                          for name in TEACH_NAMES}
    device = "mps" if torch.backends.mps.is_available() else "cpu"
    processor = AutoProcessor.from_pretrained(args.detector)
    detector = AutoModelForZeroShotObjectDetection.from_pretrained(args.detector).to(device).eval()
    proposals = []
    for frame in frames[1:]:
        image = Image.open(args.video / f"{frame:04d}_rgb.jpg").convert("RGB")
        depth = np.asarray(Image.open(args.video / f"{frame:04d}_depth.png"))
        camera = json.loads((args.video / f"{frame:04d}.json").read_text())["camera"]
        inputs = processor(images=image, text=[PROMPT], return_tensors="pt").to(device)
        with torch.no_grad():
            outputs = detector(**inputs)
        result = processor.post_process_grounded_object_detection(
            outputs, inputs.input_ids, threshold=0.25, text_threshold=0.25,
            target_sizes=[image.size[::-1]],
        )[0]
        for box, score, label in zip(result["boxes"], result["scores"], result["text_labels"]):
            xyxy = box.cpu().tolist()
            area_fraction = ((xyxy[2] - xyxy[0]) * (xyxy[3] - xyxy[1])) / (image.width * image.height)
            if not 0.0001 <= area_fraction <= 0.4:
                continue
            crop = image.crop(tuple(round(x) for x in xyxy))
            if min(crop.size) < 8:
                continue
            proposals.append({"frame": frame, "box": xyxy, "label": label,
                              "detector_score": float(score.item()),
                              "depth_world_xyz_cm": _depth_world_point(depth, xyxy, camera),
                              "known_plane_world_xyz_cm": _plane_world_point(xyxy, camera, TABLE_Z_CM)})
            crops.append(crop)
        print(f"frame {frame}: {sum(p['frame'] == frame for p in proposals)} proposals", flush=True)
    del detector
    if device == "mps":
        torch.mps.empty_cache()
    features = _features(crops, args.retriever, batch_size=16)
    rows = []
    for target_id, name in enumerate(TEACH_NAMES):
        for frame in frames[1:]:
            truth = _truth_box(args.video, frame, name, args.meshes, vertices)
            ids = [i for i, p in enumerate(proposals) if p["frame"] == frame]
            proposal_recall = max((_iou(proposals[i]["box"], truth)
                                   for i in ids), default=0.0) if truth else 0.0
            scores = features[[len(TEACH_NAMES) + i for i in ids]] @ features[target_id] if ids else []
            best_id = ids[int(np.argmax(scores))] if ids else None
            chosen_iou = _iou(proposals[best_id]["box"], truth) if truth and best_id is not None else 0.0
            teach_point = teach_points[name]
            distances = [(np.linalg.norm(np.subtract(teach_point, proposals[i]["depth_world_xyz_cm"])), i)
                         for i in ids if teach_point is not None
                         and proposals[i]["depth_world_xyz_cm"] is not None]
            depth_id = min(distances)[1] if distances else None
            depth_iou = _iou(proposals[depth_id]["box"], truth) if truth and depth_id is not None else 0.0
            plane_point = teach_plane_points[name]
            plane_distances = [(np.linalg.norm(np.subtract(plane_point, proposals[i]["known_plane_world_xyz_cm"])), i)
                               for i in ids if plane_point is not None
                               and proposals[i]["known_plane_world_xyz_cm"] is not None]
            plane_id = min(plane_distances)[1] if plane_distances else None
            plane_iou = _iou(proposals[plane_id]["box"], truth) if truth and plane_id is not None else 0.0
            rows.append({"target": name, "frame": frame, "projected_box": truth,
                         "proposal_count": len(ids), "proposal_target_iou": proposal_recall,
                         "chosen_box": proposals[best_id]["box"] if best_id is not None else None,
                         "chosen_label": proposals[best_id]["label"] if best_id is not None else None,
                         "chosen_target_iou": chosen_iou,
                         "depth_nearest_box": proposals[depth_id]["box"] if depth_id is not None else None,
                         "depth_nearest_world_distance_cm": float(min(distances)[0]) if distances else None,
                         "depth_nearest_target_iou": depth_iou,
                         "plane_nearest_box": proposals[plane_id]["box"] if plane_id is not None else None,
                         "plane_nearest_world_distance_cm": float(min(plane_distances)[0]) if plane_distances else None,
                         "plane_nearest_target_iou": plane_iou,
                         "top_similarity": float(np.max(scores)) if ids else None,
                         "margin_to_second": float(np.sort(scores)[-1] - np.sort(scores)[-2])
                         if len(ids) > 1 else None})
    in_frame = [row for row in rows if row["projected_box"] is not None]
    result = {
        "scope": ("Real robot-arm-camera motion replay; sampled chronological frames. "
                  "The agent does not choose camera motion. Ground truth is a projected "
                  "mesh box for scoring only, which may include occluded objects. RGBD "
                  "geometry uses dataset-calibrated camera poses unavailable to a walking ape. "
                  "The RGB+known-plane variant also assumes a one-time calibrated tabletop plane. "
                  "No person/hand, ape-height floor view or physical control."),
        "dataset": {"name": "NVIDIA HOPE-Video scene_0000", "archive_md5": VIDEO_MD5,
                    "sampled_frames": frames, "native_rgb_size": [640, 480]},
        "teach": {"frame": 0, "names": TEACH_NAMES, "boxes": teach_boxes,
                  "depth_world_points_cm": teach_points,
                  "known_plane_world_points_cm": teach_plane_points,
                  "selection": "evaluator annotation standing in for human click"},
        "known_table_plane_z_cm": TABLE_Z_CM,
        "detector": {"name": "IDEA-Research/grounding-dino-tiny",
                     "revision": DETECTOR_REVISION, "prompt": PROMPT,
                     "weights_sha256": hashlib.sha256((args.detector / "model.safetensors").read_bytes()).hexdigest()},
        "retriever": {"name": "facebook/dinov2-small",
                      "weights_sha256": hashlib.sha256((args.retriever / "model.safetensors").read_bytes()).hexdigest()},
        "summary": {"scored_frames": len(rows), "target_projected_in_frame": len(in_frame),
                    "target_proposed_at_iou_025": sum(r["proposal_target_iou"] >= 0.25 for r in in_frame),
                    "rgb_top_choice_correct_at_iou_025": sum(r["chosen_target_iou"] >= 0.25 for r in in_frame),
                    "rgb_known_plane_nearest_correct_at_iou_025": sum(
                        r["plane_nearest_target_iou"] >= 0.25 for r in in_frame),
                    "calibrated_rgbd_nearest_correct_at_iou_025": sum(
                        r["depth_nearest_target_iou"] >= 0.25 for r in in_frame)},
        "rows": rows,
        "proposals": proposals,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result["summary"], indent=2))


if __name__ == "__main__":
    main()
