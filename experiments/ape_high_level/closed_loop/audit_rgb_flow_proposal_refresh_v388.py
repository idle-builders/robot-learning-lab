"""V388: RGB flow prediction plus fresh appearance-box re-anchoring.

V383's actor-only 3D moving-track box bootstraps identity once. Thereafter
KLT predicts its next image position and frozen DINO+hue proposals refresh
the box/features. No depth or head angle is used after bootstrap.
"""

from __future__ import annotations

import json
from pathlib import Path

import cv2
import imageio.v2 as imageio
import numpy as np
from PIL import Image, ImageDraw

from audit_direct_structural_entry_v271 import V149, V150
from audit_entry_color_v150 import descriptor
from audit_rgb_flow_after_bootstrap_v387 import frame
from run import ROOT, Vision, iou, sha
from run_color_grounded_entry_v151 import DINO_MIN, HUE_MIN


SOURCE = Path(__file__)
TRACK_INPUT = ROOT / "artifacts/ape-closed-loop-cove50-v383/multi-candidate-motion-audit.json"
VIDEO_INPUT = ROOT / "artifacts/ape-closed-loop-cove50-v381/independent-hand-show.json"
OUT = ROOT / "artifacts/ape-closed-loop-cove50-v388"
MAX_BOX_CENTER_GAP_PX = 60.
BACKTRACK_ERROR_PX = 20.
MIN_FEATURES = 8


def appearance_boxes(vision: Vision, exemplar: np.ndarray,
                     owner_hist: np.ndarray, rgb: np.ndarray) -> list[list]:
    image = Image.fromarray(rgb)
    proposals = vision.detect(image)
    scores = (vision.feature([image.crop(row["box"])
                              for row in proposals]) @ exemplar
              if proposals else [])
    result = []
    for proposal, score in zip(proposals, scores):
        box = proposal["box"]
        hue = np.asarray(descriptor(image, box)["hue_histogram"])
        overlap = float(np.minimum(owner_hist, hue).sum())
        if score >= DINO_MIN and overlap >= HUE_MIN:
            result.append(box)
    return result


def features(gray: np.ndarray, box: np.ndarray):
    mask = np.zeros(gray.shape, dtype=np.uint8)
    x0, y0, x1, y1 = np.rint(box).astype(int)
    mask[max(0, y0):min(mask.shape[0], y1),
         max(0, x0):min(mask.shape[1], x1)] = 255
    return cv2.goodFeaturesToTrack(gray, mask=mask, maxCorners=40,
                                   qualityLevel=.005, minDistance=2)


def center(box: np.ndarray) -> np.ndarray:
    return np.asarray(((box[0] + box[2]) / 2,
                       (box[1] + box[3]) / 2))


def run_case(source: dict, bootstrap: dict, vision: Vision,
             exemplar: np.ndarray, owner_hist: np.ndarray) -> dict:
    first = next(row for row in bootstrap["frames"]
                 if row["actor_selected_box"] is not None)
    first_index = first["frame"]
    box = np.asarray(first["actor_selected_box"], dtype=float)
    branch = OUT / source["case"]
    branch.mkdir(parents=True, exist_ok=True)
    rgb = frame(source["samples"][first_index])
    gray = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)
    points = features(gray, box)
    if points is None or len(points) < MIN_FEATURES:
        raise RuntimeError("Actor bootstrap lacks RGB features")
    rows = []
    video = branch / "rgb-flow-proposal-track.mp4"
    with imageio.get_writer(video, fps=10, codec="libx264",
                            quality=7) as writer:
        for index in range(first_index, len(source["samples"])):
            candidate_count = 0
            flow_features = len(points)
            if index > first_index:
                next_rgb = frame(source["samples"][index])
                next_gray = cv2.cvtColor(next_rgb, cv2.COLOR_RGB2GRAY)
                forward, status, _ = cv2.calcOpticalFlowPyrLK(
                    gray, next_gray, points, None, winSize=(31, 31),
                    maxLevel=4)
                backward, reverse_status, _ = cv2.calcOpticalFlowPyrLK(
                    next_gray, gray, forward, None, winSize=(31, 31),
                    maxLevel=4)
                old = points.reshape(-1, 2)
                new = forward.reshape(-1, 2)
                error = np.linalg.norm(backward.reshape(-1, 2) - old,
                                       axis=1)
                good = (status.reshape(-1).astype(bool) &
                        reverse_status.reshape(-1).astype(bool) &
                        (error <= BACKTRACK_ERROR_PX))
                predicted = box.copy()
                flow_features = int(good.sum())
                if flow_features >= MIN_FEATURES:
                    delta = np.median(new[good] - old[good], axis=0)
                    predicted += np.asarray((delta[0], delta[1],
                                             delta[0], delta[1]))
                candidates = appearance_boxes(vision, exemplar, owner_hist,
                                              next_rgb)
                candidate_count = len(candidates)
                ranked = sorted((float(np.linalg.norm(center(np.asarray(c)) -
                                                       center(predicted))), c)
                                for c in candidates)
                if (not ranked or ranked[0][0] > MAX_BOX_CENTER_GAP_PX or
                        (len(ranked) > 1 and
                         ranked[1][0] <= MAX_BOX_CENTER_GAP_PX and
                         ranked[1][0] - ranked[0][0] < 10.)):
                    rows.append({"frame": index, "actor_status":
                                 "no_unique_nearby_appearance_box",
                                 "actor_flow_features": flow_features,
                                 "actor_appearance_candidates":
                                     candidate_count,
                                 "actor_box": None,
                                 "evaluator_iou": 0.})
                    break
                box = np.asarray(ranked[0][1], dtype=float)
                points = features(next_gray, box)
                if points is None or len(points) < MIN_FEATURES:
                    rows.append({"frame": index, "actor_status":
                                 "insufficient_refresh_features",
                                 "actor_flow_features": flow_features,
                                 "actor_appearance_candidates":
                                     candidate_count,
                                 "actor_box": None,
                                 "evaluator_iou": 0.})
                    break
                gray, rgb = next_gray, next_rgb
            truth = source["samples"][index]["evaluator_truth_box"]
            overlap = iou(box.tolist(), truth)
            rows.append({
                "frame": index,
                "actor_status": "tracked",
                "actor_flow_features": flow_features,
                "actor_appearance_candidates": candidate_count,
                "actor_box": np.round(box, 3).tolist(),
                "evaluator_iou": round(float(overlap), 5),
                "evaluator_held_can_match": overlap >= .25,
            })
            view = Image.fromarray(rgb)
            draw = ImageDraw.Draw(view)
            draw.rectangle(tuple(box), outline=(31, 230, 79), width=3)
            if truth is not None:
                draw.rectangle(tuple(truth), outline=(44, 145, 255), width=2)
            draw.rectangle((0, 0, 450, 26), fill=(0, 0, 0))
            draw.text((5, 6), f"RGB flow+proposal frame{index} "
                      f"green=actor blue=score IoU{overlap:.2f}",
                      fill=(255, 255, 255))
            writer.append_data(np.asarray(view))
    return {
        "case": source["case"], "actor_bootstrap_frame": first_index,
        "actor_tracked_frames": sum(row["actor_status"] == "tracked"
                                    for row in rows),
        "evaluator_held_matches": sum(row.get("evaluator_held_can_match",
                                               False) for row in rows),
        "video_path": str(video.relative_to(ROOT)),
        "video_sha256": sha(video), "frames": rows,
    }


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    vision = Vision("generic")
    exemplar = np.load(V149)
    owner_hist = np.asarray(json.loads(Path(V150).read_text())[
        "owner_color"]["hue_histogram"])
    sources = json.loads(VIDEO_INPUT.read_text())["rows"]
    bootstrap = [row for row in json.loads(TRACK_INPUT.read_text())["rows"]
                 if "v382" in row["source_candidate_report"]]
    rows = [run_case(source, next(row for row in bootstrap
                                  if row["case"] == source["case"]),
                     vision, exemplar, owner_hist)
            for source in sources]
    for row in rows:
        print(row["case"], "tracked", row["actor_tracked_frames"],
              "held", row["evaluator_held_matches"], flush=True)
    report = {
        "scope": "V383 actor 3D moving-track box bootstraps identity at frame4 of each V381 raised-replica video. Later frames use only exposed RGB: pyramidal KLT median shift (revealed <=20px forward/backward limit) predicts a box, frozen DINO+hue appearance proposals re-anchor to its nearest center within60px, and feature points refresh; ambiguous nearby matches or too few points abstain. Evaluator truth scores and overlays only. Both thresholds were selected on revealed same-geometry frames; this is not held-out, latency-qualified or tested through long occlusion and cannot identify the right can without the initial metric/owner cue.",
        "source_sha256": sha(SOURCE),
        "track_input_sha256": sha(TRACK_INPUT),
        "video_input_sha256": sha(VIDEO_INPUT),
        "rows": rows,
    }
    (OUT / "rgb-flow-proposal-audit.json").write_text(
        json.dumps(report, indent=2) + "\n")
    (OUT / SOURCE.name).write_bytes(SOURCE.read_bytes())


if __name__ == "__main__":
    main()
