"""Probe a released hand tracker on real robot-facing pointing photographs.

Innsbruck Pointing at Objects: https://iis.uibk.ac.at/datasets/ipo
The dataset supplies ten *ground-truth* object centers per scene. Using them
isolates pointing from object discovery; it is not an end-to-end teach loop.
The fixed, elevated Kinect viewpoint is also not the 50 cm ape's viewpoint.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import mediapipe as mp
import numpy as np
from PIL import Image
from scipy.io import loadmat


ROOT = Path(__file__).resolve().parents[3]
DEFAULT_DATA = ROOT / ".cache/ape-high-level/ibkpointingdataset/Innsbruck_Pointing_Dataset"
DEFAULT_MODEL = ROOT / ".cache/ape-high-level/social_test/hand_landmarker.task"
DEFAULT_GESTURE_MODEL = ROOT / ".cache/ape-high-level/social_test/gesture_recognizer.task"


def _xy(landmark, width: int, height: int) -> np.ndarray:
    return np.array([landmark.x * width, landmark.y * height], dtype=float)


def _rank_objects(landmarks, width: int, height: int, objects: np.ndarray):
    tip = _xy(landmarks[8], width, height)
    mcp = _xy(landmarks[5], width, height)
    axis = tip - mcp
    axis_norm = float(np.linalg.norm(axis))
    if axis_norm < 1:
        return None
    direction = axis / axis_norm
    delta = objects - tip
    forward = delta @ direction
    distance = np.linalg.norm(delta, axis=1)
    # 2-D projected finger direction is a weak cue: choose the object with
    # smallest angular deviation in front of the finger. Never treat this
    # dataset-annotation-only ranking as a deployed object detector.
    angles = np.arccos(np.clip(forward / np.maximum(distance, 1), -1, 1))
    angles[forward <= 0] = np.inf
    order = [int(i) for i in np.argsort(angles) if np.isfinite(angles[i])]
    return {"tip_px": tip.round(1).tolist(), "mcp_px": mcp.round(1).tolist(),
            "angle_degrees": [round(float(a * 180 / np.pi), 2) if np.isfinite(a)
                              else None for a in angles],
            "ray_order": order,
            "nearest_tip": int(np.argmin(distance))}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, default=DEFAULT_DATA)
    parser.add_argument("--model", type=Path, default=DEFAULT_MODEL)
    parser.add_argument("--gesture-model", type=Path, default=DEFAULT_GESTURE_MODEL)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--max-width", type=int, default=640,
                        help="Resize while preserving aspect ratio; dataset RGB is 640x480")
    args = parser.parse_args()
    options = mp.tasks.vision.HandLandmarkerOptions(
        base_options=mp.tasks.BaseOptions(
            model_asset_path=str(args.model), delegate=mp.tasks.BaseOptions.Delegate.CPU),
        num_hands=2,
    )
    gesture_options = mp.tasks.vision.GestureRecognizerOptions(
        base_options=mp.tasks.BaseOptions(
            model_asset_path=str(args.gesture_model),
            delegate=mp.tasks.BaseOptions.Delegate.CPU),
        num_hands=2,
    )
    rows = []
    with (mp.tasks.vision.HandLandmarker.create_from_options(options) as tracker,
          mp.tasks.vision.GestureRecognizer.create_from_options(gesture_options) as gestures):
        for scene in sorted(path for path in args.data.iterdir() if path.is_dir()):
            truth = loadmat(scene / "objLocGroundTruth.mat", squeeze_me=True)
            objects = np.asarray(truth["obj2DLoc"], dtype=float)
            object_names = [str(name).strip() for name in truth["objList"]]
            for index in range(1, 12):
                image_path = scene / f"{index:05}.png"
                image = Image.open(image_path).convert("RGB")
                scale = min(1.0, args.max_width / image.width)
                if scale < 1:
                    image = image.resize((round(image.width * scale),
                                          round(image.height * scale)),
                                         Image.Resampling.LANCZOS)
                scaled_objects = objects * scale
                frame = mp.Image(image_format=mp.ImageFormat.SRGB,
                                 data=np.asarray(image))
                result = tracker.detect(frame)
                gesture_result = gestures.recognize(frame)
                detected = []
                for landmarks in result.hand_landmarks:
                    ranked = _rank_objects(landmarks, image.width, image.height, scaled_objects)
                    if ranked is not None:
                        detected.append(ranked)
                row = {"scene": scene.name, "frame": image_path.name,
                       "kind": "natural" if int(scene.name) % 2 else "tool",
                       "has_pointing": index > 1,
                       "target_index": index - 2 if index > 1 else None,
                       "target_name": object_names[index - 2] if index > 1 else None,
                       "canned_gestures": [
                           {"label": categories[0].category_name,
                            "score": categories[0].score} for categories in gesture_result.gestures
                           if categories],
                       "hand_candidates": detected}
                hand_truth_path = scene / f"{index:05}handGt.mat"
                if index > 1 and hand_truth_path.exists():
                    hand_truth = loadmat(hand_truth_path, squeeze_me=True)
                    hand_center = np.asarray(hand_truth["hand2DLoc"], dtype=float) * scale
                    row["hand_center_gt_px"] = hand_center.tolist()
                    for hand in detected:
                        hand["tip_to_gt_hand_center_px"] = round(float(
                            np.linalg.norm(np.asarray(hand["tip_px"]) - hand_center)), 1)
                rows.append(row)
    result = {"scope": ("Released MediaPipe hand model on all 180 real pointing photos "
                        "and 18 no-pointing frames from IPO. Object centers are dataset "
                        "ground truth, so pointing ranking is an oracle-object upper bound. "
                        "Fixed elevated Kinect viewpoint; no 50 cm ape viewpoint, "
                        "no human-object association in a live scene."),
              "dataset": "https://iis.uibk.ac.at/datasets/ipo",
              "max_width": args.max_width,
              "dataset_archive_sha256": hashlib.sha256((ROOT / ".cache/ape-high-level/ibkpointingdataset.zip").read_bytes()).hexdigest(),
              "hand_model_sha256": hashlib.sha256(args.model.read_bytes()).hexdigest(),
              "gesture_model_sha256": hashlib.sha256(args.gesture_model.read_bytes()).hexdigest(),
              "rows": rows}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n")
    for kind in ("natural", "tool"):
        pointed = [row for row in rows if row["kind"] == kind and row["has_pointing"]]
        detected = [row for row in pointed if row["hand_candidates"]]
        ray = [row for row in detected if row["hand_candidates"][0]["ray_order"]
               and row["hand_candidates"][0]["ray_order"][0] == row["target_index"]]
        tip = [row for row in detected if row["hand_candidates"][0]["nearest_tip"] == row["target_index"]]
        no_point = [row for row in rows if row["kind"] == kind and not row["has_pointing"]]
        print(kind, "pointing frames", len(pointed), "hand found", len(detected),
              "first-hand 2d-ray correct", len(ray), "nearest-tip correct", len(tip),
              "hands on no-pointing frames", sum(bool(row["hand_candidates"]) for row in no_point),
              "canned Pointing_Up", sum(any(g["label"] == "Pointing_Up"
                                           for g in row["canned_gestures"]) for row in pointed))


if __name__ == "__main__":
    main()
