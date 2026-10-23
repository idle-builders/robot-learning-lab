"""Probe real-photo face and hand visibility for the ape's teaching loop.

Uses MediaPipe's released Face Detector and Hand Landmarker with upstream
vision test images from https://storage.googleapis.com/mediapipe-assets/ .
These are model-validation photos, not frames from the 50 cm ape or evidence
that a detected finger identifies a particular object. Resizing probes pixel
support only; it does not reproduce the ape's optics, viewpoint or motion blur.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import mediapipe as mp
import numpy as np
from PIL import Image


ROOT = Path(__file__).resolve().parents[3]
DEFAULT_DATA = ROOT / ".cache/ape-high-level/social_test"
IMAGES = ("portrait.jpg", "portrait_small.jpg", "male_full_height_hands.jpg",
          "pointing_up.jpg", "left_hands.jpg", "right_hands.jpg",
          "fist.jpg", "thumb_up.jpg", "cat.jpg", "burger.jpg")
SIZES = ((None, None), (640, 480), (320, 240))


def _scaled(image: Image.Image, bound: tuple[int | None, int | None]) -> Image.Image:
    if bound[0] is None:
        return image
    ratio = min(1.0, bound[0] / image.width, bound[1] / image.height)
    return image.resize((round(image.width * ratio), round(image.height * ratio)),
                        Image.Resampling.LANCZOS)


def _box(points: list, width: int, height: int) -> list[int]:
    x = [point.x * width for point in points]
    y = [point.y * height for point in points]
    return [round(min(x)), round(min(y)), round(max(x)), round(max(y))]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, default=DEFAULT_DATA)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    hand_file = args.data / "hand_landmarker.task"
    face_file = args.data / "blaze_face_short_range.tflite"
    pose_file = args.data / "pose_landmarker_lite.task"
    cpu = mp.tasks.BaseOptions.Delegate.CPU
    hand_options = mp.tasks.vision.HandLandmarkerOptions(
        base_options=mp.tasks.BaseOptions(model_asset_path=str(hand_file), delegate=cpu),
        num_hands=2,
    )
    face_options = mp.tasks.vision.FaceDetectorOptions(
        base_options=mp.tasks.BaseOptions(model_asset_path=str(face_file), delegate=cpu),
    )
    pose_options = mp.tasks.vision.PoseLandmarkerOptions(
        base_options=mp.tasks.BaseOptions(model_asset_path=str(pose_file), delegate=cpu),
        num_poses=2,
    )
    rows = []
    with (mp.tasks.vision.HandLandmarker.create_from_options(hand_options) as hands,
          mp.tasks.vision.FaceDetector.create_from_options(face_options) as faces,
          mp.tasks.vision.PoseLandmarker.create_from_options(pose_options) as poses):
        for name in IMAGES:
            original = Image.open(args.data / name).convert("RGB")
            for bound in SIZES:
                image = _scaled(original, bound)
                data = np.asarray(image)
                frame = mp.Image(image_format=mp.ImageFormat.SRGB, data=data)
                hand_result = hands.detect(frame)
                face_result = faces.detect(frame)
                pose_result = poses.detect(frame)
                hand_rows = []
                for landmarks, handedness in zip(hand_result.hand_landmarks,
                                                 hand_result.handedness):
                    hand_rows.append({
                        "bbox": _box(landmarks, image.width, image.height),
                        "index_tip_px": [round(landmarks[8].x * image.width),
                                         round(landmarks[8].y * image.height)],
                        "handedness": handedness[0].category_name,
                        "handedness_score": handedness[0].score,
                    })
                face_rows = []
                for detection in face_result.detections:
                    box = detection.bounding_box
                    face_rows.append({
                        "bbox": [box.origin_x, box.origin_y,
                                 box.origin_x + box.width, box.origin_y + box.height],
                        "score": detection.categories[0].score,
                    })
                pose_rows = []
                for landmarks in pose_result.pose_landmarks:
                    pose_rows.append({
                        "nose_px": [round(landmarks[0].x * image.width),
                                    round(landmarks[0].y * image.height)],
                        "shoulders_px": [[round(landmarks[i].x * image.width),
                                          round(landmarks[i].y * image.height)]
                                         for i in (11, 12)],
                        "nose_visibility": landmarks[0].visibility,
                    })
                rows.append({"image": name,
                             "variant": "native" if bound[0] is None else f"fit_{bound[0]}x{bound[1]}",
                             "size": [image.width, image.height],
                             "hands": hand_rows, "faces": face_rows,
                             "poses": pose_rows})
    result = {
        "scope": ("Released face/hand detector on a small set of official real-photo test images "
                  "and resized versions; no ape camera, no gesture-to-object association, "
                  "no multi-person or temporal evaluation. Reported model scores are "
                  "not calibrated probabilities of a person offering an object."),
        "mediapipe_version": mp.__version__,
        "models": {"hand_landmarker_sha256": hashlib.sha256(hand_file.read_bytes()).hexdigest(),
                   "face_detector_sha256": hashlib.sha256(face_file.read_bytes()).hexdigest(),
                   "pose_landmarker_sha256": hashlib.sha256(pose_file.read_bytes()).hexdigest()},
        "images_sha256": {name: hashlib.sha256((args.data / name).read_bytes()).hexdigest()
                          for name in IMAGES},
        "rows": rows,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n")
    for row in rows:
        print(row["image"], row["variant"], row["size"],
              "hands", len(row["hands"]), "faces", len(row["faces"]),
              "poses", len(row["poses"]))


if __name__ == "__main__":
    main()
