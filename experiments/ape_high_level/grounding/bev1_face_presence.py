"""Check face presence on the same BEV1 frames as bev1_person_presence.py.

Uses MediaPipe's released short-range Face Detector. Binary human face labels
cannot verify that a detection is on the right person or locate a speaker.
"""

import argparse
import hashlib
import json
from pathlib import Path

import mediapipe as mp
import numpy as np
from PIL import Image


ROOT = Path(__file__).resolve().parents[3]
DEFAULT_MODEL = ROOT / ".cache/ape-high-level/social_test/blaze_face_short_range.tflite"


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("sample", type=Path, help="Output of bev1_person_presence.py")
    parser.add_argument("--model", type=Path, default=DEFAULT_MODEL)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    source = json.loads(args.sample.read_text())
    options = mp.tasks.vision.FaceDetectorOptions(
        base_options=mp.tasks.BaseOptions(
            model_asset_path=str(args.model), delegate=mp.tasks.BaseOptions.Delegate.CPU))
    rows = []
    with mp.tasks.vision.FaceDetector.create_from_options(options) as detector:
        for item in source["rows"]:
            path = Path(item["image"])
            if hashlib.sha256(path.read_bytes()).hexdigest() != item["image_sha256"]:
                raise ValueError(f"Image changed: {path}")
            frame = mp.Image(image_format=mp.ImageFormat.SRGB,
                             data=np.asarray(Image.open(path).convert("RGB")))
            found = detector.detect(frame).detections
            rows.append({"image": item["image"], "cohort": item["cohort"],
                         "face_label": item["face_label"],
                         "person_label": item["person_label"],
                         "max_face_score": max((float(d.categories[0].score) for d in found),
                                               default=0.0),
                         "face_count": len(found)})
    cohorts = sorted({row["cohort"] for row in rows})
    summary = {cohort: {str(threshold): sum(
        row["max_face_score"] >= threshold for row in rows if row["cohort"] == cohort)
        for threshold in (0.5, 0.7)} for cohort in cohorts}
    result = {"scope": __doc__.strip(),
              "sample_sha256": hashlib.sha256(args.sample.read_bytes()).hexdigest(),
              "mediapipe_version": mp.__version__,
              "face_model_sha256": hashlib.sha256(args.model.read_bytes()).hexdigest(),
              "summary_counts_at_threshold": summary,
              "rows": rows}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
