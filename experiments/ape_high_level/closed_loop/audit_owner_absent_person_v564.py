"""Matched owner-absent image diagnostic for the weak-score recovery gate."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

from PIL import Image

from audit_person_grounding_v344 import detect
from robocasa_owner_reacquire_rgbd_v560 import _model, DETECTOR


ROOT = Path(__file__).resolve().parents[3]
IMAGE = ROOT / "artifacts/ape-robocasa-first-return-track-health-v554/owner-absent-blackout-view.png"
OUT = ROOT / "artifacts/ape-robocasa-first-return-person-reacquire-v562/matched-empty-v564.json"


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    processor, model, device = _model()
    detections = detect(processor, model, device,
                        Image.open(IMAGE).convert("RGB"), "person .")
    result = {"source_sha256": sha(Path(__file__)),
              "matched_empty_image": str(IMAGE),
              "matched_empty_image_sha256": sha(IMAGE),
              "model_weights_sha256": sha(DETECTOR / "model.safetensors"),
              "person_prompt": "person .", "score_threshold": .25,
              "detections": detections}
    OUT.write_text(json.dumps(result, indent=2) + "\n")
    print(result)


if __name__ == "__main__":
    main()
